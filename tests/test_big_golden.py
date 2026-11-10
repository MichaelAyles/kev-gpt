"""Big-Kev golden vectors (fabric/big/ref/golden.py) on a tiny random 1-bit model: the trace
does not perturb the forward, step / chunked / seq traces agree, and the .mem round-trip
verifies (and catches a corrupted vector). No model weights needed; runs in seconds.

    pytest tests/test_big_golden.py        # or, without pytest:
    python tests/test_big_golden.py
"""
import functools
import sys
import tempfile
from pathlib import Path

try:
    import torch
except ImportError:                     # no torch here: skip under pytest
    import pytest
    pytest.skip("needs torch", allow_module_level=True)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fabric.big.ref import golden  # noqa: E402
from fabric.big.ref.fabric_ref import FabricRef, Trace  # noqa: E402
from fabric.big.ref.qwen3_ref import Cfg, Q1Matrix  # noqa: E402

T = 19          # > P = 16, so the lane-strided reductions over positions wrap


@functools.lru_cache(None)
def _model():
    g = torch.Generator().manual_seed(0)
    c = Cfg(n_layer=2, d=256, ffn=384, n_head=4, n_kv=2, head_dim=128, vocab=320,
            rope_theta=1e6, eps=1e-6, tied=True)

    def mat(o, i, s=1.0):      # ~10% negative scales; s = 1e-4 gives fp16 subnormals
        sc = (torch.rand(o, i // 128, generator=g) * 0.08 + 0.01) * (1 - 2 * (torch.rand(o, i // 128, generator=g) < 0.1)) * s
        return Q1Matrix(torch.randint(0, 256, (o, i // 8), generator=g, dtype=torch.uint8), sc.to(torch.float16))

    def vec(n):
        return torch.rand(n, generator=g) + 0.5

    W = {"embed": mat(c.vocab, c.d), "out_norm": vec(c.d)}
    W["lm_head"] = W["embed"]
    for L in range(c.n_layer):
        p = f"l{L}."
        W.update({p + "q": mat(c.n_head * c.head_dim, c.d), p + "k": mat(c.n_kv * c.head_dim, c.d),
                  p + "v": mat(c.n_kv * c.head_dim, c.d), p + "o": mat(c.d, c.n_head * c.head_dim),
                  p + "gate": mat(c.ffn, c.d, 1e-4 ** L), p + "up": mat(c.ffn, c.d, 1e-4 ** L),
                  p + "down": mat(c.d, c.ffn),      # layer 1: tiny act -> terms below the LSB (floor)
                  p + "attn_norm": vec(c.d), p + "ffn_norm": vec(c.d),
                  p + "q_norm": vec(c.head_dim), p + "k_norm": vec(c.head_dim)})
    toks = torch.randint(0, c.vocab, (T,), generator=g).tolist()
    return FabricRef(c, W, device="cpu", max_pos=64), toks


def _trace(**kw):
    return Trace(rows=(8, 40), partials={"l0.q": (0, 64), "l1.down": (250, 256)}, **kw)


def test_trace_is_passive_and_mode_independent():
    fab, toks = _model()
    ref = fab.forward_seq(toks)
    tr = _trace()
    assert torch.equal(fab.forward_seq(toks, trace=tr), ref)
    assert tr.pos == tr.cache_pos == list(range(T))
    ts, tc = _trace(), _trace()
    fab.reset()
    assert torch.equal(torch.stack([fab.step(t, trace=ts) for t in toks]), ref)
    assert torch.equal(fab.prefill_chunked(toks, chunk=7, trace=tc), ref)
    assert tr.names() == ts.names() == tc.names()
    for n in tr.names():
        assert golden.beq(tr[n], ts[n]) and golden.beq(tr[n], tc[n]), n
    assert golden.beq(tr["l0.q.accp"][..., -1], tr["l0.q.acc"])
    assert tr["l1.down.term"].shape == (T, 6, 384 // 128) and tr["l1.q.acc"].shape == (T, 32)
    assert int(tr["l1.down.sh"].max()) < 0 < int(tr["l0.q.sh"].min())      # both shift directions traced
    small = _trace()                    # a row window that spans several GEMV row chunks
    fab.gemv.ELEMS = 4 * T              # -> 16-row chunks
    try:
        assert torch.equal(fab.forward_seq(toks, trace=small), ref)
    finally:
        del fab.gemv.ELEMS
    assert all(golden.beq(tr[n], small[n]) for n in tr.names())


def test_layer_and_position_selection():
    fab, toks = _model()
    full, sub = _trace(), _trace(layers={1}, positions={3, 17})
    fab.forward_seq(toks, trace=full)
    fab.forward_seq(toks, trace=sub)
    assert sub.pos == [3, 17] and not any(n.startswith("l0.") for n in sub.names())
    for n in sub.names():
        want = full[n] if n.rsplit(".", 1)[-1] in ("k_q", "k_e", "v_q", "v_e") else full[n][[3, 17]]
        assert golden.beq(sub[n], want), n


def test_roundtrip_and_corruption(tmp_path):
    fab, toks = _model()
    sub = tmp_path / "sub"              # a layer / position subset, no final stage
    golden.dump(fab, toks, sub, layers=[1], positions=[3, 17], rows=None, partials={"l1.k": (5, 9)}, final=False)
    assert golden.verify(sub)
    # every row of every GEMV exported: both layers recomputable end to end from the files
    man = golden.dump(fab, toks, tmp_path, layers=[0, 1], rows=(0, 1 << 30),
                      partials={"l0.q": (0, 64), "l1.down": (250, 256)}, selfcheck=True)
    assert golden.verify(tmp_path)
    by = {e["signal"]: e for e in man["files"]}
    assert by["l0.q.xq"]["hex_digits"] == 3 and by["l0.q.inner"]["hex_digits"] == 5
    assert by["l0.q.acc"]["hex_digits"] == 16 and by["l0.x_in"]["hex_digits"] == 8
    for sig in ("l0.q.xq", "l1.att.p", "l0.k_q", "l1.down.w.sign", "tab.exp"):
        f = tmp_path / by[sig]["file"]
        good = f.read_bytes()
        i = (len(good) // 2 // (by[sig]["hex_digits"] + 1)) * (by[sig]["hex_digits"] + 1) + by[sig]["hex_digits"] - 1
        f.write_bytes(good[:i] + (b"1" if good[i:i + 1] != b"1" else b"2") + good[i + 1:])
        assert not golden.verify(tmp_path), sig
        f.write_bytes(good)
    assert golden.verify(tmp_path)


if __name__ == "__main__":
    test_trace_is_passive_and_mode_independent()
    test_layer_and_position_selection()
    with tempfile.TemporaryDirectory() as d:
        test_roundtrip_and_corruption(Path(d))
    print("test_big_golden OK")
