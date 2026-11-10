"""Bonsai-8B FABRIC reference — the bit-true spec the RTL copies (doc 10, R1c).

Every operation below is defined exactly (operation order, rounding, tables), so a
hardware implementation that follows it reproduces these numbers bit-for-bit.

GEMV (≈99% of the work, all of the DDR traffic) — exact integer arithmetic:
  input x (fp32) -> BFP10 per 128-group: e = smallest int with absmax <= 511*2^e,
      q = RNE(x / 2^e) in [-511, 511]
  per group: inner = sum_j sign_j * q_j            (exact, |inner| <= 65408, 17 bits)
  weight scale (fp16) = m_d * 2^e_d exactly        (m_d <= 2047, 11 bits)
  term = inner * m_d * 2^(e_d + e_x)  placed in a 64-bit two's-complement accumulator
      with LSB 2^-ACC_LSB (left shift, or arithmetic right shift = floor, if below LSB)
  y = RNE_fp32(acc) * 2^-ACC_LSB                   (one rounding, at the end)

Vector path — IEEE-754 fp32 (round-to-nearest-even) with pinned order:
  fsum: lane-strided reduction — element i goes to lane i % P (P=16), each lane sums
      sequentially in index order starting from +0, then a pairwise tree
      (lanes [0:P/2] + [P/2:P], repeat). This is what a P-lane pipelined adder does.
  rmsnorm(x, w):  ss = fsum(x*x); mean = ss * 2^-log2(n); r = rsqrt_fab(mean + eps);
      y = w * (x * r)
  rsqrt_fab: table seed on the mantissa (512+512 entries) + 2 Newton steps
      y <- y * (1.5 - (0.5*v) * (y*y))
  exp_fab(x): y = x*LOG2E; n = floor(y); f = y - n; k = RNE(f*4096);
      result = TAB[k] * 2^n  (TAB[k] = fp32(2^(k/4096)), k in 0..4096), n<-126 -> +0,
      n>127 -> +inf
  softmax over the valid positions: m = max; e_t = exp_fab(s_t - m); Z = fsum(e);
      p_t = e_t * (1/Z)
  silu(x) = x * (1 / (1 + exp_fab(-x)))
  rope: cos/sin tables fp32(cos/sin(pos*inv_freq) * yarn_attn_factor) built offline in
      float64 (the engine streams them, 512 B/token); out = [x1*c - x2*s, x2*c + x1*s]
  attention: s_t = fsum(q*k_t) * fp32(1/sqrt(hd)); ctx_d = fsum_t(p_t * v_t,d)
  KV cache: K = BFP8 with one exponent per 16 channels (after RoPE); V = BFP8 per
      (head, position) 128-vector.
  residual: x = x + y (fp32)

Golden vectors: pass a `Trace` to forward_seq / step / prefill_chunked (`trace=`) and every
intermediate above is recorded per layer and per position, including the integer side of
the GEMV (BFP10 mantissas/exponents, per-group inner / shift / term, the 64-bit
accumulator) and of the KV cache (BFP8 mantissas/exponents). Tracing only reads values the
forward already computes; with trace=None (the default) nothing is recorded.
`python -m fabric.big.ref.golden` exports a trace as $readmemh files + MANIFEST.json and
round-trips it (GOLDEN_VERDICT).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .qwen3_ref import Q1Matrix, Qwen3Ref, rope_params

P = 16                 # reduction lanes
ACC_LSB = 40           # GEMV accumulator LSB = 2^-40
GEMV_BITS = 10         # activation mantissa bits into the GEMV
KV_BITS = 8
EXP_K = 4096
LOG2E = np.float32(1.4426950408889634)


# ----------------------------------------------------------------------------- helpers
def pow2_f32(n: torch.Tensor) -> torch.Tensor:
    """exact fp32 2^n for integer n in [-126, 127] via bit construction."""
    return ((n.to(torch.int32) + 127) << 23).view(torch.float32)


def fsum(x: torch.Tensor) -> torch.Tensor:
    """Lane-strided fp32 reduction over the last dim (see module doc)."""
    n = x.shape[-1]
    pad = (-n) % P
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    xv = x.reshape(*x.shape[:-1], -1, P)
    acc = torch.zeros(*x.shape[:-1], P, dtype=torch.float32, device=x.device)
    for i in range(xv.shape[-2]):
        acc = acc + xv[..., i, :]
    w = P
    while w > 1:
        w //= 2
        acc = acc[..., :w] + acc[..., w:2 * w]
    return acc[..., 0]


def bfp_quant(x: torch.Tensor, bits: int, group: int):
    """x [..., n] fp32 -> (q int-valued float64 [..., n], e int64 [..., n/group])."""
    qmax = 2 ** (bits - 1) - 1
    sh = x.shape
    xg = x.reshape(*sh[:-1], sh[-1] // group, group).double()
    am = xg.abs().amax(-1)
    m, E = torch.frexp(am)
    e = torch.where(m * 2 ** (bits - 1) <= qmax, E - (bits - 1), E - (bits - 2)).to(torch.int64)
    e = torch.where(am == 0, torch.zeros_like(e), e)
    q = torch.clamp(torch.round(xg * torch.exp2(-e.double())[..., None]), -qmax, qmax)
    return q.reshape(sh), e


def bfp_dequant(q, e, group):
    sh = q.shape
    qg = q.reshape(*sh[:-1], sh[-1] // group, group)
    return (qg * torch.exp2(e.double())[..., None]).reshape(sh).to(torch.float32)


def fp16_decompose(d: torch.Tensor):
    """fp16 tensor -> (m int64 signed, e int64) with d == m * 2^e exactly."""
    b = d.view(torch.int16).to(torch.int64) & 0xFFFF
    sgn = (b >> 15) & 1
    ex = (b >> 10) & 0x1F
    fr = b & 0x3FF
    m = torch.where(ex == 0, fr, fr | 0x400)
    e = torch.where(ex == 0, torch.full_like(ex, -24), ex - 25)
    return torch.where(sgn == 1, -m, m), e


class GemvStats:
    def __init__(self):
        self.max_shift = -999
        self.min_shift = 999
        self.max_acc_bits = 0
        self.floor_terms = 0
        self.terms = 0


# ----------------------------------------------------------------------------- trace
class Trace:
    """Golden-vector recorder (exported by golden.py). The default path never builds one.

        tr = Trace(layers={0, 35}, positions=None, rows=(0, 64), partials={"l0.q": (0, 64)})
        fab.forward_seq(toks, trace=tr)     # or step(tok, trace=tr) / prefill_chunked(..)
        tr["l0.q.xq"]                        # CPU tensor, dim 0 = traced position (tr.pos)

    layers / positions: sets of ints, None = all; final = also trace out_norm / lm_head.
    Every signal is stored position-major (dim 0 = query position, in call order) except
    the KV cache rows (k_q / k_e / v_q / v_e), which are kept for EVERY position of a traced
    layer (tr.cache_pos), because attention at a traced position reads the whole history.
    rows = the output-row window [r0, r1) for which every traced GEMV records its 64-bit
    accumulator; partials maps a GEMV name ("l0.q") to the row window that also gets the
    per-group inner / prod / sh / term / accp. Signal names: the taps below; widths and
    file formats: SIG / GEMV_SIG in golden.py.
    """

    def __init__(self, layers=None, positions=None, rows=(0, 64), partials=None, final=True):
        self.layers = None if layers is None else set(layers)
        self.positions = None if positions is None else set(positions)
        self.rows, self.partials, self.final = rows, dict(partials or {}), final
        self.sig, self.fill = {}, {}
        self.pos, self.cache_pos, self._sel, self._all = [], [], [], True

    def begin(self, pos):
        """Start one _layers call over absolute positions `pos` [Tq]."""
        pl = [int(p) for p in pos]
        self._sel = [i for i, p in enumerate(pl) if self.positions is None or p in self.positions]
        self._all = len(self._sel) == len(pl)
        self.pos += [pl[i] for i in self._sel]
        self.cache_pos += pl

    def on(self, L):
        return self.layers is None or L in self.layers

    def window(self, name, n_out):
        """(r0, r1, partials?) for GEMV `name`, clipped to its n_out rows."""
        part = name in self.partials
        r0, r1 = self.partials[name] if part else (self.rows or (0, 0))
        return r0, min(r1, n_out), part

    def tap(self, name, t, tdim=0, fill=None, cache=False):
        """Record t with its position axis `tdim` moved first. fill = pad value for a ragged
        last axis (attention over a growing history); cache = keep all positions."""
        t = t.detach().movedim(tdim, 0)
        if not cache:
            if not self._sel:
                return
            if not self._all:
                t = t[torch.as_tensor(self._sel, device=t.device)]
        self.sig.setdefault(name, []).append(t.to("cpu", copy=True))
        if fill is not None:
            self.fill[name] = fill

    def __contains__(self, name):
        return name in self.sig

    def __getitem__(self, name):
        ch = self.sig[name]
        if name in self.fill:
            n = max(c.shape[-1] for c in ch)
            ch = [torch.nn.functional.pad(c, (0, n - c.shape[-1]), value=self.fill[name]) for c in ch]
        return torch.cat(ch)

    def names(self):
        return list(self.sig)


# ----------------------------------------------------------------------------- GEMV
class ExactBfpGemv:
    """Bit-true 1-bit x BFP10 GEMV (see module doc). Weights must be Q1Matrix."""
    ELEMS = 8 * 1024 * 1024    # cap on G*T*R per int64 temporary (~64 MB)

    def __init__(self):
        self.stats = GemvStats()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self._dec = {}

    def _scales(self, W: Q1Matrix):
        k = id(W)
        if k not in self._dec:
            self._dec[k] = fp16_decompose(W.scales)       # [R, G] each
        return self._dec[k]

    def mm(self, W, X: torch.Tensor, name: str = "", tr=None) -> torch.Tensor:
        """Y [T, out]. tr (a Trace) taps name.xq/.xe/.y, .acc over the trace's row window
        and, for a partials matrix, .inner/.prod/.sh/.term/.accp as [T, row, group]."""
        assert isinstance(W, Q1Matrix), name
        T, n = X.shape
        G = n // 128
        q, ex = bfp_quant(X, GEMV_BITS, 128)                # q [T, n], ex [T, G]
        Xq = q.to(torch.float32).view(T, G, 128).transpose(0, 1)   # [G, T, 128] exact ints
        md, ed = self._scales(W)
        st = self.stats
        out = []
        w0 = w1 = 0
        if tr is not None:
            tr.tap(name + ".xq", q)
            tr.tap(name + ".xe", ex)
            w0, w1, part = tr.window(name, W.out)
            pc = {k: [] for k in ("acc", "inner", "prod", "sh", "term")}
        rows = max(16, min(4096, 1 << ((self.ELEMS // (G * T)).bit_length() - 1)))
        for r0 in range(0, W.out, rows):
            r1 = min(r0 + rows, W.out)
            S = W.sign_pm(r0, r1).view(r1 - r0, G, 128).permute(1, 2, 0)   # [G, 128, R]
            inner = torch.bmm(Xq, S).round().to(torch.int64)                 # [G, T, R] exact
            term = inner * md[r0:r1].T[:, None, :]                           # [G, T, R]
            sh = ed[r0:r1].T[:, None, :] + ex.T[:, :, None] + ACC_LSB       # [G, T, R]
            lo, hi = max(w0, r0) - r0, min(w1, r1) - r0                      # trace window
            if lo < hi and part:
                for k, t in (("inner", inner), ("prod", term), ("sh", sh)):
                    pc[k].append(t[..., lo:hi].clone())
            st.max_shift = max(st.max_shift, int(sh.max()))
            st.min_shift = min(st.min_shift, int(sh.min()))
            nfl = int((sh < 0).sum())
            st.floor_terms += nfl
            st.terms += sh.numel()
            if nfl:
                term = torch.where(sh >= 0, torch.bitwise_left_shift(term, sh.clamp(min=0)),
                                   torch.bitwise_right_shift(term, (-sh).clamp(min=0)))
            else:
                term = torch.bitwise_left_shift(term, sh)
            del sh
            acc = term.sum(0)                                                 # [T, R] int64
            if lo < hi:
                pc["acc"].append(acc[:, lo:hi].clone())
                if part:
                    pc["term"].append(term[..., lo:hi].clone())
            st.max_acc_bits = max(st.max_acc_bits, int(acc.abs().max()).bit_length())
            out.append(acc.to(torch.float32) * pow2_f32(torch.tensor(-ACC_LSB)).to(X.device))
        y = torch.cat(out, 1)
        if tr is not None:
            tr.tap(name + ".y", y)
            for k, v in pc.items():
                if v:                                         # acc [T, R]; partials [T, R, G]
                    tr.tap(name + "." + k, torch.cat(v, -1) if k == "acc" else torch.cat(v, -1).permute(1, 2, 0))
            if pc["term"]:                                    # accumulator after each group
                tr.tap(name + ".accp", torch.cat(pc["term"], -1).permute(1, 2, 0).cumsum(-1))
        return y

    def __call__(self, W, x, name=""):
        return self.mm(W, x[None], name)[0]


# ----------------------------------------------------------------------------- tables
def exp_table(device):
    k = np.arange(EXP_K + 1, dtype=np.float64)
    return torch.from_numpy((2.0 ** (k / EXP_K)).astype(np.float32)).to(device)


def rsqrt_tables(device):
    lo = 0.5 + (np.arange(512) + 0.5) / 1024       # m in [0.5, 1), 512 bins
    hi = 1.0 + (np.arange(512) + 0.5) / 512        # m in [1, 2), 512 bins
    return (torch.from_numpy((1 / np.sqrt(lo)).astype(np.float32)).to(device),
            torch.from_numpy((1 / np.sqrt(hi)).astype(np.float32)).to(device))


# ----------------------------------------------------------------------------- model
class FabricRef(Qwen3Ref):
    def __init__(self, cfg, W, device="cuda", max_pos=4096):
        super().__init__(cfg, W, device, gemv=ExactBfpGemv())
        self.EXP = exp_table(device)
        self.RS_LO, self.RS_HI = rsqrt_tables(device)
        inv, att = rope_params(cfg)
        ang = np.arange(max_pos, dtype=np.float64)[:, None] * inv.double().numpy()[None]
        self.COS = torch.from_numpy((np.cos(ang) * att).astype(np.float32)).to(device)
        self.SIN = torch.from_numpy((np.sin(ang) * att).astype(np.float32)).to(device)
        self.eps32 = torch.tensor(np.float32(cfg.eps), device=device)
        self.inv_sqrt_hd = torch.tensor(np.float32(1 / math.sqrt(cfg.head_dim)), device=device)

    # --- transcendentals ---
    def exp_fab(self, x):
        y = x * LOG2E.item()
        n = torch.floor(y)
        f = y - n
        k = torch.round(f * EXP_K).to(torch.int64).clamp(0, EXP_K)
        r = self.EXP[k] * pow2_f32(n.clamp(-126, 127))
        r = torch.where(n < -126, torch.zeros_like(r), r)
        return torch.where(n > 127, torch.full_like(r, float("inf")), r)

    def rsqrt_fab(self, v):
        m, E = torch.frexp(v)                      # v = m 2^E, m in [0.5,1)
        odd = (E % 2) != 0
        m2 = torch.where(odd, m * 2, m)            # [1,2) if odd else [0.5,1)
        E2 = torch.where(odd, E - 1, E)            # even
        ilo = ((m2 - 0.5) * 1024).floor().to(torch.int64).clamp(0, 511)
        ihi = ((m2 - 1.0) * 512).floor().to(torch.int64).clamp(0, 511)
        y = torch.where(odd, self.RS_HI[ihi], self.RS_LO[ilo]) * pow2_f32(-(E2 // 2))
        h = v * 0.5
        for _ in range(2):
            y = y * (1.5 - h * (y * y))
        return y

    # --- vector ops (override the float reference) ---
    def rmsnorm(self, x, w, tr=None, name="", tdim=0):
        n = x.shape[-1]
        ss = fsum(x * x)
        mean = ss * pow2_f32(torch.tensor(-int(math.log2(n)), device=x.device))
        r = self.rsqrt_fab(mean + self.eps32)
        y = w * (x * r[..., None])
        if tr is not None:
            for k, t in ((".ss", ss), (".mean", mean), (".r", r), ("", y)):
                tr.tap(name + k, t, tdim)
        return y

    def silu(self, x):
        return x * (1.0 / (1.0 + self.exp_fab(-x)))

    def rope_tab(self, x, pos, tr=None, name=""):   # x [..., T, hd]; pos LongTensor [T]
        hd = x.shape[-1]
        c, s = self.COS[pos], self.SIN[pos]
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        y = torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)
        if tr is not None:
            tr.tap(name, y, y.dim() - 2)
        return y

    # KV formats: (bits, group) per tensor; returned dequantised (exact in fp32)
    K_FMT = (KV_BITS, 16)     # R1c: Qwen post-RoPE K outliers need an exponent per 16 channels
    V_FMT = (KV_BITS, 128)

    def kv_q(self, x, fmt, tr=None, name=""):       # x [n_kv, T, hd]
        if fmt is None:
            return x
        bits, grp = fmt
        q, e = bfp_quant(x, bits, grp)
        if tr is not None:                          # the cache as stored: mantissas + exponents
            tr.tap(name + "_q", q, 1, cache=True)
            tr.tap(name + "_e", e, 1, cache=True)
        return bfp_dequant(q, e, grp)

    def attend(self, q, K, V, valid, tr=None, name=""):
        """q [H, Tq, hd]; K,V [H, Tk, hd] (already repeated to H); valid [Tq, Tk] bool."""
        H, Tq, hd = q.shape
        Tk = K.shape[1]
        # scores: lane-strided over d, chunks of P dims
        acc = torch.zeros(H, Tq, Tk, P, device=q.device)
        for j in range(hd // P):
            acc = acc + q[:, :, None, j * P:(j + 1) * P] * K[:, None, :, j * P:(j + 1) * P]
        w = P
        while w > 1:
            w //= 2
            acc = acc[..., :w] + acc[..., w:2 * w]
        s = acc[..., 0] * self.inv_sqrt_hd
        s = torch.where(valid[None], s, torch.full_like(s, float("-inf")))
        m = s.amax(-1, keepdim=True)
        e = torch.where(valid[None], self.exp_fab(s - m), torch.zeros_like(s))
        Z = fsum(e)
        rz = 1.0 / Z
        p = e * rz[..., None]
        if tr is not None:                  # [pos, head, t]; masked t: s = -inf, e = p = +0
            tr.tap(name + "att.s", s, 1, fill=float("-inf"))
            tr.tap(name + "att.m", m[..., 0], 1)
            tr.tap(name + "att.e", e, 1, fill=0.0)
            tr.tap(name + "att.z", Z, 1)
            tr.tap(name + "att.rz", rz, 1)
            tr.tap(name + "att.p", p, 1, fill=0.0)
        # ctx: lane-strided over positions t (lane = t % P), sequential per lane
        pad = (-Tk) % P
        if pad:
            p = torch.nn.functional.pad(p, (0, pad))
            V = torch.nn.functional.pad(V, (0, 0, 0, pad))
        cacc = torch.zeros(H, Tq, P, hd, device=q.device)
        for j in range(p.shape[-1] // P):
            cacc = cacc + p[:, :, j * P:(j + 1) * P, None] * V[:, None, j * P:(j + 1) * P, :]
        w = P
        while w > 1:
            w //= 2
            cacc = cacc[:, :, :w] + cacc[:, :, w:2 * w]
        if tr is not None:
            tr.tap(name + "ctx", cacc[:, :, 0], 1)
        return cacc[:, :, 0]                                      # [H, Tq, hd]

    def embed_rows(self, toks):
        E = self.W["embed"]
        return torch.stack([E.dequant(int(t), int(t) + 1)[0] for t in toks]).to(torch.float32)

    # --- shared layer body: positions pos [Tq], with K/V history supplied ---
    def _layers(self, x, pos, kv_hist, valid, trace=None):
        c, W, g = self.c, self.W, self.gemv
        Tq = x.shape[0]
        rep = c.n_head // c.n_kv
        new_kv = []
        if trace is not None:
            trace.begin(pos)
            trace.tap("embed", x)
        for L in range(c.n_layer):
            p = f"l{L}."
            tr = trace if trace is not None and trace.on(L) else None
            if tr is not None:
                tr.tap(p + "x_in", x)
            h = self.rmsnorm(x, W[p + "attn_norm"], tr, p + "attn_norm")
            q = g.mm(W[p + "q"], h, p + "q", tr).view(Tq, c.n_head, c.head_dim).transpose(0, 1)
            k = g.mm(W[p + "k"], h, p + "k", tr).view(Tq, c.n_kv, c.head_dim).transpose(0, 1)
            v = g.mm(W[p + "v"], h, p + "v", tr).view(Tq, c.n_kv, c.head_dim).transpose(0, 1)
            q = self.rope_tab(self.rmsnorm(q, W[p + "q_norm"], tr, p + "q_norm", 1), pos, tr, p + "q_rope")
            k = self.rope_tab(self.rmsnorm(k, W[p + "k_norm"], tr, p + "k_norm", 1), pos, tr, p + "k_rope")
            k = self.kv_q(k, self.K_FMT, tr, p + "k")
            v = self.kv_q(v, self.V_FMT, tr, p + "v")
            if kv_hist is not None:
                Kc = torch.cat([kv_hist[L][0], k], 1)
                Vc = torch.cat([kv_hist[L][1], v], 1)
            else:
                Kc, Vc = k, v
            new_kv.append((Kc, Vc))
            ctx = self.attend(q, Kc.repeat_interleave(rep, 0), Vc.repeat_interleave(rep, 0), valid, tr, p)
            x = x + g.mm(W[p + "o"], ctx.transpose(0, 1).reshape(Tq, -1), p + "o", tr)
            if tr is not None:
                tr.tap(p + "x_mid", x)
            h = self.rmsnorm(x, W[p + "ffn_norm"], tr, p + "ffn_norm")
            gt = g.mm(W[p + "gate"], h, p + "gate", tr)
            up = g.mm(W[p + "up"], h, p + "up", tr)
            sg = self.silu(gt)
            act = sg * up
            if tr is not None:
                tr.tap(p + "silu", sg)
                tr.tap(p + "act", act)
            x = x + g.mm(W[p + "down"], act, p + "down", tr)
            if tr is not None:
                tr.tap(p + "x_out", x)
        tr = trace if trace is not None and trace.final else None
        if tr is not None:
            tr.tap("x_final", x)
        h = self.rmsnorm(x, W["out_norm"], tr, "out_norm")
        lg = g.mm(W["lm_head"], h, "lm_head", tr)
        if tr is not None:                  # argmax = the LOWEST index holding the maximum
            i = torch.arange(lg.shape[-1], device=lg.device)
            tr.tap("argmax", torch.where(lg == lg.amax(-1, keepdim=True), i, lg.shape[-1]).amin(-1))
        return lg, new_kv

    @torch.no_grad()
    def forward_seq(self, toks, trace=None):
        T = len(toks)
        pos = torch.arange(T, device=self.dev)
        valid = torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()
        lg, _ = self._layers(self.embed_rows(toks), pos, None, valid, trace)
        return lg

    @torch.no_grad()
    def prefill_chunked(self, toks, chunk=64, out_device="cpu", trace=None):
        """forward_seq in query chunks of `chunk` positions against the growing K/V history.

        Bit-identical to forward_seq (masked positions contribute p = +0, and the lane of
        position t is t % P whatever the history length), but the attention temporaries are
        [H, chunk, T, P] instead of [H, T, T, P], so T = 1-2k fits a 4 GB card. Logits are
        moved to `out_device` per chunk ([T, vocab] fp32 is 1.2 GB at T = 2048).
        """
        T = len(toks)
        kv, outs = None, []
        for t0 in range(0, T, chunk):
            t1 = min(t0 + chunk, T)
            pos = torch.arange(t0, t1, device=self.dev)
            valid = torch.arange(t1, device=self.dev)[None, :] <= pos[:, None]
            lg, kv = self._layers(self.embed_rows(toks[t0:t1]), pos, kv, valid, trace)
            outs.append(lg.to(out_device))
        return torch.cat(outs)

    def reset(self):
        super().reset()
        self.kv = None

    @torch.no_grad()
    def step(self, tok: int, trace=None):
        pos = torch.tensor([self.pos], device=self.dev)
        valid = torch.ones(1, self.pos + 1, dtype=torch.bool, device=self.dev)
        lg, self.kv = self._layers(self.embed_rows([tok]), pos, self.kv, valid, trace)
        self.pos += 1
        return lg[0]
