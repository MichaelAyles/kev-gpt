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

    def mm(self, W, X: torch.Tensor, name: str = "") -> torch.Tensor:
        assert isinstance(W, Q1Matrix), name
        T, n = X.shape
        G = n // 128
        q, ex = bfp_quant(X, GEMV_BITS, 128)                # q [T, n], ex [T, G]
        Xq = q.to(torch.float32).view(T, G, 128).transpose(0, 1)   # [G, T, 128] exact ints
        md, ed = self._scales(W)
        st = self.stats
        out = []
        rows = max(16, min(4096, 1 << ((self.ELEMS // (G * T)).bit_length() - 1)))
        for r0 in range(0, W.out, rows):
            r1 = min(r0 + rows, W.out)
            S = W.sign_pm(r0, r1).view(r1 - r0, G, 128).permute(1, 2, 0)   # [G, 128, R]
            inner = torch.bmm(Xq, S).round().to(torch.int64)                 # [G, T, R] exact
            term = inner * md[r0:r1].T[:, None, :]                           # [G, T, R]
            sh = ed[r0:r1].T[:, None, :] + ex.T[:, :, None] + ACC_LSB       # [G, T, R]
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
            st.max_acc_bits = max(st.max_acc_bits, int(acc.abs().max()).bit_length())
            out.append(acc.to(torch.float32) * pow2_f32(torch.tensor(-ACC_LSB)).to(X.device))
        return torch.cat(out, 1)

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
    def rmsnorm(self, x, w):
        n = x.shape[-1]
        ss = fsum(x * x)
        mean = ss * pow2_f32(torch.tensor(-int(math.log2(n)), device=x.device))
        r = self.rsqrt_fab(mean + self.eps32)
        return w * (x * r[..., None])

    def silu(self, x):
        return x * (1.0 / (1.0 + self.exp_fab(-x)))

    def rope_tab(self, x, pos):            # x [..., T, hd]; pos LongTensor [T]
        hd = x.shape[-1]
        c, s = self.COS[pos], self.SIN[pos]
        x1, x2 = x[..., : hd // 2], x[..., hd // 2:]
        return torch.cat([x1 * c - x2 * s, x2 * c + x1 * s], -1)

    # KV formats: (bits, group) per tensor; returned dequantised (exact in fp32)
    K_FMT = (KV_BITS, 16)     # R1c: Qwen post-RoPE K outliers need an exponent per 16 channels
    V_FMT = (KV_BITS, 128)

    def kv_q(self, x, fmt):
        if fmt is None:
            return x
        bits, grp = fmt
        q, e = bfp_quant(x, bits, grp)
        return bfp_dequant(q, e, grp)

    def attend(self, q, K, V, valid):
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
        p = e * (1.0 / Z)[..., None]
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
        return cacc[:, :, 0]                                      # [H, Tq, hd]

    def embed_rows(self, toks):
        E = self.W["embed"]
        return torch.stack([E.dequant(int(t), int(t) + 1)[0] for t in toks]).to(torch.float32)

    # --- shared layer body: positions pos [Tq], with K/V history supplied ---
    def _layers(self, x, pos, kv_hist, valid):
        c, W, g = self.c, self.W, self.gemv
        Tq = x.shape[0]
        rep = c.n_head // c.n_kv
        new_kv = []
        for L in range(c.n_layer):
            p = f"l{L}."
            h = self.rmsnorm(x, W[p + "attn_norm"])
            q = g.mm(W[p + "q"], h, p + "q").view(Tq, c.n_head, c.head_dim).transpose(0, 1)
            k = g.mm(W[p + "k"], h, p + "k").view(Tq, c.n_kv, c.head_dim).transpose(0, 1)
            v = g.mm(W[p + "v"], h, p + "v").view(Tq, c.n_kv, c.head_dim).transpose(0, 1)
            q = self.rope_tab(self.rmsnorm(q, W[p + "q_norm"]), pos)
            k = self.kv_q(self.rope_tab(self.rmsnorm(k, W[p + "k_norm"]), pos), self.K_FMT)
            v = self.kv_q(v, self.V_FMT)
            if kv_hist is not None:
                Kc = torch.cat([kv_hist[L][0], k], 1)
                Vc = torch.cat([kv_hist[L][1], v], 1)
            else:
                Kc, Vc = k, v
            new_kv.append((Kc, Vc))
            ctx = self.attend(q, Kc.repeat_interleave(rep, 0), Vc.repeat_interleave(rep, 0), valid)
            x = x + g.mm(W[p + "o"], ctx.transpose(0, 1).reshape(Tq, -1), p + "o")
            h = self.rmsnorm(x, W[p + "ffn_norm"])
            gt = g.mm(W[p + "gate"], h, p + "gate")
            up = g.mm(W[p + "up"], h, p + "up")
            x = x + g.mm(W[p + "down"], self.silu(gt) * up, p + "down")
        h = self.rmsnorm(x, W["out_norm"])
        return g.mm(W["lm_head"], h, "lm_head"), new_kv

    @torch.no_grad()
    def forward_seq(self, toks):
        T = len(toks)
        pos = torch.arange(T, device=self.dev)
        valid = torch.ones(T, T, dtype=torch.bool, device=self.dev).tril()
        lg, _ = self._layers(self.embed_rows(toks), pos, None, valid)
        return lg

    def reset(self):
        super().reset()
        self.kv = None

    @torch.no_grad()
    def step(self, tok: int, trace=None):
        pos = torch.tensor([self.pos], device=self.dev)
        valid = torch.ones(1, self.pos + 1, dtype=torch.bool, device=self.dev)
        lg, self.kv = self._layers(self.embed_rows([tok]), pos, self.kv, valid)
        self.pos += 1
        return lg[0]
