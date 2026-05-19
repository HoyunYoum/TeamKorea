#!/usr/bin/env python3
"""Synthetic benchmark generator for pattern_solver stress-testing.

Emits competition-format JSON problems mirroring common ML/LLM topologies:
  - MLP: linear classifier chain (MM + activation repeated)
  - FFN: (up-MM + GELU + down-MM) stack (transformer-style)
  - Attention: Q·K^T + softmax + scores·V per head, parallel
  - Transformer block: attention + FFN + residual adds + norms

Each scenario is parameterized by:
  - Model dims (d_model, d_ff, n_heads, seq_len, depth)
  - Hardware (capacity, bandwidth, native [w,h])

Capacity variants (tight / medium / loose) reveal:
  - tight: forces split-K, boundary decomposition, retention pressure
  - medium: typical working-set pressure with some fusion opportunities
  - loose: everything fits — baseline achievability

Usage:
  python synthetic/gen_bench.py <config.yaml | scenario_name> [--out <path>]
  python synthetic/gen_bench.py --list    # show available scenarios
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


# ── Cost Model (calibrated against released BMs) ─────────────────────────────
#
# PROBLEM.md `base_cost` is the per-NATIVE-TILE execution cost — NOT the full
# output FLOPs. Released BMs use base_cost ∈ [200..2000] regardless of tensor
# size (Ex1B: 128×128 MM = 1000; Ex5B: 128×128 MM = 2000). compute_lb formula
# multiplies base_cost by num_tiles_at_native, so a per-tile constant matches
# the lower-bound's intent. Cost functions here use NATIVE_W × NATIVE_H × K
# for MMs (proportional to per-tile FLOPs) and a small constant for PWs.
NATIVE = 128

def mm_cost(h: int, w: int, k: int, native: int = NATIVE) -> int:
    """Per-native-tile MM cost: native × native × k FLOPs / 1024 (rough scale).
    h, w of the *output* are unused — base_cost is per-tile, not per-output."""
    del h, w  # unused
    return max(1, (native * native * k) // 1024)


def pw_cost(h: int, w: int, native: int = NATIVE) -> int:
    """Per-native-tile PW cost: native × native FLOPs / 512 (rough constant)."""
    del h, w  # unused
    return max(1, (native * native) // 512)


# ── Graph Builder ────────────────────────────────────────────────────────────

@dataclass
class Graph:
    widths: List[int] = field(default_factory=list)
    heights: List[int] = field(default_factory=list)
    inputs: List[List[int]] = field(default_factory=list)   # per-op input tensor ids
    outputs: List[List[int]] = field(default_factory=list)  # per-op output tensor ids
    op_types: List[str] = field(default_factory=list)
    base_costs: List[int] = field(default_factory=list)

    def tensor(self, w: int, h: int) -> int:
        """Allocate a tensor, return its index."""
        self.widths.append(w)
        self.heights.append(h)
        return len(self.widths) - 1

    def op(self, op_type: str, inputs: List[int], output: int, cost: int) -> int:
        self.op_types.append(op_type)
        self.inputs.append(list(inputs))
        self.outputs.append([output])
        self.base_costs.append(cost)
        return len(self.op_types) - 1

    def matmul(self, lhs: int, rhs: int, cost: Optional[int] = None) -> int:
        """lhs: (h × k), rhs: (k × w) → output (h × w)."""
        h = self.heights[lhs]
        k = self.widths[lhs]
        assert k == self.heights[rhs], f"MatMul shape: lhs.width {k} != rhs.height {self.heights[rhs]}"
        w = self.widths[rhs]
        out = self.tensor(w, h)
        c = cost if cost is not None else mm_cost(h, w, k)
        self.op("MatMul", [lhs, rhs], out, c)
        return out

    def pointwise(self, ins: List[int], out_h: Optional[int] = None,
                  out_w: Optional[int] = None, cost: Optional[int] = None) -> int:
        """All inputs share spatial shape; output same shape."""
        if out_h is None:
            out_h = self.heights[ins[0]]
        if out_w is None:
            out_w = self.widths[ins[0]]
        for t in ins:
            assert self.heights[t] == out_h and self.widths[t] == out_w, \
                f"PW shape: tensor {t} {self.widths[t]}x{self.heights[t]} != {out_w}x{out_h}"
        out = self.tensor(out_w, out_h)
        c = cost if cost is not None else pw_cost(out_h, out_w)
        self.op("Pointwise", ins, out, c)
        return out

    def to_dict(self, capacity: int, bandwidth: int,
                native: tuple = (128, 128)) -> dict:
        return {
            "widths": self.widths,
            "heights": self.heights,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "base_costs": self.base_costs,
            "op_types": self.op_types,
            "fast_memory_capacity": capacity,
            "slow_memory_bandwidth": bandwidth,
            "native_granularity": list(native),
        }


# ── High-level Model Builders ────────────────────────────────────────────────

def build_mlp(g: Graph, x: int, depth: int, d_model: int,
              activation: bool = True) -> int:
    """MLP: (Linear → ReLU) × depth."""
    for _ in range(depth):
        w = g.tensor(d_model, d_model)  # weight (graph input)
        x = g.matmul(x, w)
        if activation:
            x = g.pointwise([x])  # ReLU / GELU
    return x


def build_ffn(g: Graph, x: int, d_model: int, d_ff: int) -> int:
    """FFN: up-proj → GELU → down-proj (single block)."""
    w_up = g.tensor(d_ff, d_model)       # (d_model × d_ff) — wait, MatMul convention
    # lhs h×k @ rhs k×w → out h×w. x is (batch=h, d_model=k).
    # w_up shape: (k=d_model, w=d_ff). So widths=d_ff, heights=d_model? Let me re-verify.
    # LHS.width = k (reduction) = LHS.height of MM? No, LHS.width = reduction.
    # In graph: widths=[d_ff], heights=[d_model] for w_up means width=d_ff, height=d_model.
    # MatMul: lhs.width == rhs.height → d_model == w_up.height ✓
    # Output: h = lhs.height = batch, w = rhs.width = d_ff
    h = g.matmul(x, w_up)
    h = g.pointwise([h])  # GELU
    w_down = g.tensor(d_model, d_ff)
    out = g.matmul(h, w_down)
    return out


def build_rmsnorm(g: Graph, x: int) -> int:
    """RMSNorm (explicit): 2 PWs modeling mean-square + normalize-and-scale.
    Real RMSNorm reduces over last dim; here kept same-shape (per-tile cost
    is what base_cost models). Produces same-shape output as x."""
    ms = g.pointwise([x])           # reduce-square (same-shape proxy)
    out = g.pointwise([x, ms])      # x * rsqrt(ms) * weight (folded)
    return out


def build_swiglu_ffn(g: Graph, x: int, d_model: int, d_ff: int) -> int:
    """SwiGLU FFN (Llama/Mistral/Qwen): gate, up, silu(gate)*up, down.
    3 MMs + 1 two-input PW — replaces regular FFN's 2 MMs + 1 PW. gate/up
    share x as LHS, enabling RHS-chain fusion opportunities."""
    W_gate = g.tensor(d_ff, d_model)
    W_up = g.tensor(d_ff, d_model)
    W_down = g.tensor(d_model, d_ff)
    gate = g.matmul(x, W_gate)         # (seq × d_ff)
    up = g.matmul(x, W_up)             # (seq × d_ff)
    act = g.pointwise([gate, up])      # silu(gate) * up
    out = g.matmul(act, W_down)        # (seq × d_model)
    return out


def build_bigtile_mm(g: Graph, dim: int) -> int:
    """MM → PW → MM at (dim × dim) square shape. Minimal topology with high
    tile-count per op ((dim/native)² tiles each) — stresses split-K and
    tile-decomposition scheduling with no fusion-chain pressure."""
    x = g.tensor(dim, dim)
    w1 = g.tensor(dim, dim)
    h = g.matmul(x, w1)
    h = g.pointwise([h])
    w2 = g.tensor(dim, dim)
    return g.matmul(h, w2)


def build_attention_head(g: Graph, x: int, d_model: int, d_head: int,
                          seq_len: int) -> int:
    """Single attention head: per-head projections + scores + attn + out.

      Q_h = x @ W_q_h   (seq × d_head)
      K_h = x @ W_k_h   (seq × d_head)
      V_h = x @ W_v_h   (seq × d_head)
      scores = Q_h @ K_h^T  → (seq × seq)
      attn = softmax(scores)
      out = attn @ V_h  → (seq × d_head)
    """
    # Projections: x (seq × d_model) @ W (d_model × d_head). W has width=d_head, height=d_model.
    W_q = g.tensor(d_head, d_model)
    W_k = g.tensor(d_head, d_model)
    W_v = g.tensor(d_head, d_model)
    Q = g.matmul(x, W_q)          # (seq × d_head)
    K = g.matmul(x, W_k)          # (seq × d_head)
    V = g.matmul(x, W_v)          # (seq × d_head)
    # scores = Q @ K^T : lhs=Q (h=seq, k=d_head), rhs=K^T (k=d_head, w=seq).
    # K^T is a fresh tensor (graph input proxy for the transpose).
    K_T = g.tensor(seq_len, d_head)
    scores = g.matmul(Q, K_T)     # (seq × seq)
    attn = g.pointwise([scores])  # softmax
    out = g.matmul(attn, V)       # (seq × d_head)
    return out


def build_multi_head_attention(g: Graph, x: int, d_model: int,
                                n_heads: int, seq_len: int,
                                with_output_proj: bool = True) -> int:
    """Multi-head attention: n_heads independent heads sharing x, merge, optional proj."""
    d_head = d_model // n_heads
    outs = []
    for _ in range(n_heads):
        out = build_attention_head(g, x, d_model, d_head, seq_len)
        outs.append(out)
    merged = outs[0]
    for o in outs[1:]:
        merged = g.pointwise([merged, o])
    if with_output_proj:
        W_o = g.tensor(d_model, d_head)
        merged = g.matmul(merged, W_o)
    return merged


# ── GQA / MQA: grouped/shared K-V heads (LLM-friendly) ───────────────────────

def build_gqa(g: Graph, x: int, d_model: int,
              n_q_heads: int, n_kv_heads: int, seq_len: int) -> int:
    """Grouped-query attention. n_kv_heads K/V heads shared by (n_q_heads//n_kv_heads)
    Q heads each. n_kv_heads=1 is MQA; n_kv_heads=n_q_heads is MHA."""
    assert n_q_heads % n_kv_heads == 0
    d_head = d_model // n_q_heads
    heads_per_group = n_q_heads // n_kv_heads
    outs = []
    for _ in range(n_kv_heads):
        W_k = g.tensor(d_head, d_model)
        W_v = g.tensor(d_head, d_model)
        K = g.matmul(x, W_k)   # (seq × d_head), shared across group
        V = g.matmul(x, W_v)
        K_T = g.tensor(seq_len, d_head)
        for _ in range(heads_per_group):
            W_q = g.tensor(d_head, d_model)
            Q = g.matmul(x, W_q)
            scores = g.matmul(Q, K_T)      # Q consumes shared K_T structure
            attn = g.pointwise([scores])
            out = g.matmul(attn, V)         # also consumes shared V
            outs.append(out)
    merged = outs[0]
    for o in outs[1:]:
        merged = g.pointwise([merged, o])
    W_o = g.tensor(d_model, d_head)
    return g.matmul(merged, W_o)


# ── KV-cache decoder step: K/V are graph inputs (cached history) ─────────────

def build_kv_cache_decoder_step(g: Graph, x: int, d_model: int,
                                 n_heads: int, kv_seq_len: int,
                                 q_seq_len: int = 1) -> int:
    """Single-step decoder with KV cache. Q is computed for q_seq_len tokens,
    K/V are graph inputs covering kv_seq_len tokens.
    Typical: q_seq_len=1 (one new token), kv_seq_len = context window."""
    d_head = d_model // n_heads
    outs = []
    for _ in range(n_heads):
        W_q = g.tensor(d_head, d_model)
        Q = g.matmul(x, W_q)           # (q_seq × d_head)
        # K cached: (kv_seq × d_head). K^T: (d_head × kv_seq).
        K_T = g.tensor(kv_seq_len, d_head)
        # V cached: (kv_seq × d_head).
        V = g.tensor(d_head, kv_seq_len)
        scores = g.matmul(Q, K_T)       # (q_seq × kv_seq)
        attn = g.pointwise([scores])
        out = g.matmul(attn, V)         # (q_seq × d_head)
        outs.append(out)
    merged = outs[0]
    for o in outs[1:]:
        merged = g.pointwise([merged, o])
    W_o = g.tensor(d_model, d_head)
    return g.matmul(merged, W_o)


# ── MoE FFN: top-k routed experts (top-all simplification) ───────────────────

def build_moe_ffn(g: Graph, x: int, d_model: int, d_ff: int,
                   n_experts: int) -> int:
    """MoE FFN: router computes gating scores, all experts contribute via
    weighted sum. Real top-k routing requires dynamic dispatch not modelable
    in static DAG — we use top-all with PW gating as a stress proxy."""
    # Router: x → logits (seq × n_experts) → gate weights (PW)
    W_router = g.tensor(n_experts, d_model)
    logits = g.matmul(x, W_router)   # (seq × n_experts)
    gates = g.pointwise([logits])     # softmax
    # Each expert: up + gelu + down; result scaled by gate (PW).
    expert_outs = []
    for _ in range(n_experts):
        W_up = g.tensor(d_ff, d_model)
        W_down = g.tensor(d_model, d_ff)
        h = g.matmul(x, W_up)          # (seq × d_ff)
        h = g.pointwise([h])           # gelu
        eo = g.matmul(h, W_down)       # (seq × d_model)
        # Gate-weight the output (PW reading gates + eo; in reality a broadcast).
        eo = g.pointwise([eo, eo])     # 2-input PW proxy for gate × expert
        expert_outs.append(eo)
    merged = expert_outs[0]
    for eo in expert_outs[1:]:
        merged = g.pointwise([merged, eo])
    # Include gates as a consumer of router (ensures it's non-ephemeral-external).
    _ = gates
    return merged


# ── Decoder block with GQA + FFN ─────────────────────────────────────────────

def build_decoder_block(g: Graph, x: int, d_model: int,
                         n_q_heads: int, n_kv_heads: int,
                         d_ff: int, seq_len: int) -> int:
    attn_out = build_gqa(g, x, d_model, n_q_heads, n_kv_heads, seq_len)
    x1 = g.pointwise([x, attn_out])
    ffn_out = build_ffn(g, x1, d_model, d_ff)
    x2 = g.pointwise([x1, ffn_out])
    return x2


def build_llama3_block(g: Graph, x: int, d_model: int,
                        n_q_heads: int, n_kv_heads: int,
                        d_ff: int, seq_len: int) -> int:
    """Llama-3 decoder block: RMSNorm → GQA → +residual → RMSNorm → SwiGLU → +residual.
    Pre-norm topology (modern LLMs). Distinct from build_decoder_block in:
      - explicit 2-PW RMSNorm (not folded into residual)
      - SwiGLU instead of ReLU/GELU FFN (3 MMs vs 2)
    Per-block ops: 10 + 2·n_kv + 5·n_q."""
    h1 = build_rmsnorm(g, x)
    attn = build_gqa(g, h1, d_model, n_q_heads, n_kv_heads, seq_len)
    x1 = g.pointwise([x, attn])
    h2 = build_rmsnorm(g, x1)
    ffn = build_swiglu_ffn(g, h2, d_model, d_ff)
    x2 = g.pointwise([x1, ffn])
    return x2


def build_transformer_block(g: Graph, x: int, d_model: int,
                             n_heads: int, d_ff: int, seq_len: int) -> int:
    """Attn + residual + FFN + residual. Norms folded into PWs."""
    attn_out = build_multi_head_attention(g, x, d_model, n_heads, seq_len)
    x1 = g.pointwise([x, attn_out])  # residual + norm
    ffn_out = build_ffn(g, x1, d_model, d_ff)
    x2 = g.pointwise([x1, ffn_out])  # residual + norm
    return x2


# ── Scenario Catalog ─────────────────────────────────────────────────────────

SCENARIOS: Dict[str, dict] = {
    # ── MLP: linear-chain fusion stress ──
    "mlp_small":       dict(arch="mlp", depth=3,  d_model=128, seq_len=128,
                            cap=60000,   bw=20),
    "mlp_medium":      dict(arch="mlp", depth=8,  d_model=512, seq_len=128,
                            cap=200000,  bw=25),
    "mlp_large":       dict(arch="mlp", depth=16, d_model=1024, seq_len=128,
                            cap=800000,  bw=40),
    "mlp_medium_tight": dict(arch="mlp", depth=8, d_model=512, seq_len=128,
                             cap=50000,  bw=25),   # tight cap
    "mlp_large_tight": dict(arch="mlp", depth=16, d_model=1024, seq_len=128,
                            cap=200000, bw=40),

    # ── FFN stack: PW+MM mixed fusion ──
    "ffn_small":       dict(arch="ffn", depth=3,  d_model=128, d_ff=512,   seq_len=128,
                            cap=60000,  bw=20),
    "ffn_medium":      dict(arch="ffn", depth=6,  d_model=512, d_ff=2048,  seq_len=128,
                            cap=400000, bw=30),
    "ffn_large":       dict(arch="ffn", depth=12, d_model=1024, d_ff=4096, seq_len=128,
                            cap=1500000, bw=50),
    "ffn_medium_tight": dict(arch="ffn", depth=6,  d_model=512, d_ff=2048,  seq_len=128,
                             cap=100000, bw=30),
    "ffn_large_tight":  dict(arch="ffn", depth=12, d_model=1024, d_ff=4096, seq_len=128,
                             cap=400000, bw=50),

    # ── Attention: multi-head parallel fusion + retention ──
    "attn_small":      dict(arch="mha", n_heads=2,  d_model=128,  seq_len=128,
                            cap=80000,  bw=20),
    "attn_medium":     dict(arch="mha", n_heads=8,  d_model=512,  seq_len=256,
                            cap=500000, bw=50),
    "attn_large":      dict(arch="mha", n_heads=16, d_model=1024, seq_len=1024,
                            cap=2000000, bw=100),
    "attn_medium_tight": dict(arch="mha", n_heads=8,  d_model=512,  seq_len=256,
                              cap=200000, bw=50),
    "attn_large_tight":  dict(arch="mha", n_heads=16, d_model=1024, seq_len=1024,
                              cap=600000, bw=100),

    # ── Transformer block: full attn + FFN + residuals ──
    "transformer_small":  dict(arch="transformer", n_blocks=1, d_model=128,  n_heads=4,
                               d_ff=512,  seq_len=128, cap=80000,  bw=20),
    "transformer_medium": dict(arch="transformer", n_blocks=2, d_model=512,  n_heads=8,
                               d_ff=2048, seq_len=256, cap=600000, bw=50),
    "transformer_large":  dict(arch="transformer", n_blocks=4, d_model=1024, n_heads=16,
                               d_ff=4096, seq_len=1024, cap=2500000, bw=100),
    "transformer_medium_tight": dict(arch="transformer", n_blocks=2, d_model=512,
                                     n_heads=8, d_ff=2048, seq_len=256,
                                     cap=150000, bw=50),

    # ── GQA / MQA (LLM-style attention) ──
    # GQA with 8 Q heads, 2 KV groups (4 heads/group): Llama3-ish ratio.
    "gqa_medium":       dict(arch="gqa", n_q_heads=8, n_kv_heads=2, d_model=512,
                              seq_len=256, cap=400000, bw=50),
    "gqa_medium_tight": dict(arch="gqa", n_q_heads=8, n_kv_heads=2, d_model=512,
                              seq_len=256, cap=120000, bw=50),
    "gqa_large":        dict(arch="gqa", n_q_heads=16, n_kv_heads=4, d_model=1024,
                              seq_len=1024, cap=1500000, bw=100),
    # MQA: single KV head shared across all Q heads.
    "mqa_medium":       dict(arch="gqa", n_q_heads=8, n_kv_heads=1, d_model=512,
                              seq_len=256, cap=400000, bw=50),

    # ── KV-cache decoder step (long context) ──
    # q_seq=1 (one new token), kv_seq=large (context window).
    "kvcache_short":  dict(arch="kvcache", n_heads=8,  d_model=512,
                            kv_seq=512,  q_seq=1, cap=300000, bw=50),
    "kvcache_long":   dict(arch="kvcache", n_heads=16, d_model=1024,
                            kv_seq=4096, q_seq=1, cap=1500000, bw=100),
    "kvcache_long_tight": dict(arch="kvcache", n_heads=16, d_model=1024,
                                kv_seq=4096, q_seq=1, cap=500000, bw=100),

    # ── MoE (fixed top-all routing, static DAG proxy) ──
    "moe_small":   dict(arch="moe", d_model=128, d_ff=512,  n_experts=4,
                         seq_len=128, cap=80000,  bw=20),
    "moe_medium":  dict(arch="moe", d_model=512, d_ff=2048, n_experts=8,
                         seq_len=256, cap=600000, bw=50),

    # ── Decoder-only LLM: stack of GQA decoder blocks (Llama-ish) ──
    "decoder_small":  dict(arch="decoder", n_blocks=2, d_model=512,
                            n_q_heads=8, n_kv_heads=2, d_ff=2048, seq_len=256,
                            cap=500000, bw=50),
    "decoder_large":  dict(arch="decoder", n_blocks=4, d_model=1024,
                            n_q_heads=16, n_kv_heads=4, d_ff=4096, seq_len=1024,
                            cap=2000000, bw=100),

    # ── Memory-capacity sweep on same arch (attn_medium) ──
    "attn_medium_cap1x":   dict(arch="mha", n_heads=8, d_model=512, seq_len=256,
                                 cap=250000, bw=50),   # cap ≈ 1× max tensor
    "attn_medium_cap10x":  dict(arch="mha", n_heads=8, d_model=512, seq_len=256,
                                 cap=2500000, bw=50),  # cap ≫ max tensor

    # ── SwiGLU FFN stack (Llama/Mistral/Qwen FFN variant) ──
    # 3 MMs + 1 two-input PW per block (vs regular FFN 2 MMs + 1 PW).
    # Exposes RHS-chain fusion on gate/up projections sharing x.
    "swiglu_small":        dict(arch="swiglu", depth=3,  d_model=128,  d_ff=512,
                                 seq_len=128, cap=60000,   bw=20),
    "swiglu_medium":       dict(arch="swiglu", depth=6,  d_model=512,  d_ff=2048,
                                 seq_len=128, cap=400000,  bw=30),
    "swiglu_medium_tight": dict(arch="swiglu", depth=6,  d_model=512,  d_ff=2048,
                                 seq_len=128, cap=100000,  bw=30),
    "swiglu_large":        dict(arch="swiglu", depth=12, d_model=1024, d_ff=4096,
                                 seq_len=128, cap=1500000, bw=50),

    # ── Llama-3 decoder block (RMSNorm + GQA + SwiGLU, realistic modern LLM) ──
    # Per-block ops: 10 + 2·n_kv + 5·n_q. llama3_small ≈ Llama-3 scaled down.
    "llama3_small":        dict(arch="llama3", n_blocks=1, d_model=512,
                                 n_q_heads=8,  n_kv_heads=2,  d_ff=2048,
                                 seq_len=256,  cap=500000,  bw=50),
    "llama3_medium":       dict(arch="llama3", n_blocks=2, d_model=1024,
                                 n_q_heads=16, n_kv_heads=4,  d_ff=4096,
                                 seq_len=512,  cap=1500000, bw=100),
    "llama3_medium_tight": dict(arch="llama3", n_blocks=2, d_model=1024,
                                 n_q_heads=16, n_kv_heads=4,  d_ff=4096,
                                 seq_len=512,  cap=500000,  bw=100),
    "llama3_large":        dict(arch="llama3", n_blocks=4, d_model=1024,
                                 n_q_heads=32, n_kv_heads=8,  d_ff=4096,
                                 seq_len=1024, cap=2500000, bw=100),

    # ── Tile-regime extremes (orthogonal to architectural stress) ──
    # tile_chain_*: d_model = native → every tensor is exactly 1 native tile.
    # Zero split-K freedom; pure fusion-chain pressure over many small ops.
    "tile_chain_short":  dict(arch="mlp", depth=5,  d_model=128, seq_len=128,
                                cap=30000,  bw=15),
    "tile_chain_medium": dict(arch="mlp", depth=10, d_model=128, seq_len=128,
                                cap=30000,  bw=15),
    "tile_chain_long":   dict(arch="mlp", depth=20, d_model=128, seq_len=128,
                                cap=40000,  bw=20),
    # bigtile_mm_*: square MM → PW → MM at (dim×dim). Minimal topology but
    # (dim/128)² tiles per op → heavy split-K / decomposition stress.
    "bigtile_mm_16x16":  dict(arch="bigtile", dim=2048, cap=800000,  bw=100),
    "bigtile_mm_32x32":  dict(arch="bigtile", dim=4096, cap=2000000, bw=150),

    # ── Prefill GQA: long-seq full prefill (contrasts kvcache_* decode) ──
    # scores are seq×seq (vs decode's 1×kv_seq). Stresses retention + tile-split.
    "prefill_gqa_short":     dict(arch="gqa", n_q_heads=8,  n_kv_heads=2,
                                    d_model=512,  seq_len=512,  cap=400000,  bw=50),
    "prefill_gqa_long":      dict(arch="gqa", n_q_heads=16, n_kv_heads=4,
                                    d_model=1024, seq_len=2048, cap=2500000, bw=100),
    "prefill_gqa_long_tight": dict(arch="gqa", n_q_heads=16, n_kv_heads=4,
                                    d_model=1024, seq_len=2048, cap=800000,  bw=100),
}


def build_scenario(cfg: dict) -> dict:
    g = Graph()
    arch = cfg["arch"]
    seq = cfg.get("seq_len", 128)

    if arch == "mlp":
        x = g.tensor(cfg["d_model"], seq)   # input
        build_mlp(g, x, cfg["depth"], cfg["d_model"])
    elif arch == "ffn":
        x = g.tensor(cfg["d_model"], seq)
        for _ in range(cfg["depth"]):
            x = build_ffn(g, x, cfg["d_model"], cfg["d_ff"])
    elif arch == "mha":
        x = g.tensor(cfg["d_model"], seq)
        build_multi_head_attention(g, x, cfg["d_model"], cfg["n_heads"], seq)
    elif arch == "transformer":
        x = g.tensor(cfg["d_model"], seq)
        for _ in range(cfg["n_blocks"]):
            x = build_transformer_block(g, x, cfg["d_model"], cfg["n_heads"],
                                         cfg["d_ff"], seq)
    elif arch == "gqa":
        x = g.tensor(cfg["d_model"], seq)
        build_gqa(g, x, cfg["d_model"], cfg["n_q_heads"], cfg["n_kv_heads"], seq)
    elif arch == "kvcache":
        q_seq = cfg.get("q_seq", 1)
        x = g.tensor(cfg["d_model"], q_seq)
        build_kv_cache_decoder_step(g, x, cfg["d_model"], cfg["n_heads"],
                                      cfg["kv_seq"], q_seq)
    elif arch == "moe":
        x = g.tensor(cfg["d_model"], seq)
        build_moe_ffn(g, x, cfg["d_model"], cfg["d_ff"], cfg["n_experts"])
    elif arch == "decoder":
        x = g.tensor(cfg["d_model"], seq)
        for _ in range(cfg["n_blocks"]):
            x = build_decoder_block(g, x, cfg["d_model"], cfg["n_q_heads"],
                                      cfg["n_kv_heads"], cfg["d_ff"], seq)
    elif arch == "swiglu":
        x = g.tensor(cfg["d_model"], seq)
        for _ in range(cfg["depth"]):
            x = build_swiglu_ffn(g, x, cfg["d_model"], cfg["d_ff"])
    elif arch == "llama3":
        x = g.tensor(cfg["d_model"], seq)
        for _ in range(cfg["n_blocks"]):
            x = build_llama3_block(g, x, cfg["d_model"], cfg["n_q_heads"],
                                    cfg["n_kv_heads"], cfg["d_ff"], seq)
    elif arch == "bigtile":
        build_bigtile_mm(g, cfg["dim"])
    else:
        raise ValueError(f"Unknown arch: {arch}")

    return g.to_dict(cfg["cap"], cfg["bw"], tuple(cfg.get("native", (128, 128))))


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("scenario", nargs="?", help="scenario name (see --list)")
    ap.add_argument("--list", action="store_true", help="list scenarios")
    ap.add_argument("--out", type=str, default=None, help="output path")
    ap.add_argument("--all", action="store_true",
                    help="generate all scenarios into synthetic/generated/")
    args = ap.parse_args()

    if args.list:
        for name, cfg in SCENARIOS.items():
            print(f"  {name:30s} arch={cfg['arch']:12s}  cap={cfg['cap']:>10}  bw={cfg['bw']:>4}")
        return

    if args.all:
        out_dir = Path(__file__).parent / "generated"
        out_dir.mkdir(exist_ok=True)
        for name, cfg in SCENARIOS.items():
            prob = build_scenario(cfg)
            out_path = out_dir / f"{name}.json"
            out_path.write_text(json.dumps(prob, indent=2))
            n_ops = len(prob["op_types"])
            n_mm = sum(1 for t in prob["op_types"] if t == "MatMul")
            print(f"  {name:30s} N={n_ops:>4} ({n_mm} MM)  → {out_path.name}")
        return

    if not args.scenario:
        ap.print_help()
        sys.exit(1)
    if args.scenario not in SCENARIOS:
        sys.exit(f"Unknown scenario: {args.scenario}")
    prob = build_scenario(SCENARIOS[args.scenario])
    out_path = args.out or f"synthetic/generated/{args.scenario}.json"
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(prob, indent=2))
    print(f"  {args.scenario}: N={len(prob['op_types'])}  → {out_path}")


if __name__ == "__main__":
    main()
