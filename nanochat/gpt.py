"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from nanochat.common import COMPUTE_DTYPE, print0

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn
from nanochat.optim import MuonAdamW


@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6  # number of query heads
    n_kv_head: int = 6  # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


# https://arxiv.org/html/1910.07467v1
def norm(x):
    return F.rms_norm(x, (x.size(-1),))  # note that this will run in bf16, seems ok

# nn.Linear: fully connected (dense) layer
# -----------------------------------------
# Applies an affine transformation to the input:
#     y = x @ W.T + b
#
# Parameters:
#   - weight: shape (out_features, in_features)
#   - bias:   shape (out_features,), can be disabled with bias=False
#   Both are nn.Parameter, so they are registered automatically and
#   updated by the optimizer.
#
# Input/output shapes:
#   - Input:  (*, in_features)
#   - Output: (*, out_features)
#   Only the last dimension is transformed; all leading dimensions are
#   treated as batch dimensions. For example, a Transformer tensor of
#   shape (batch, seq_len, d_model) maps to (batch, seq_len, out_features).
#
# Notes:
#   - A Linear bias can usually be omitted when BatchNorm immediately follows
#     and normalizes each output feature: batch centering cancels the bias
#     during training, and running statistics can absorb it for inference.
#     This does not generally apply to LayerNorm, which centers across features;
#     a per-feature bias can still change the normalized output.
#   - Default init: weight uses Kaiming uniform; bias is sampled from a
#     uniform distribution whose bounds depend on in_features.
#   - Under the hood it calls torch.nn.functional.linear, which dispatches
#     to a BLAS GEMM (MKL/OpenBLAS on CPU, cuBLAS on GPU). The bias add is
#     often fused into the GEMM epilogue.
#   - A single Linear layer is purely linear, and stacking linear layers is
#     still linear. Insert a non-linear activation (e.g. ReLU) between them.
#   - In Transformers, the Q/K/V projections, the output projection, and
#     both FFN layers are all nn.Linear. They hold most of the parameters
#     and compute in large models.
#
#
#
# F.linear (torch.nn.functional.linear)
# -------------------------------------
# Note: the function name is lowercase `F.linear`; there is no `F.Linear`.
#
# F.linear is the stateless functional version of nn.Linear. It computes
# the same affine transformation:
#     y = x @ W.T + b
# but it does not own any parameters. The caller must pass weight and
# bias explicitly.
#
# Arguments:
#   - input:  shape (*, in_features)
#   - weight: shape (out_features, in_features)
#   - bias:   shape (out_features,), optional
#   Output:   shape (*, out_features)
#
# Relationship to nn.Linear:
#   nn.Linear.forward is essentially a single line:
#       return F.linear(input, self.weight, self.bias)
#   The only difference is who owns the parameters:
#     - nn.Linear: an nn.Module that creates, registers, and initializes
#       its own weight/bias. They show up in model.parameters() and are
#       saved/loaded via state_dict automatically.
#     - F.linear: a plain function. Parameters are passed in by the
#       caller and are not tracked unless you register them yourself.
#
# When to use F.linear:
#   - Weight tying: e.g. reuse the embedding matrix as the output
#     projection in a language model.
#   - Dynamically computed weights: hypernetworks, merged LoRA weights,
#     meta-learning (MAML), where weights are computed rather than fixed
#     parameters.
#   - Custom Modules: you manage parameters yourself (special init,
#     quantization, etc.) and call F.linear inside forward().
#
# For ordinary network building, nn.Linear is enough. Use F.linear
# directly only when you need non-standard handling of the weights.
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""

    def forward(self, x):  # ty: ignore[invalid-method-override] - parameter is named `input` in nn.Linear
        # Calling layer.to(dtype=x.dtype) would change the parameters themselves.
        # Storing the updated weights in BF16 could round away small optimizer
        # updates; casting back to FP32 later cannot recover the lost precision.
        # Cast only the weight Tensor used in this computation, keeping the original
        # self.weight in FP32 (unless the model was explicitly cast elsewhere).
        # Matmul uses x's dtype, while the differentiable Tensor.to() propagates
        # gradients back to the original FP32 parameter for optimizer updates.
        # If the dtype already matches, Tensor.to() returns the original Tensor.
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

# x.shape = (B, T, H, D)
def apply_rotary_emb(x, cos, sin):
    """Apply rotary position embeddings (RoPE) to Q or K, preserving (B, T, H, D).

    B is the batch size, T is the number of tokens in this call, H is the number
    of Q or K heads, and D is the head dimension, which must be even.
    Q and K may have different head counts (e.g. in GQA).
    cos and sin have shape (1, T, 1, D/2); the caller slices them for the current
    token positions. With a KV cache, these positions include the cache offset
    rather than always starting at zero.

    At position p, coordinate pair i uses angle phi_i = p * omega_i.
    Frequencies are precomputed as omega_i = base ** (-2*i/D), with base=100000
    by default. This function uses the supplied cosines and sines; it does not
    compute the angles itself.
    """
    # Attention applies this function separately to Q and K, but not to V.
    assert x.ndim == 4  # multihead attention

    # Here d is D/2, not the full head dimension.
    # Matching entries in the two halves form 2D coordinate pairs (x_i, x_{i+d}).
    # For D=8, the pairs are (x0,x4), (x1,x5), (x2,x6), and (x3,x7),
    # rather than adjacent pairs (x0,x1), (x2,x3), etc.
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last dim into two halves

    # x1 and x2 are slice views of x, each with shape (B, T, H, d).
    # cos and sin have shape (1, T, 1, d) and broadcast across B and H:
    # all batches/heads share angles at a given position, while each coordinate
    # pair has its own frequency. Elementwise operations implement the following
    # matrix multiplication for each pair, written as a column vector:
    #
    #   [y1_i]   [ cos(phi_i)   sin(phi_i)] [x1_i]
    #   [y2_i] = [-sin(phi_i)   cos(phi_i)] [x2_i]
    #
    # Comparing with the RoFormer paper (Eq. 13), using column vectors:
    #   paper: R(a)  = [[cos(a), -sin(a)], [ sin(a), cos(a)]]
    #   here:  R(-a) = [[cos(a),  sin(a)], [-sin(a), cos(a)]]
    # These rotate in opposite directions. When comparing formulas, note that
    # switching from column-vector left multiplication to row-vector right
    # multiplication transposes the matrix for the SAME geometric rotation:
    #   y = R(a)x  <=>  y^T = x^T R(a)^T.
    #
    # Why do both directions work for RoPE? For a single coordinate pair with
    # frequency w, R(a)^T = R(-a) and R(a)R(b) = R(a+b), so:
    #   positive: (R( m*w)q)^T (R( n*w)k) = q^T R((n-m)*w) k
    #   negative: (R(-m*w)q)^T (R(-n*w)k) = q^T R((m-n)*w) k.
    # Both introduce position dependence only through the relative position;
    # the sign convention differs. Q and K must use the same convention.
    # Opposite conventions for Q and K would instead produce a position SUM.
    # This does not mean both directions give identical scores for fixed q/k.
    # They are equivalent parameterizations at a fixed frequency: reflecting
    # the second coordinate of every pair in BOTH Q and K converts one to the
    # other, since S R(a) S = R(-a) for S = diag(1, -1). Such a reflection can
    # be absorbed into the Q/K projection weights. The existing direction is
    # kept for checkpoint compatibility; changing it alone changes behavior.
    # Each pair's squared norm is preserved in exact arithmetic, subject to
    # rounding error in floating-point computation.
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos

    # Concatenate the halves back to (B, T, H, D), preserving the original layout:
    # [y1_0, ..., y1_{d-1}, y2_0, ..., y2_{d-1}].
    # No explicit reordering into adjacent pairs or D-by-D matrix is needed.
    #
    # Why does this layout work even though each pair's entries are not adjacent?
    # The pairs (x_i, x_{i+d}) define independent 2D coordinate planes covering
    # all D coordinates. Adjacency in storage is not required for 2D rotation.
    # Since Q and K use the same layout, their dot product can be regrouped
    # into the sum of those 2D dot products. For D=4, write the rotated vectors as
    #   q' = [u0, u1, v0, v1], k' = [r0, r1, s0, s1].
    # Then
    #   q'^T k' = u0*r0 + u1*r1 + v0*s0 + v1*s1
    #           = (u0*r0 + v0*s0) + (u1*r1 + v1*s1).
    # Each parenthesis is one rotated pair's dot product and therefore has the
    # relative-position property derived above; their sum has it as well.
    # Equivalently, applying the same coordinate permutation P to both vectors
    # preserves the dot product: (Pq')^T(Pk') = q'^T k', since P^T P = I.
    # This explains why the output need not interleave y1_i and y2_i.
    # It does NOT mean that changing the input pairing with fixed Q/K weights
    # gives the same scores: matching the paper's adjacent pairing requires a
    # corresponding permutation of the Q/K projection output channels (and a
    # sign change to account for the opposite rotation direction).
    #
    # These operations preserve the autograd graph, allowing gradients to flow
    # through x back to the Q/K projection weights.
    return torch.cat([y1, y2], 3)


class CausalSelfAttention(nn.Module):
    """Shape notation: B=batch size, T=tokens in this call, C=n_embd,
    Hq=n_head, Hkv=n_kv_head, D=head_dim, S=KV cache capacity.
    C = Hq * D; GQA shares each KV head across Hq/Hkv query heads.
    Tensor layouts in this class use (batch, time, heads, head_dim).
    """

    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # Linear weights use (out_features, in_features); all biases are disabled.
        # c_q.weight: (Hq*D, C) = (C, C)
        # c_k.weight and c_v.weight: (Hkv*D, C)
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)  # weight: (C, C)
        self.ve_gate_channels = 12
        # When present, ve_gate.weight: (Hkv, 12); one gate per KV head/token.
        self.ve_gate = (
            Linear(self.ve_gate_channels, self.n_kv_head, bias=False)
            if has_ve(layer_idx, config.n_layer)
            else None
        )

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        # x: (B, T, C); ve: None or (B, T, Hkv*D).
        # cos_sin: two tensors, each (1, T, 1, D/2), sliced for current positions.
        # window_size: a pair of integer bounds, not a Tensor.
        # kv_cache: None or a cache object; T counts only tokens in this call,
        # including a prefill chunk or typically one token during decode.
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        # c_q(x): (B, T, Hq*D) -> q: (B, T, Hq, D)
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        # c_k(x): (B, T, Hkv*D) -> k: (B, T, Hkv, D)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        # c_v(x): (B, T, Hkv*D) -> v: (B, T, Hkv, D)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            assert self.ve_gate is not None  # layers that receive ve always have a gate
            # ve: (B, T, Hkv*D) -> (B, T, Hkv, D), matching v.
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            # x[..., :12]: (B, T, 12), assuming C >= 12 as required by ve_gate.
            # ve_gate output, sigmoid, and gate: (B, T, Hkv), range (0, 3).
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., : self.ve_gate_channels]))
            # gate.unsqueeze(-1): (B, T, Hkv, 1), broadcast across D.
            # gate * ve and the updated v: (B, T, Hkv, D).
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin  # each: (1, T, 1, D/2), broadcast over B and heads
        # RoPE preserves q: (B, T, Hq, D) and k: (B, T, Hkv, D).
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # norm reduces over D internally; normalization preserves both shapes.
        q, k = norm(q), norm(k)  # QK norm
        # sharper attention (split scale between Q and K), TODO think through better
        # Scalar scaling preserves q/k shapes.
        q = q * 1.2
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            # Logical attention scores/weights: (B, Hq, T, T), with KV heads
            # shared under GQA. Flash Attention need not materialize this tensor.
            # Output y: (B, T, Hq, D), using the QUERY head count.
            y = flash_attn.flash_attn_func(
                q, k, v, causal=True, window_size=window_size
            )
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            # Each layer cache: (B, S, Hkv, D); S is allocated capacity.
            # cache_seqlens: (B,) int32, containing each sequence's cached length.
            # New k/v: (B, T, Hkv, D), written into the cache in place.
            # For a uniform previous length P, attention uses P+T keys/values:
            # logical scores/weights are (B, Hq, T, P+T), subject to masking.
            # Output y: (B, T, Hq, D), containing only this call's query outputs.
            y = flash_attn.flash_attn_with_kvcache(
                q,
                k_cache,
                v_cache,
                k=k,
                v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                # Add scalar T to cache_seqlens; its shape remains (B,).
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        # contiguous(): same (B, T, Hq, D) shape, with contiguous storage.
        # view merges Hq and D: (B, T, Hq*D) = (B, T, C).
        y = y.contiguous().view(B, T, -1)
        # c_proj: (B, T, C) -> (B, T, C), matching the residual stream.
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = (
            (config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to
        ) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(
                f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency"
            )
        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(padded_vocab_size, config.n_embd),
                "h": nn.ModuleList(
                    [Block(config, layer_idx) for layer_idx in range(config.n_layer)]
                ),
            }
        )
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        # fake init, real init in init_weights()
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))
        # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict(
            {
                str(i): nn.Embedding(padded_vocab_size, kv_dim)
                for i in range(config.n_layer)
                if has_ve(i, config.n_layer)
            }
        )
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        # 10X over-compute should be enough, TODO make nicer?
        self.rotary_seq_len = config.sequence_len * 10
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # persistent=False means it's not saved to the checkpoint
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

    # Typed views of the submodules stored in self.transformer. Attribute access on
    # nn.ModuleDict is typed as `Tensor | Module`; these return the same objects,
    # so parameter names and state_dict keys are unchanged.
    @property
    def wte(self) -> nn.Embedding:
        return cast(nn.Embedding, self.transformer["wte"])

    @property
    def blocks(self) -> Sequence["Block"]:
        return cast(Sequence[Block], self.transformer["h"])

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        s = 3**0.5 * n_embd**-0.5
        for block in self.blocks:
            # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight)  # projections are zero
            # 0.4x init scale for c_fc
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
        # Decaying x0 init: earlier layers get more input embedding blending
        for i in range(n_layer):
            self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))

        # Smear/backout scalars and smear gate must be explicitly initialized
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.constant_(self.backout_lambda, 0.2)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(cast(nn.Embedding, ve).weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.blocks:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def _precompute_rotary_embeddings(
        self, seq_len, head_dim, base=100000, device=None
    ):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        # add batch and head dims for later broadcasting
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), (
            f"Invalid window_pattern: {pattern}. Use only S and L."
        )
        # Map characters to window sizes
        long_window = config.sequence_len
        # ceil to FA3 tile size (2048 -> 768)
        short_window = -(-long_window // 4 // 128) * 128
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = (
            self.config.n_head,
            self.config.n_embd // self.config.n_head,
            self.config.sequence_len,
        )
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(
            m.weight.numel() for m in self.modules() if isinstance(m, Linear)
        )
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(
            4 * h * q * min(context_len, window) for window, _ in self.window_sizes
        )
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            # ramp up to w, then flat
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        # the KV cache is kept in the compute dtype
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        return (
            self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes
        )

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += (
                2
                * self.config.n_kv_head
                * head_dim
                * kv_dtype_bytes
                * min(context_len, window)
            )
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(
            p.numel() for p in self.transformer["h"].parameters()
        )
        scalars = (
            self.resid_lambdas.numel()
            + self.x0_lambdas.numel()
            + self.smear_gate.weight.numel()
            + self.smear_lambda.numel()
            + self.backout_lambda.numel()
        )
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), (
            "Parameter count mismatch"
        )
        return {
            "wte": wte,
            "value_embeds": value_embeds,
            "lm_head": lm_head,
            "transformer_matrices": transformer_matrices,
            "scalars": scalars,
            "total": total,
        }

    def setup_optimizer(
        self,
        unembedding_lr=0.004,
        embedding_lr=0.2,
        matrix_lr=0.02,
        weight_decay=0.0,
        scalar_lr=0.5,
    ):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer["h"].parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(
            embedding_params
        ) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(
            x0_params
        ) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(
            f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}"
        )

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(
                kind="adamw",
                params=lm_head_params,
                lr=unembedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.96),
                eps=1e-10,
                weight_decay=0.01,
            ),
            dict(
                kind="adamw",
                params=embedding_params,
                lr=embedding_lr * dmodel_lr_scale,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.001,
            ),
            dict(
                kind="adamw",
                params=value_embeds_params,
                lr=embedding_lr * dmodel_lr_scale * 0.5,
                betas=(0.8, 0.995),
                eps=1e-10,
                weight_decay=0.01,
            ),
            dict(
                kind="adamw",
                params=resid_params,
                lr=scalar_lr * 0.01,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.05,
            ),
            # higher beta1 for x0
            dict(
                kind="adamw",
                params=x0_params,
                lr=scalar_lr,
                betas=(0.96, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
            dict(
                kind="adamw",
                params=smear_params,
                lr=0.2,
                betas=(0.8, 0.95),
                eps=1e-10,
                weight_decay=0.0,
            ),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(
                dict(
                    kind="muon",
                    params=group_params,
                    lr=matrix_lr,
                    momentum=0.95,
                    ns_steps=5,
                    beta2=0.9,
                    weight_decay=weight_decay,
                )
            )

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction="mean"):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), (
            f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        )
        assert idx.device == self.cos.device, (
            f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        )
        assert self.cos.dtype == COMPUTE_DTYPE, (
            f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        )
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        # truncate cache to current sequence length
        cos_sin = (
            self.cos[:, T0 : T0 + T],
            self.sin[:, T0 : T0 + T],
        )

        # Embed the tokens
        x = self.wte(idx)  # embed current token
        # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = x.to(COMPUTE_DTYPE)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                self.smear_gate(x[:, 1:, :24])
            )
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, 1:, :24])
                )
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(
                    self.smear_gate(x[:, :, :24])
                )
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.blocks):
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = (
                self.value_embeds[str(i)](idx).to(x.dtype)
                if str(i) in self.value_embeds
                else None
            )
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        softcap = 15  # smoothly cap the logits to the range [-softcap, softcap]
        # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = self.lm_head(x)
        logits = logits[..., : self.config.vocab_size]  # slice to remove padding
        logits = logits.float()  # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap)  # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
                reduction=loss_reduction,
            )
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device)  # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids)  # (B, T, vocab_size)
            logits = logits[:, -1, :]  # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
