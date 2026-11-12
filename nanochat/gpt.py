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
# [STUDY] [NOTE] FILE OVERVIEW
# [STUDY]   Role: defines the whole model -- config, the attention/MLP blocks, weight init, the forward pass
# [STUDY]   (training loss or inference logits), optimizer parameter grouping, and FLOP/param accounting used
# [STUDY]   by the scaling-law scripts. It is a decoder-only, pre-norm Transformer with a long list of
# [STUDY]   deviations from the 2017 original (see the author's bullet list above).
# [STUDY]   Relations to other modules:
# [STUDY]     - nanochat/flash_attention.py : the only place attention is actually computed. Exposes the
# [STUDY]       Flash-Attention-3 API and falls back to F.scaled_dot_product_attention (SDPA).
# [STUDY]     - nanochat/engine.py (KVCache): inference-time cache. This file reads kv_cache.get_pos(),
# [STUDY]       kv_cache.prev_embedding, kv_cache.get_layer_cache() and calls kv_cache.advance().
# [STUDY]     - nanochat/optim.py (MuonAdamW): consumes the param_groups built in setup_optimizer().
# [STUDY]     - nanochat/common.py (COMPUTE_DTYPE): bf16 on Ampere+ CUDA, else fp32. There is NO autocast;
# [STUDY]       the dtype is threaded by hand (see class Linear).
# [STUDY]     - scripts/base_train.py, nanochat/checkpoint_manager.py: build GPT under torch.device("meta"),
# [STUDY]       then model.to_empty(device), then model.init_weights().
# [STUDY]   Papers referenced below (verified this session against the arXiv text unless tagged [PAPER?]):
# [STUDY]     Attention Is All You Need (Vaswani 2017); RoFormer (Su 2021); Root Mean Square Layer
# [STUDY]     Normalization (Zhang 2019); Query-Key Normalization for Transformers (Henry 2020); GQA
# [STUDY]     (Ainslie 2023); Value Residual Learning (Zhou 2024); Primer (So 2021); Gemma 2 (Riviere 2024) and
# [STUDY]     Neural Combinatorial Optimization with RL (Bello 2016) for logit soft-capping; PaLM
# [STUDY]     (Chowdhery 2022), Chinchilla (Hoffmann 2022) and Kaplan 2020 for FLOP/param counting;
# [STUDY]     Using the Output Embedding to Improve Language Models (Press 2017); GPT-3 (Brown 2020) and
# [STUDY]     Longformer (Beltagy 2020) for windowed attention; Hierarchical Neural Story Generation
# [STUDY]     (Fan 2018) for top-k sampling; the Muon blog post (Jordan 2024).
# [STUDY]   Not backed by a paper I could find: resid/x0 lambdas, smear, backout (see [PAPER?] notes).
# [STUDY]   Suggested reading order: GPTConfig -> norm / Linear -> apply_rotary_emb ->
# [STUDY]     CausalSelfAttention -> MLP / Block -> GPT.__init__ -> init_weights -> forward ->
# [STUDY]     setup_optimizer -> estimate_flops and friends -> generate.
# [STUDY]   Shape legend used throughout: B=batch, T=sequence length, C=n_embd (residual width),
# [STUDY]     H=n_head, Hkv=n_kv_head, D=head_dim=C/H, V=vocab size.
# [STUDY]   All [PITFALL] notes were checked by running a script on PyTorch 2.9.1+cu128.

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"


# [STUDY] [PAPER] Root Mean Square Layer Normalization (Zhang 2019), Sec 4 "RMSNorm", Eq. 4:
# [STUDY]     a_bar_i = a_i / RMS(a) * g_i,   RMS(a) = sqrt( (1/n) * sum_i a_i^2 )
# [STUDY] [DIFF] The paper keeps a learnable per-channel gain g_i. Here there is none ("no learnable params
# [STUDY]   in rmsnorm" in the docstring): the function is a pure projection onto the sphere of radius
# [STUDY]   sqrt(n). Unlike LayerNorm it does not subtract the mean either.
# [STUDY] [MATH] x: (..., n) -> same shape, with mean(y^2) over the last dim == 1, i.e. ||y||_2 = sqrt(n).
# [STUDY]   Why it matters downstream: the Linears that read the residual stream (c_q, c_k, c_v, c_fc,
# [STUDY]   lm_head) see unit-RMS input, so an init with std = 1/sqrt(n) gives unit-variance outputs (see
# [STUDY]   init_weights); the two c_proj and the gate Linears do not. For q/k it turns the dot
# [STUDY]   product into sqrt(D)-scaled cosine similarity (see the QK-norm note in CausalSelfAttention).
# [STUDY] [PITFALL] (verified, torch 2.9.1) F.rms_norm(x, shape) with eps omitted computes
# [STUDY]   x * rsqrt(mean(x^2) + eps) with eps = torch.finfo(float32).eps ~= 1.19e-7 (bit-exact match
# [STUDY]   in fp32). bf16 and fp16 inputs also behave as eps ~= 1.19e-7, NOT their own finfo eps
# [STUDY]   (finfo(bfloat16).eps = 0.0078): an all-1e-3 vector comes out as 0.945 in fp32, bf16 and fp16,
# [STUDY]   on both CPU and CUDA. fp64 does use its own, smaller eps (the same vector gives 1.0000).
# [STUDY]   Consequences: (a) an all-zero vector maps to zeros, not NaN; (b) vectors with RMS below ~1e-3
# [STUDY]   are NOT brought to unit RMS (1e-4 -> 0.28).
# [STUDY]   The output dtype follows the input dtype (bf16 in -> bf16 out), hence the author's remark.
def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

# [STUDY] [NOTE] Manual mixed precision. The usual recipe is `with torch.autocast(...)`, which leaves
# [STUDY]   weights in fp32 and casts inside each op. Here the activations themselves are created in
# [STUDY]   COMPUTE_DTYPE (the embedding tables are stored in bf16) and each Linear casts its fp32 weight
# [STUDY]   to the activation dtype on every forward. The cast is a differentiable op, so the gradient
# [STUDY]   flows back to the fp32 "master" weight; the optimizer therefore accumulates small updates in
# [STUDY]   fp32 even though the matmul ran in bf16 (bf16 has an 8-bit significand, so an update smaller
# [STUDY]   than ~0.4% of a weight's magnitude would be rounded away if the weight itself were bf16).
# [STUDY] [PITFALL] (verified, torch 2.9.1)
# [STUDY]   - With a bf16 input: output is bf16, weight.grad is fp32.
# [STUDY]   - When x is already fp32, weight.to(dtype=x.dtype) returns the very same tensor object (no copy).
# [STUDY]   - forward() never passes self.bias to F.linear. Constructing Linear(..., bias=True) silently
# [STUDY]     creates a bias parameter that is ignored and receives no gradient (grad stays None). Every
# [STUDY]     call site in this file passes bias=False, so this is latent rather than active.
# [STUDY]   - A plain nn.Linear with fp32 weights given a bf16 input raises
# [STUDY]     "expected m1 and m2 to have the same dtype"; that is why the subclass is needed at all.
# [STUDY]   - GPT.num_matmul_params() counts parameters via isinstance(m, Linear), so replacing this class
# [STUDY]     with nn.Linear anywhere would silently break the FLOP estimate.
class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


# [STUDY] [NOTE] Parity trick: a layer has a value embedding iff its index has the same parity as the
# [STUDY]   last layer's index. So the last layer always has one, then every other layer going backwards
# [STUDY]   (n_layer=12 -> layers 1,3,5,7,9,11; n_layer=5 -> layers 0,2,4).
def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

# [STUDY] [PAPER] RoFormer: Enhanced Transformer with Rotary Position Embedding (Su 2021),
# [STUDY]   Sec 3.2.1 "A 2D case" (Eq. 12-13), Sec 3.2.2 "General form" (Eq. 14-16),
# [STUDY]   Sec 3.4.2 "Computational efficient realization of rotary matrix multiplication" (Eq. 34).
# [STUDY] [MATH] Shapes: x: (B, T, H, D) queries or keys; cos, sin: (1, T, 1, D/2), broadcast over B and H.
# [STUDY]   d = D/2 here (NOT the paper's d, which is the full dimension).
# [STUDY]   Channel i in [0, d) is paired with channel i+d, and the pair is rotated at position m by the
# [STUDY]   angle m*theta_i, theta_i = base^(-2i/D) (built in _precompute_rotary_embeddings):
# [STUDY]       [y1_i]   [ cos(m*theta_i)   sin(m*theta_i)] [x1_i]
# [STUDY]       [y2_i] = [-sin(m*theta_i)   cos(m*theta_i)] [x2_i]        =: R(-m*theta_i) [x1_i, x2_i]^T
# [STUDY]   Why this encodes RELATIVE position: rotations are orthogonal and compose additively, so for a
# [STUDY]   query at position m and a key at position n
# [STUDY]       <R(-m*t) q, R(-n*t) k> = q^T R(-m*t)^T R(-n*t) k = q^T R((m-n)*t) k
# [STUDY]   which depends on the positions only through m-n (paper Eq. 16). Nothing is added to the
# [STUDY]   residual stream; position enters only through the q.k dot products, and v is not rotated.
# [STUDY]   Rotation preserves the norm of each pair, so it commutes with the RMS norm applied right after
# [STUDY]   (norm(R q) == R norm(q) up to rounding); the order of the two lines in forward() is immaterial.
# [STUDY] [DIFF] vs. the paper:
# [STUDY]   (1) Pairing. The paper rotates ADJACENT channels (x1,x2), (x3,x4), ... (Eq. 15 / Eq. 34); this
# [STUDY]       code pairs the first half with the second half. The two differ by a fixed permutation of
# [STUDY]       channels applied identically to q and k, which the learned c_q / c_k can absorb, so the
# [STUDY]       model family is the same; only checkpoints are incompatible between conventions.
# [STUDY]   (2) Sign. Paper Eq. 13 uses [[cos, -sin], [sin, cos]] (rotate by +m*theta); this code uses the
# [STUDY]       transpose, as the author's own comment says. By the derivation above only the relative
# [STUDY]       angle changes sign, which is again absorbed by the learned weights.
# [STUDY]   (3) Base. The paper sets theta_i = 10000^(-2(i-1)/d), i in [1, d/2] (text after Eq. 15); the
# [STUDY]       default here is base=100000.
def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        # [STUDY] [PAPER] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints
        # [STUDY]   (Ainslie 2023), Sec 2.2 "Grouped-query attention": the H query heads are split into G groups
        # [STUDY]   and each group shares one key head and one value head; GQA-1 is multi-query attention and
        # [STUDY]   GQA-H is ordinary multi-head attention. Here G = n_kv_head, so c_k / c_v emit only
        # [STUDY]   Hkv*D channels while c_q emits H*D. The main saving is the KV cache (see kv_bytes_per_token),
        # [STUDY]   i.e. inference memory bandwidth. c_k / c_v also shrink by a factor H/Hkv, a minor parameter
        # [STUDY]   and FLOP saving; the attention matmuls themselves still run over all H query heads.
        # [STUDY] [DIFF] The module docstring spells it "Group-Query Attention"; the paper's name is
        # [STUDY]   "Grouped-query attention".
        # [STUDY] [NOTE] scripts/base_train.py builds the config with n_kv_head == n_head, so by default this is
        # [STUDY]   plain multi-head attention and the GQA path is exercised only if a config sets Hkv < H.
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        # [STUDY] [NOTE] The gate reads only x[..., :12] of the (normed) block input, so it costs 12*Hkv parameters
        # [STUDY]   instead of C*Hkv. History (git): the gate arrived with 32 channels in commit e85db6b
        # [STUDY]   ("alternating design"); the autoresearch commit 6ed7d1d changed 32 -> 12. smear_gate's 24
        # [STUDY]   came from autoresearch round 2 (a825e63). So the small-probe design is the author's and the
        # [STUDY]   specific widths are tuning results.
        # [STUDY] [UNCLEAR] Why a prefix slice rather than a learned low-rank projection of all C channels is not
        # [STUDY]   stated anywhere in the repo. My guess: it is the cheapest possible probe (no extra matmul over
        # [STUDY]   C), and the channels are not special a priori; the model simply has to learn to route gating
        # [STUDY]   information into the first 12 coordinates of the residual stream.
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # [STUDY] [PAPER] Value Residual Learning (Zhou 2024, arXiv 2410.17897; earlier versions were titled
        # [STUDY]   "Value Residual Learning For Alleviating Attention Concentration In Transformers"),
        # [STUDY]   Sec 3.2 "ResFormer", Eq. 5:   V_n = lambda_{n,1} * V_1 + lambda_{n,2} * H_{n-1} W^V_n
        # [STUDY]   i.e. every layer's value is mixed with the FIRST layer's value V_1 = H_0 W^V_1.
        # [STUDY] [DIFF] This code is only loosely "ResFormer":
        # [STUDY]   - The residual is not layer 1's V. It is a separate token-indexed lookup table per VE layer
        # [STUDY]     (GPT.value_embeds[str(i)], shape (padded_V, Hkv*D)), so ve depends only on the token id at
        # [STUDY]     that position, not on context. The paper has no such table (its V_1 is a projection of H_0).
        # [STUDY]   - Only alternating layers get one (has_ve), and the coefficient on the layer's own value is
        # [STUDY]     fixed at 1 instead of a second lambda.
        # [STUDY]   - The mixing coefficient is not a scalar: it is an input-dependent gate per kv-head,
        # [STUDY]     gate = 3*sigmoid(W_g x[..., :12]) in (0, 3), computed from the block's normed input.
        # [STUDY]   dev/LOG.md attributes the design to modded-nanogpt's "Value Embeddings" and links this paper.
        # [STUDY] [MATH] Shapes: ve: (B, T, Hkv*D) -> (B, T, Hkv, D); gate: (B, T, Hkv) -> unsqueeze -> (B, T, Hkv, 1),
        # [STUDY]   broadcast over D; v: (B, T, Hkv, D). Attention is linear in v, so the output for
        # [STUDY]   head h at position t becomes sum_s A[t,s] * (v_s + gate_s * ve_s): each attended position
        # [STUDY]   contributes its raw token identity directly, bypassing whatever the residual stream has turned
        # [STUDY]   that position into. The table adds padded_V*Hkv*D parameters per VE layer at zero matmul FLOPs
        # [STUDY]   (it is a lookup), which is why dev/LOG.md calls it nearly free capacity.
        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        # [STUDY] [PAPER] Query-Key Normalization for Transformers (Henry 2020), Sec 2 Eq. 2 (L2-normalize q and
        # [STUDY]   k along the head dimension) and Sec 3 Eq. 5: softmax(g * Q_hat K_hat^T) V, replacing Eq. 4's
        # [STUDY]   softmax(Q K^T / sqrt(d)) V. g is a learnable scalar initialised by Eq. 3, g0 = log2(L^2 - L).
        # [STUDY] [MATH] Here norm() gives unit RMS rather than unit L2 norm: ||q|| = ||k|| = sqrt(D). Hence
        # [STUDY]   q.k = D*cos(angle), and the attention kernel then divides by sqrt(D) (SDPA does this by default;
        # [STUDY]   the FA3 kernel's flash_attn_interface.py likewise defaults softmax_scale to q.shape[-1]**-0.5), so
        # [STUDY]       logit = sqrt(D) * cos(angle)   in [-sqrt(D), +sqrt(D)].
        # [STUDY]   With the two 1.2 factors below: logit = 1.44 * sqrt(D) * cos(angle). E.g. for D = 128 the range
        # [STUDY]   is about +-16.3 instead of +-11.3. Bounded logits mean attention can never fully saturate: the
        # [STUDY]   largest possible ratio between two attention weights is exp(2 * 1.44 * sqrt(D)).
        # [STUDY] [DIFF] vs. the paper: the scale is a fixed constant (effective g = 1.44*sqrt(D)), not a learned
        # [STUDY]   parameter, and it grows with head_dim instead of with sequence length.
        # [STUDY] [UNCLEAR] The 1.2 is flagged "TODO think through better"; both lines and the TODO were added by
        # [STUDY]   the autoresearch commit a825e63. My reading: it is an
        # [STUDY]   auto-tuned inverse temperature (1.44x sharper attention); splitting it across q and k
        # [STUDY]   rather than using 1.44 on one side makes no mathematical difference to the logits, and may just
        # [STUDY]   keep the cached keys and the queries at the same magnitude. That last part is speculation.
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        # [STUDY] [NOTE] Layout: q: (B, T, H, D), k, v: (B, T, Hkv, D) -- heads on axis 2. This is FA3's native
        # [STUDY]   layout; the textbook / SDPA layout is (B, H, T, D), and the fallback in flash_attention.py
        # [STUDY]   transposes in and out. window_size = (left, 0): each query sees itself plus `left` previous
        # [STUDY]   positions (the SDPA fallback implements this as the mask 0 <= row - col <= left).
        # [STUDY]   y: (B, T, H, D), the per-head weighted averages of v (Vaswani 2017, Sec 3.2.1, Eq. 1).
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # [STUDY] [NOTE] Hidden side effect: a model forward mutates the cache. All layers must see the same
            # [STUDY]   cache_seqlens while they insert their k/v at [pos, pos+T), so the position is bumped exactly
            # [STUDY]   once, by the last layer. GPT.forward reads kv_cache.get_pos() for the rotary offset BEFORE
            # [STUDY]   running the blocks, which is why this ordering is safe.
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        # [STUDY] [PITFALL] (verified, torch 2.9.1) .view() needs strides compatible with the new shape; on a
        # [STUDY]   transposed tensor it raises "view size is not compatible with input tensor's size and stride".
        # [STUDY]   The SDPA fallback returns y.transpose(1, 2), hence the .contiguous() guard. .contiguous()
        # [STUDY]   returns the same object when the tensor is already contiguous (no copy) and copies otherwise;
        # [STUDY]   .reshape() would be the one-call equivalent. Merging (H, D) -> C is the "concat heads" step.
        # [STUDY]   Whether the fallback's result really is non-contiguous depends on the SDPA backend (observed:
        # [STUDY]   non-contiguous on CPU and on CUDA bf16 with is_causal; already contiguous on CUDA fp32 and on
        # [STUDY]   CUDA bf16 with an explicit mask), so the guard is needed for portability, not on every path.
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


# [STUDY] [PAPER] Primer: Searching for Efficient Transformers for Language Modeling (So 2021), Sec 3
# [STUDY]   "Primer", paragraph "Squared ReLU": found by architecture search, described there as the most
# [STUDY]   effective single modification; replaces ReLU in the feed-forward block with relu(x)^2.
# [STUDY] [MATH] f(x) = max(0, x)^2, f'(x) = 2*max(0, x). Unlike ReLU the derivative is continuous at 0
# [STUDY]   and grows linearly, and the function is quadratic rather than linear for large x (verified
# [STUDY]   numerically: f(-1, 0, 0.5, 2) = (0, 0, 0.25, 4), grad = (0, 0, 1, 4)).
# [STUDY]   Shapes: x: (B, T, C) -> c_fc -> (B, T, 4C) -> relu^2 -> c_proj -> (B, T, C). The 4x expansion
# [STUDY]   is the original Transformer's ratio (Vaswani 2017, Sec 3.3: d_ff = 2048 for d_model = 512).
# [STUDY] [DEBATE] Activation choice. GELU is the GPT-2/BERT default; SwiGLU (gated, 3 matrices at 8/3x
# [STUDY]   width) is common in recent LLMs; Primer argues for ReLU^2. The evidence in this repo is
# [STUDY]   empirical and local: dev/LOG.md records a SwiGLU swap at d12 and d24 that was worse on steps,
# [STUDY]   wall-clock and FLOPs, and a leaky_relu(x, 0.5)^2 variant that was slightly better per step but
# [STUDY]   slower, net worse on wall-clock. Those results are for this architecture/optimizer (Muon,
# [STUDY]   zero-init c_proj, no biases) and need not transfer elsewhere.
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


# [STUDY] [PAPER] Pre-norm residual block: x + f(norm(x)). The original Transformer is post-norm:
# [STUDY]   "LayerNorm(x + Sublayer(x))" (Vaswani 2017, Sec 3.1). GPT-2 and GPT-3 use pre-normalization
# [STUDY]   (Brown 2020, Sec 2.1 says so explicitly).
# [STUDY] [PAPER] On Layer Normalization in the Transformer Architecture (Xiong 2020), Sec 3.3
# [STUDY]   "Understanding the Transformer at initialization", Theorem 1: at init the gradient norm of
# [STUDY]   the last layer's weights is bounded by O(d*sqrt(ln d)) for Post-LN, independent of depth L,
# [STUDY]   but by O(d*sqrt(ln d / L)) for Pre-LN. This is the paper's argument for why Post-LN needs the
# [STUDY]   LR warm-up stage (its Sec 3.2) and Pre-LN can drop it.
# [STUDY] [DEBATE] Pre-norm keeps an un-normalized identity path from input to output, which makes deep
# [STUDY]   stacks easy to optimize; the price is that the residual stream's scale is unconstrained, so a
# [STUDY]   final norm before the head is mandatory (see `x = norm(x)` at the end of GPT.forward).
# [STUDY]   Post-norm re-normalizes the stream after every sublayer, which some report gives better final
# [STUDY]   quality when it can be trained at all, but it is more sensitive to warm-up and init. This
# [STUDY]   code additionally leans on the identity path: with zero-init c_proj every block is exactly
# [STUDY]   the identity at step 0 (see init_weights).
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
    # [STUDY] [PITFALL] (verified, torch 2.9.1) Callers construct this under `with torch.device("meta"):`.
    # [STUDY]   Inside that context nn.Embedding weights, nn.Parameter(torch.ones(n)) and even torch.arange
    # [STUDY]   all land on device "meta": they carry shape/dtype but no storage. Consequences:
    # [STUDY]   - .item() on such a tensor raises "Tensor.item() cannot be called on meta tensors", so no
    # [STUDY]     data-dependent Python control flow is possible in here.
    # [STUDY]   - module.to(device) raises NotImplementedError ("Cannot copy out of meta tensor; no data!");
    # [STUDY]     the caller must use module.to_empty(device=...), which allocates UNINITIALIZED memory.
    # [STUDY]   - Therefore every value assigned in __init__ (torch.ones, torch.zeros, 0.2 * torch.ones,
    # [STUDY]     PyTorch's default Linear/Embedding init, the cos/sin tables) is discarded. A parameter that
    # [STUDY]     init_weights() forgets is garbage, not its "fake init" value. Commit 94b73ad in this repo's
    # [STUDY]     history ("fix: initialize smear and backout lambdas in init_weights") is exactly that bug.
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
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)]),
        })
        # [STUDY] [DEBATE] Untied input/output embeddings.
        # [STUDY]   - Tied: Using the Output Embedding to Improve Language Models (Press 2017) recommends sharing
        # [STUDY]     the input embedding and the output projection; Vaswani 2017, Sec 3.4 does it ("we share the
        # [STUDY]     same weight matrix between the two embedding layers and the pre-softmax linear
        # [STUDY]     transformation"). Arguments: saves V*C parameters and acts as a regularizer.
        # [STUDY]   - Untied (this file): the two matrices play different roles and can then get different init
        # [STUDY]     (std 0.8 vs 0.001), different optimizer settings (lr 0.2 vs 0.004, see setup_optimizer) and
        # [STUDY]     different dtypes (wte is stored in bf16, lm_head stays an fp32 Linear). The input side is
        # [STUDY]     a lookup, so the extra V*C parameters add memory but no matmul FLOPs.
        # [STUDY]   Which is better is scale- and budget-dependent; this repo does not, as far as I found in
        # [STUDY]   dev/LOG.md, record a head-to-head comparison.
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        # [STUDY] [PAPER?] resid_lambdas / x0_lambdas: I know of no paper for this exact form. The author credits
        # [STUDY]   modded-nanogpt (a community speedrun repo). Feeding the embedding x0 back into every layer is
        # [STUDY]   reminiscent of dense/skip connections, but I am not citing a specific paper because I could
        # [STUDY]   not tie the design to one. Usage: x = resid_lambdas[i] * x + x0_lambdas[i] * x0 in forward().
        # [STUDY] [DIFF] The author's comment below says "init 1.0 = neutral" and "init 0.0 = disabled", and the
        # [STUDY]   fake inits here match that, but init_weights() actually sets resid_lambdas to a linear ramp
        # [STUDY]   1.15 -> 1.05 and x0_lambdas to 0.20 -> 0.05 across depth. The real init is neither neutral
        # [STUDY]   nor disabled.
        # Per-layer learnable scalars (inspired by modded-nanogpt)
        # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
        # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
        # Separate parameters so they can have different optimizer treatment
        self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
        self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
        # [STUDY] [PAPER?] Smear and backout: no paper found. dev/LOG.md lists both as ideas taken from
        # [STUDY]   modded-nanogpt and, in an early entry, rated them "negligible" / "no improvement"; git history
        # [STUDY]   shows they were adopted later (commit a825e63, "Autoresearch round 2: smear, backout, and
        # [STUDY]   hyperparameter tuning"). See forward() for what they compute.
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Backout: subtract cached mid-layer residual before final norm to remove low-level features
        self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # [STUDY] [NOTE] buffer vs parameter: a buffer is module state that follows .to(device/dtype) and
        # [STUDY]   to_empty() but is not returned by parameters() (so no gradient, no optimizer). persistent=False
        # [STUDY]   additionally keeps it out of state_dict(), so checkpoints do not store the tables and loading
        # [STUDY]   never complains about them; the cost is that they must be recomputed (init_weights()).
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    # [STUDY] [MATH] Uniform vs normal with matched std. For U(-s, s): Var = (2s)^2 / 12 = s^2 / 3, so
    # [STUDY]   std = sigma requires s = sqrt(3) * sigma. With sigma = 1/sqrt(n_embd) and unit-RMS input x
    # [STUDY]   (c_q, c_k, c_v and c_fc take their input straight out of norm()), each output coordinate
    # [STUDY]   y_j = sum_i W_ji x_i has Var(y_j) = n_embd * sigma^2 * mean(x_i^2) = 1: activations keep unit
    # [STUDY]   scale at init. The stated reason for uniform is bounded support (verified on 1e6 samples with
    # [STUDY]   sigma = 0.1: max |w| = 0.173 = sqrt(3)*sigma for uniform vs 0.469 for normal).
    # [STUDY] [MATH] Zero-init of both c_proj matrices makes every block output exactly 0, so at step 0 the
    # [STUDY]   stream is just x0 times a positive scalar, which the final norm removes: the logits are
    # [STUDY]   softcap(lm_head(norm(wte(idx)))), a function of the current token only. For the same reason
    # [STUDY]   resid_lambdas, x0_lambdas and backout_lambda get (numerically) zero gradient at step 0: they
    # [STUDY]   only change that scalar. Gradient at step 0: c_proj.weight gets a
    # [STUDY]   non-zero gradient (its input is non-zero), but everything upstream of it inside the block
    # [STUDY]   (c_q, c_k, c_v, c_fc, ve tables) gets exactly zero, because the backward signal is multiplied
    # [STUDY]   by c_proj^T = 0. They start learning from step 1 on. lm_head std = 0.001 makes the initial
    # [STUDY]   logits ~0, so the initial loss is ~ln(vocab_size) = ln(32768) = 10.40.
    # [STUDY] [PAPER?] Zero-initialising the last layer of each residual branch is a known family of tricks
    # [STUDY]   (Fixup, ReZero, zero-gamma in ResNets). I did not verify any of those sources this session,
    # [STUDY]   and the author cites none, so treat the attribution as unconfirmed.
    # [STUDY] [DIFF] The docstring below disagrees with the code in three places:
    # [STUDY]   - "wte: normal, std=1.0" -> code uses std=0.8.
    # [STUDY]   - "mlp.c_fc: uniform, std=1/sqrt(n_embd)" -> code uses 0.4x that bound.
    # [STUDY]   - value embeds, ve_gate, smear_gate and the per-layer scalars are not listed at all.
    # [STUDY] [NOTE] wte's std has no effect on the forward pass at init, because forward() applies norm()
    # [STUDY]   immediately after the lookup (RMS norm is scale-invariant). What it does change is the size
    # [STUDY]   of an optimizer step RELATIVE to the weights, i.e. the effective learning rate of the table.
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
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for block in self.transformer.h:
            torch.nn.init.uniform_(block.attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(block.attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(block.attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(block.attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(block.mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(block.mlp.c_proj.weight)

        # Per-layer scalars
        # Per-layer resid init: stronger residual at early layers, weaker at deep layers
        n_layer = self.config.n_layer
        for i in range(n_layer):
            # [STUDY] [PITFALL] (verified, torch 2.9.1) Indexed assignment into a leaf Parameter outside no_grad raises
            # [STUDY]   "a view of a leaf Variable that requires grad is being used in an in-place operation". Either
            # [STUDY]   no_grad() or .data avoids that; this method already runs under @torch.no_grad(), so the .data
            # [STUDY]   is redundant here (p[i] = v works under no_grad). .data additionally bypasses autograd's
            # [STUDY]   version counter, which is harmless at init time but is why it is discouraged elsewhere.
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
            torch.nn.init.uniform_(ve.weight, -s, s)

        # [STUDY] [DIFF] "slightly above neutral" is stale. Before commit 6ed7d1d the gate was 2*sigmoid(W x) with
        # [STUDY]   W = 0, i.e. exactly 1.0 (plain v + ve), and the comment read "scaled by 2 -> 1.0 (neutral)".
        # [STUDY]   That commit changed the factor to 3 and the init to U(0, 0.02), so the gate now starts at
        # [STUDY]   3*sigmoid(~0) ~= 1.5: about 50% above the old neutral, not slightly. The pre-activation is a
        # [STUDY]   sum of 12 terms of either sign (weights <= 0.02, unit-RMS inputs), so the "small positive"
        # [STUDY]   weights do not make it reliably positive either.
        # Gate weights init with small positive values so gates start slightly above neutral
        for block in self.transformer.h:
            if block.attn.ve_gate is not None:
                torch.nn.init.uniform_(block.attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        # [STUDY] [PITFALL] (verified, torch 2.9.1) Plain attribute assignment to a name that is already a
        # [STUDY]   registered buffer goes through nn.Module.__setattr__, which stores the new tensor back into
        # [STUDY]   self._buffers (not into __dict__). The buffer keeps its persistent=False status and still
        # [STUDY]   follows .to(). So this line replaces the buffer contents without re-registering.
        # [STUDY]   The check is asymmetric: assigning an nn.Parameter to a buffer name does not raise; it
        # [STUDY]   silently removes the buffer and registers a parameter of that name, which then shows up in
        # [STUDY]   parameters() and state_dict(). The reverse, a plain tensor assigned to a parameter name,
        # [STUDY]   raises TypeError.
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        # [STUDY] [PITFALL] (verified, torch 2.9.1) nn.Module.to(dtype=...) converts the module's parameters in
        # [STUDY]   place: it returns self and module.weight is still the same Parameter object, now bf16. That is
        # [STUDY]   why the return value can be dropped here. Tensor.to(dtype) is the opposite: it returns a new
        # [STUDY]   tensor and leaves the original untouched; writing `t.to(dtype)` without assignment is a no-op.
        # [STUDY]   An nn.Embedding stored in bf16 returns bf16 rows, which is how COMPUTE_DTYPE enters the
        # [STUDY]   network without autocast. Trade-off: these tables are their own master weights (no fp32
        # [STUDY]   copy), so their precision is bf16's; the author's comment above asserts that is tolerable.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    # [STUDY] [MATH] channel_range = (0, 2, 4, ..., D-2): (D/2,)
    # [STUDY]   inv_freq[i] = base^(-2i/D) = theta_i: (D/2,)   -- geometric progression from 1 down to ~1/base
    # [STUDY]   freqs[m, i] = m * theta_i: (seq_len, D/2)      -- the rotation angle for position m, pair i
    # [STUDY]   cos, sin: (1, seq_len, 1, D/2), broadcast against (B, T, H, D/2) halves in apply_rotary_emb.
    # [STUDY]   Pair i completes a full turn every 2*pi/theta_i positions: from ~6.3 tokens (i=0) up to
    # [STUDY]   ~2*pi*base tokens for the slowest pair, so every offset within the context gets a distinct
    # [STUDY]   rotation pattern. The trig is computed in fp32 and only then cast to COMPUTE_DTYPE.
    # [STUDY] [DIFF] RoFormer (Su 2021, text after Eq. 15) uses base 10000; the default here is 100000. The
    # [STUDY]   author's TODO below ("bump base theta more? e.g. 100K is more common") is stale: commit
    # [STUDY]   6ed7d1d changed the default from 10000 to 100000 and left the TODO in place.
    # [STUDY] [NOTE] Called twice: once in __init__ (under the meta device: shapes only) and once in
    # [STUDY]   init_weights() (real data). The table covers rotary_seq_len = 10 * sequence_len positions.
    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    # [STUDY] [PAPER] Language Models are Few-Shot Learners (Brown 2020), Sec 2.1: "we use alternating dense
    # [STUDY]   and locally banded sparse attention patterns in the layers of the transformer, similar to the
    # [STUDY]   Sparse Transformer". dev/LOG.md (2026-01-11) names GPT-3 as the inspiration. The sliding
    # [STUDY]   window itself is also described in Longformer (Beltagy 2020), Sec 3.1 "Attention Pattern".
    # [STUDY] [NOTE] Information still propagates across the full context: stacking k short layers reaches
    # [STUDY]   roughly k * short_window tokens back, and every len(pattern)-th layer plus the last one is
    # [STUDY]   full-context. Attention cost in an S layer drops from O(T) to O(window) per token with FA3;
    # [STUDY]   the SDPA fallback in flash_attention.py instead masks a full T x T computation.
    # [STUDY] [DIFF] Three mismatches between comments and code in this function and its callers:
    # [STUDY]   - "(-1, 0) for full context" (comments in GPT.__init__ and in CausalSelfAttention.forward):
    # [STUDY]     the code never produces -1; "L" maps to (sequence_len, 0). Equivalent while T <= sequence_len.
    # [STUDY]     But the rotary table allows up to 10x sequence_len at inference, and there an "L" layer
    # [STUDY]     becomes a sliding window of sequence_len rather than truly unlimited context.
    # [STUDY]   - "S=short (quarter context)" vs dev/LOG.md's original "Short window = sequence_len // 2":
    # [STUDY]     the code is a quarter; the log entry describes an earlier version. Git history: half
    # [STUDY]     (fbc1484), then a third (6ed7d1d), then a quarter (a825e63).
    # [STUDY]   - The inline "(2048 -> 768)" below is wrong: see the note on that line.
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
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        # [STUDY] [DIFF] (verified by evaluating the expression) -(-2048 // 4 // 128) * 128 == 512, not 768 as the
        # [STUDY]   trailing comment claims. Unary minus binds tighter than //, and // is left-associative, so
        # [STUDY]   this is -(((-L) // 4) // 128) * 128 = ceil(L/4/128) * 128: a quarter of the context rounded UP
        # [STUDY]   to a multiple of 128 (floor division of a negative number rounds towards -inf, the standard
        # [STUDY]   integer-ceil idiom). Other values: 1024 -> 256, 256 -> 128, 3000 -> 768. The 768 was correct
        # [STUDY]   for the `// 3` version introduced in commit 6ed7d1d; commit a825e63 changed the divisor to 4
        # [STUDY]   without updating the comment. Side effect of rounding up: for sequence_len < 512 the "short"
        # [STUDY]   window is not shorter than the long one (sequence_len = 64 -> S = 128), so S layers are
        # [STUDY]   effectively full-context.
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
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
        return self.transformer.wte.weight.device

    # [STUDY] [PAPER] PaLM: Scaling Language Modeling with Pathways (Chowdhery 2022), Appendix B: "an
    # [STUDY]   N-parameter decoder-only model requires 6N matmul FLOPs per token seen (2N for forward and
    # [STUDY]   4N for backward)" and "the matmuls in dense self-attention add 6LH(2QT) FLOPs per token",
    # [STUDY]   giving 6N + 12*L*H*Q*T. The 6N rule is credited there to Kaplan 2020 (Scaling Laws for Neural
    # [STUDY]   Language Models, Sec 2.1, Eq. 2.2: C_forward ~ 2N + 2*n_layer*n_ctx*d_model; Sec 1.3: C ~ 6NBS).
    # [STUDY] [MATH] Symbols -> code: N = num_matmul_params(), H = h, Q = q = head_dim, T = t (or the window).
    # [STUDY]   - A weight w_ji used once per token costs 1 multiply + 1 add = 2 FLOPs forward. Backward needs
    # [STUDY]     two matmuls per forward matmul (one for dL/dW, one for dL/dinput), so 4. Total 6 per weight.
    # [STUDY]   - Attention has matmuls with no parameters: per query token and head, q.k against T keys costs
    # [STUDY]     2*Q*T FLOPs, and the weighted sum of T values another 2*Q*T. Forward = 4*H*Q*T per layer
    # [STUDY]     (this is the 4*h*q*... in estimate_decode_flops); times 3 for forward+backward = 12*H*Q*T.
    # [STUDY]   - Here L is replaced by a per-layer sum because T_eff = min(window, T) differs per layer. The
    # [STUDY]     `window < 0` branch is dead code today (see the [DIFF] on _compute_window_sizes).
    # [STUDY] [DIFF] vs. the cited conventions: N here includes lm_head (a real matmul; Kaplan's N excludes
    # [STUDY]   it and lists de-embedding separately) and the tiny gate Linears, and excludes all lookup tables.
    # [STUDY]   Chinchilla (Hoffmann 2022, Appendix F "FLOPs computation") does count embedding matrices in
    # [STUDY]   both FLOPs and parameters, as the docstring says. Causal masking is ignored (a token at
    # [STUDY]   position t attends to t keys, T/2 on average), so the attention term is an upper bound for
    # [STUDY]   training; estimate_prefill_flops does the triangular count (approximately: the real mask lets
    # [STUDY]   a token see window+1 keys including itself, the estimate uses min(t, window)).
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
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
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
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        return matmul_params

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
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
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters())
        scalars = self.resid_lambdas.numel() + self.x0_lambdas.numel() + self.smear_gate.weight.numel() + self.smear_lambda.numel() + self.backout_lambda.numel()
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'scalars': scalars,
            'total': total,
        }

    # [STUDY] [PAPER] Muon: An optimizer for hidden layers in neural networks (Keller Jordan, blog post,
    # [STUDY]   Dec 8 2024, kellerjordan.github.io/posts/muon/). It targets "2D parameters of neural network
    # [STUDY]   hidden layers"; scalar/vector parameters and the input and output layers "should be optimized
    # [STUDY]   by a standard method such as AdamW". That is the split below: everything under transformer.h
    # [STUDY]   (all 2D Linear weights, including each ve_gate of shape (Hkv, 12)) goes to Muon; wte,
    # [STUDY]   value_embeds, lm_head and the scalars go to AdamW. The math of both lives in nanochat/optim.py.
    # [STUDY] [NOTE] Muon groups are keyed by shape because optim.py stacks all same-shape matrices into one
    # [STUDY]   (K, rows, cols) tensor and orthogonalizes them in a single batched call.
    # [STUDY]   (Verified, torch 2.9.1: torch.Size is a tuple subclass, so a set of shapes sorts
    # [STUDY]   lexicographically, giving a deterministic group order.)
    # [STUDY] [MATH] dmodel_lr_scale = (n_embd/768)^-0.5: wider models get a smaller AdamW step. The same
    # [STUDY]   d_model^-0.5 factor appears in the original schedule, lrate = d_model^-0.5 * min(...)
    # [STUDY]   (Vaswani 2017, Sec 5.3, Eq. 3).
    # [STUDY] [DIFF] The comment says the scale applies to "the AdamW parameters", but only lm_head, wte and
    # [STUDY]   value_embeds use it; the three scalar groups (resid, x0, smear/backout) do not.
    # [STUDY] [NOTE] The `weight_decay` argument reaches only the Muon groups; the AdamW groups have
    # [STUDY]   hard-coded decays. "initial_lr" is stashed so the training loop can apply an LR schedule as a
    # [STUDY]   multiplier on it.
    # [STUDY] [NOTE] The per-group betas / eps / decay values are search results, not derivations. dev/LOG.md
    # [STUDY]   (the ~320-run optimizer sweep) documents two of them: resid_lambdas get 100x less LR than
    # [STUDY]   x0_lambdas (stated reason: a multiplicative scalar compounds through depth, an additive one
    # [STUDY]   does not), and x0's beta1 = 0.96 ("just x0_beta1=0.96 alone captured nearly all the gains";
    # [STUDY]   0.96 optimal at d20, 0.98 "catastrophic"). So betas=(0.96, 0.95), with beta1 > beta2, the
    # [STUDY]   reverse of the usual Adam ordering, is purely empirical with a sharp cliff next to it; the
    # [STUDY]   author offers no theory for it.
    # [STUDY] [UNCLEAR] The remaining per-group values (e.g. beta2 = 0.96 for lm_head, 0.995 for the embedding
    # [STUDY]   tables, the individual weight decays) are not explained in dev/LOG.md as far as I found; I
    # [STUDY]   assume they come from the later autoresearch commits but did not trace each one.
    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        matrix_params = list(self.transformer.h.parameters())
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [self.resid_lambdas]
        x0_params = [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda, self.backout_lambda]
        assert len(list(self.parameters())) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
            dict(kind='adamw', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            dict(kind='adamw', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0),
        ]
        # Muon groups (matrix params, grouped by shape for stacking)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    # [STUDY] [NOTE] Data flow and shapes. idx: (B, T) int64 token ids.
    # [STUDY]     wte -> x: (B, T, C) in COMPUTE_DTYPE -> norm -> smear -> x0
    # [STUDY]     for each layer i: x = resid_lambdas[i]*x + x0_lambdas[i]*x0;  x = block_i(x, ve_i, ...)
    # [STUDY]     backout -> norm -> lm_head -> logits: (B, T, padded_V) -> crop to V -> fp32 -> softcap
    # [STUDY]   targets given: returns a scalar loss (or (B*T,) per-token losses with loss_reduction='none').
    # [STUDY]   targets None: returns logits (B, T, V).
    # [STUDY]   With a kv_cache, idx holds only the NEW tokens and T0 = tokens already cached; the rotary
    # [STUDY]   tables are sliced at [T0, T0+T) so each new token is rotated by its absolute position.
    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean'):
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # [STUDY] [MATH] Smear: x_t <- x_t + lambda * sigmoid(w . x_t[:24]) * x_{t-1}   for t >= 1; x_0 unchanged.
        # [STUDY]   gate: (B, T-1, 1), broadcast over C; w = smear_gate.weight: (1, 24); lambda = smear_lambda: (1,).
        # [STUDY]   The gate looks at the CURRENT token (24 channels of its normed embedding) to decide how much
        # [STUDY]   of the PREVIOUS token's embedding to add. It only uses t-1, so causality is preserved. All
        # [STUDY]   x on the right-hand side are pre-smear values (the slices are taken before torch.cat), and
        # [STUDY]   the KV-cache path mirrors that by caching the pre-smear embedding in kv_cache.prev_embedding.
        # [STUDY]   The three code paths (full sequence / prefill / single-token decode) compute the same function,
        # [STUDY]   provided multi-token chunks are only fed into an EMPTY cache: the `if T > 1` branch ignores
        # [STUDY]   x_pre_smear, so a T > 1 forward on a non-empty cache would skip the smear for its first token.
        # [STUDY]   lambda is initialised to 0, so smear is off at init and dL/dw = 0 until lambda moves; lambda
        # [STUDY]   itself gets a non-zero gradient at step 0, proportional to sigmoid(.) * x_{t-1}.
        # [STUDY] [PITFALL] (verified, torch 2.9.1) Why `.to(x.dtype)` appears on smear_lambda/backout_lambda but
        # [STUDY]   not on resid_lambdas[i]: type promotion treats 0-dim tensors like Python scalars. A 0-dim
        # [STUDY]   fp32 tensor times a bf16 tensor stays bf16, but a shape-(1,) fp32 tensor times a bf16 tensor
        # [STUDY]   promotes the result to fp32. smear_lambda and backout_lambda have shape (1,), so without the
        # [STUDY]   cast the whole residual stream would silently become fp32 (and the next Linear would then run
        # [STUDY]   in fp32 too, since it follows the input dtype). resid_lambdas[i] is 0-dim and is safe.
        # [STUDY]   Same result with fp16.
        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            # [STUDY] [NOTE] This assert also constrains generate() below, which calls forward() without a kv_cache:
            # [STUDY]   a single-token prompt (e.g. just BOS) has T == 1 on the first step and trips it. The
            # [STUDY]   KV-cache path used by nanochat/engine.py handles T == 1 explicitly.
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        x0 = x  # save initial normalized embedding for x0 residual
        n_layer = self.config.n_layer
        backout_layer = n_layer // 2  # cache at halfway point
        x_backout = None
        for i, block in enumerate(self.transformer.h):
            # [STUDY] [MATH] x <- a_i * x + b_i * x0 with learned scalars a_i (init 1.15 -> 1.05 over depth) and
            # [STUDY]   b_i (init 0.20 -> 0.05). Unrolling over layers, a_i multiplies everything accumulated so far,
            # [STUDY]   so the a's compound as a product across depth, while b_i re-injects the (smeared, normed)
            # [STUDY]   token embedding x0 at every layer. Both are 0-dim after indexing: no dtype promotion.
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
            x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
            if i == backout_layer:
                x_backout = x
        # [STUDY] [MATH] Backout: x_final <- x_L - lambda * x_{L/2}, lambda = backout_lambda (init 0.2), where
        # [STUDY]   x_{L/2} =: x_mid is the stream right after block n_layer // 2. x_L contains x_mid scaled by
        # [STUDY]   P = prod_{i > mid} resid_lambdas[i] (plus x0 re-injections and later block outputs); backout
        # [STUDY]   changes that coefficient to P - lambda. It thus reduces the weight of everything written by
        # [STUDY]   blocks 0..n_layer//2 (and the embedding) relative to later blocks, just before the final norm.
        # [STUDY] [UNCLEAR] The stated motivation ("remove low-level features") is the author's interpretation;
        # [STUDY]   nothing here measures it. Alternative reading: it simply gives the model one cheap degree of
        # [STUDY]   freedom to rebalance early vs late layers at the output. I cannot distinguish the two from
        # [STUDY]   the code. dev/LOG.md's earlier trial cached at ~60% depth; this code uses 50%.
        # Subtract mid-layer residual to remove low-level features before logit projection
        if x_backout is not None:
            x = x - self.backout_lambda.to(x.dtype) * x_backout
        x = norm(x)

        # Forward the lm_head (compute logits)
        # [STUDY] [PAPER] Logit soft-capping: logits <- soft_cap * tanh(logits / soft_cap).
        # [STUDY]   - Gemma 2: Improving Open Language Models at a Practical Size (Riviere 2024), Sec 2 "Model
        # [STUDY]     Architecture", paragraph "Logit soft-capping": cap 50.0 in attention layers, 30.0 at the
        # [STUDY]     final layer; it cites Bello 2016 for the idea.
        # [STUDY]   - Neural Combinatorial Optimization with Reinforcement Learning (Bello 2016), Appendix A.2,
        # [STUDY]     Eq. 16: softmax(C * tanh(u)), with C = 10 in their experiments.
        # [STUDY] [DIFF] Here the cap is 15 and is applied only to the final logits (attention logits are bounded
        # [STUDY]   by QK norm instead). dev/LOG.md (2026-03-02) reports tuning it from 15 to 20; git history shows
        # [STUDY]   a later autoresearch commit (6ed7d1d) returned it to 15, and the LOG entry was not updated.
        # [STUDY] [MATH] f(z) = c*tanh(z/c), f'(z) = 1 - tanh^2(z/c) in (0, 1]. For |z| << c it is the identity
        # [STUDY]   (f(0.1) = 0.1000, f(5) = 4.82); it saturates smoothly at +-c (f(40) = 14.86, f(100) = 15.00).
        # [STUDY]   Effects: (1) every logit lies in (-15, 15), so any token's probability is at least
        # [STUDY]   e^-15 / (V * e^15) and the per-token loss is bounded by 30 + ln(V); a confidently wrong
        # [STUDY]   prediction cannot produce an arbitrarily large loss or gradient. (2) Saturated logits get a
        # [STUDY]   vanishing gradient, which discourages pushing logits ever further out.
        # [STUDY]   The cast to fp32 comes first because tanh near saturation and the log-sum-exp in the loss
        # [STUDY]   need more than bf16's ~3 significant decimal digits.
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            # [STUDY] [PITFALL] (verified, torch 2.9.1) F.cross_entropy with ignore_index:
            # [STUDY]   - reduction='mean' divides by the number of NON-ignored targets, not by B*T.
            # [STUDY]   - If every target is ignored the result is nan (0/0), not 0.
            # [STUDY]   - reduction='none' returns 0.0 at ignored positions, and 'sum' skips them.
            # [STUDY]   - The default ignore_index is -100; a target of -1 with the default raises
            # [STUDY]     "IndexError: Target -1 is out of bounds" on CPU. The -1 here is this repo's convention
            # [STUDY]     for masked positions and must match what the data loaders / tasks emit.
            # [STUDY]   logits: (B, T, V) -> (B*T, V); targets: (B, T) -> (B*T,). cross_entropy takes raw logits and
            # [STUDY]   applies log_softmax internally; do not pre-apply softmax.
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    # [STUDY] [NOTE] Reference sampler without a KV cache: every step re-runs the whole prefix, so producing
    # [STUDY]   n tokens costs O(n) full forwards. The efficient path is the Engine in nanochat/engine.py.
    # [STUDY] [PAPER] Top-k sampling: Hierarchical Neural Story Generation (Fan 2018), Sec 5.4 "Generation"
    # [STUDY]   ("a top-k random sampling scheme ... the k = 10 most likely" words). Temperature divides the
    # [STUDY]   logits before softmax: temperature < 1 sharpens, > 1 flattens, and temperature == 0 is handled
    # [STUDY]   separately as argmax.
    # [STUDY] [PITFALL] (verified, torch 2.9.1)
    # [STUDY]   - This is a generator function. torch.inference_mode() as a decorator detects that and
    # [STUDY]     re-enters the mode each time the generator is resumed: inside the body
    # [STUDY]     torch.is_inference_mode_enabled() is True, while between yields the caller's code is NOT
    # [STUDY]     in inference mode. Wrapping only the call (`with torch.inference_mode(): g = gen()`) would
    # [STUDY]     not cover the body at all, since the body runs later, on next().
    # [STUDY]   - inference_mode is stricter than no_grad: tensors created inside are "inference tensors".
    # [STUDY]     Using one in autograd later raises "Inference tensors cannot be saved for backward", and
    # [STUDY]     modifying one in place outside inference mode also raises. Fine here: only ints are yielded.
    # [STUDY]   - torch.Generator(device=...) must live on the same device type as the tensor being sampled:
    # [STUDY]     a CPU generator with CUDA probs raises "Expected a 'cuda' device type for generator".
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
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # [STUDY] [PITFALL] (verified, torch 2.9.1) v[:, [-1]] indexes with a list, which keeps the dimension:
                # [STUDY]   shape (B, 1), the k-th largest logit per row, broadcasting against logits (B, V) row-wise.
                # [STUDY]   v[:, -1] would have shape (B,) and broadcast against the LAST axis instead: an error when
                # [STUDY]   B != V, and a silent wrong answer (thresholds applied per column) when B == V.
                # [STUDY]   The masked assignment mutates `logits` in place; -inf becomes probability 0 after softmax.
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
