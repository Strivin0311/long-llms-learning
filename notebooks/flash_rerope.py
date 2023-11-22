# --------------------------------------------------------
# the flash_attn algorithm implemented in Triton is refered to:
# https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py
# and the flash attn version we support is 2.2.1
# NOTE that: 
#   the triton version we support is 2.1.0.dev20231014192330, 
#   which is the nightly version that can be installed with the command below:
#   pip install -U --index-url https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/Triton-Nightly/pypi/simple/ triton-nightly
# --------------------------------------------------------

import math
import torch

import flash_attn.flash_attn_interface as fi

import triton
import triton.language as tl



### NOTE: this right one let the computation of q1,k2,q2,k1 outside the kernel
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_with_fused_rerope_outter(
    Q1, K1, Q2, K2, V,
    Bias, Out,
    Lse, # shape: [bs, nh, q_len_round]
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug, with # shape: [bs, nh, q_len_round]
    softmax_scale,
    stride_q1b, stride_q1h, stride_q1m, # q_len
    stride_k1b, stride_k1h, stride_k1n, # kv_len
    stride_q2b, stride_q2h, stride_q2m, # q_len
    stride_k2b, stride_k2h, stride_k2n, # kv_len
    stride_vb, stride_vh, stride_vn, # kv_len
    stride_bb, stride_bh, stride_bm, # q_len
    stride_ob, stride_oh, stride_om, # q_len
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr, # usually d itself
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ## get the start id in m and n dim
    start_m = tl.program_id(0) # ith block of Q
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    
    ## initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # the ith block in q_len dim (outer loop)
    offs_n = tl.arange(0, BLOCK_N) # the initial 0th block in kv_len dim (inner loop)
    offs_d = tl.arange(0, BLOCK_HEADDIM) # the only one block in head_dim dim
    
    ## Initialize pointers to Q, K, V, and rotate-half Q, K
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q1_ptrs = (
        Q1 + off_b * stride_q1b + off_h * stride_q1h +
        (offs_m[:, None] * stride_q1m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    k1_ptrs = (
        K1 + off_b * stride_k1b + off_h * stride_k1h +
        (offs_n[:, None] * stride_k1n + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    q2_ptrs = (
        Q2 + off_b * stride_q2b + off_h * stride_q2h +
        (offs_m[:, None] * stride_q2m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    k2_ptrs = (
        K2 + off_b * stride_k2b + off_h * stride_k2h +
        (offs_n[:, None] * stride_k2n + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + 
        (offs_n[:, None] * stride_vn + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    
    ## initialize bias pointers
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias + off_b * stride_bb + off_h * stride_bh +
            (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
        
    ## initialize pointer to m, l, o
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # init to -inf for li
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # init to -inf for mi
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32) # init the empty output, with shpe: [block_sz_m, hd]
    
    ## load q1, q2 block
    # on different conditions whether the q_len / kv_len can be divided by block_sz_m / block_sz_n 
    # and they will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q1 = tl.load(q1_ptrs)
            q2 = tl.load(q2_ptrs)
        else:
            q1 = tl.load(q1_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q2 = tl.load(q2_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q1 = tl.load(q1_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q2 = tl.load(q2_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q1 = tl.load(q1_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q2 = tl.load(q2_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            
            
    ## loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # jth block of K,V
        
        # -- load k1, k2 ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k1 = tl.load(k1_ptrs + start_n * stride_k1n)
                k2 = tl.load(k2_ptrs + start_n * stride_k2n)
            else:
                k1 = tl.load(k1_ptrs + start_n * stride_k1n, mask=offs_d[None, :] < headdim, other=0.0)
                k2 = tl.load(k2_ptrs + start_n * stride_k2n, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k1 = tl.load(k1_ptrs + start_n * stride_k1n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                k2 = tl.load(k2_ptrs + start_n * stride_k2n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k1 = tl.load(k1_ptrs + start_n * stride_k1n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                k2 = tl.load(k2_ptrs + start_n * stride_k2n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                
        # -- compute qk1, qk2 ----
        qk1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) # shape: [block_sz_m, block_sz_n]
        qk2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) # shape: [block_sz_m, block_sz_n]
        # qk += tl.dot(q, k, trans_b=True) # Q * K^T => get wrong using triton-nightly 2.1.0
        qk1 += tl.dot(q1, tl.trans(k1)) # Q1 * K1^T
        qk2 += tl.dot(q2, tl.trans(k2)) # Q2 * K2^T
        
        # -- apply rectified mask to get qk --
        reM = tl.abs(offs_m[:, None] - (start_n + offs_n)[None, :]) < WINDOW
        qk = tl.where(reM, qk1, qk2)
        
        # -- apply causal mask --
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        
        # -- compute p and update mij, lij with bias applied if exists --
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1) # temporary lij as rowsum(Pij)

        # -- scale acc_o --
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale) # BUG: have to store
        acc_o_scale = tl.load(t_ptrs) # BUG: and immediately load
        acc_o = acc_o * acc_o_scale[:, None]
        
        # -- load v --
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        
        # -- update acc_o by PV --
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update mi and li --
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    ## scale the Oi
    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    
    ## store Li
    # re-materialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    
    ## initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out + off_b * stride_ob + off_h * stride_oh +
        (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    
    ## store Oi
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )

def _flash_attn_forward_with_fused_rerope_outter(
                                        q, k, v, 
                                        cos, sin, position_ids, window_size,
                                        bias=None, causal=False, softmax_scale=None):
    ## check constraints in shape, dtype, device and index boundary
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    max_position_len, _ = cos.shape
    
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert position_ids.shape == (batch, seqlen_q)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype == cos.dtype == sin.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda and cos.is_cuda and sin.is_cuda and position_ids.is_cuda
    assert position_ids.max() < max_position_len
    assert window_size <= seqlen_k
    assert window_size <= max_position_len
    
    ## prepare rerope q1, k1, q2, k2
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    
    rhq, rhk = rotate_half(q), rotate_half(k)
    cos1 = cos[position_ids].unsqueeze(2).expand((batch, seqlen_q, nheads, d))  
    sin1 = sin[position_ids].unsqueeze(2).expand((batch, seqlen_q, nheads, d))
    cos2 = cos[position_ids * 0 + window_size].unsqueeze(2).expand((batch, seqlen_q, nheads, d))
    sin2 = sin[position_ids * 0 + window_size].unsqueeze(2).expand((batch, seqlen_q, nheads, d)) 
    
    q1 = q * cos1 + rhq * sin1
    k1 = k * cos1 + rhk * sin1
    q2 = q * cos2 + rhq * sin2
    k2 = k
    
    ## prepare scaling factor
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    ## prepare bias
    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    ## prepare O, L, tmp
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128 # the closest q_len which is times of 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    ## prepare block / warp / grid size
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16) # usually d itself
    ## set block size
    BLOCK = 128
    # BLOCK = 256 # FIXME: out of resource
    
    ## set num of warps
    # num_warps = 4 if d <= 64 else 8
    num_warps = 4 # FIXME: when d = 128, not all close if num_warps = 8 as the line above
    
    ## set grid split
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    
    ## apply forward kernel with fused rope func for all blocks in grid
    _fwd_kernel_with_fused_rerope_outter[grid](
        q1, k1, q2, k2, v,
        bias, # shape: [bs, nh, q_len, kv_len]
        o, # shape: [bs, q_len, nh, hd]
        lse, tmp, # shape: [bs, nh, q_len_round]
        softmax_scale,
        q1.stride(0), q1.stride(2), q1.stride(1), # bs, nh, sq
        k1.stride(0), k1.stride(2), k1.stride(1), # bs, nh, sk
        q2.stride(0), q2.stride(2), q2.stride(1), # bs, nh, sq
        k2.stride(0), k2.stride(2), k2.stride(1), # bs, nh, sk
        v.stride(0), v.stride(2), v.stride(1), # bs, nh, sv
        # bias
        *bias_strides,
        o.stride(0), o.stride(2), o.stride(1), # bs, nh, sq
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d,
        seqlen_q // 32, seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type, causal, window_size, # rerope window size
        BLOCK_HEADDIM, BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


#### NOTE: this also right one let the computation of q1,k2,q2,k1 inside the kernel
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _fwd_kernel_with_fused_rerope_inner(
    Q, K, V, rhQ, rhK,
    Cos1, Sin1, Cos2, Sin2,
    Bias, Out,
    Lse, # shape: [bs, nh, q_len_round]
    TMP,  # NOTE: TMP is a scratchpad buffer to workaround a compiler bug, with # shape: [bs, nh, q_len_round]
    softmax_scale,
    stride_qb, stride_qh, stride_qm, # q_len
    stride_kb, stride_kh, stride_kn, # kv_len
    stride_vb, stride_vh, stride_vn, # kv_len
    stride_rhqb, stride_rhqh, stride_rhqm, # q_len
    stride_rhkb, stride_rhkh, stride_rhkn, # kv_len
    stride_qc1b, stride_qc1h, stride_qc1m, # q_len
    stride_qs1b, stride_qs1h, stride_qs1m, # q_len
    stride_kc1b, stride_kc1h, stride_kc1n, # kv_len
    stride_ks1b, stride_ks1h, stride_ks1n, # kv_len
    stride_qc2b, stride_qc2h, stride_qc2m, # q_len
    stride_qs2b, stride_qs2h, stride_qs2m, # q_len
    stride_bb, stride_bh, stride_bm, # q_len
    stride_ob, stride_oh, stride_om, # q_len
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr, WINDOW: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr, # usually d itself
    EVEN_M: tl.constexpr, EVEN_N: tl.constexpr, EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    ## get the start id in m and n dim
    start_m = tl.program_id(0) # ith block of Q
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    
    ## initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M) # the ith block in q_len dim (outer loop)
    offs_n = tl.arange(0, BLOCK_N) # the initial 0th block in kv_len dim (inner loop)
    offs_d = tl.arange(0, BLOCK_HEADDIM) # the only one block in head_dim dim
    
    ## Initialize pointers to Q, K, V, and rotate-half Q, K
    # Adding parenthesis around indexing might use int32 math instead of int64 math?
    # https://github.com/openai/triton/issues/741
    # I'm seeing a tiny bit of difference (5-7us)
    q_ptrs = (
        Q + off_b * stride_qb + off_h * stride_qh +
        (offs_m[:, None] * stride_qm + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    k_ptrs = (
        K + off_b * stride_kb + off_h * stride_kh +
        (offs_n[:, None] * stride_kn + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    v_ptrs = (
        V + off_b * stride_vb + off_h * stride_vh + 
        (offs_n[:, None] * stride_vn + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    rhq_ptrs = (
        rhQ + off_b * stride_rhqb + off_h * stride_rhqh +
        (offs_m[:, None] * stride_rhqm + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    rhk_ptrs = (
        rhK + off_b * stride_rhkb + off_h * stride_rhkh +
        (offs_n[:, None] * stride_rhkn + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    
    ## Initialize pointers to Cos1(q/k), Sin1(q/k), Cos2(q/k), Sin2(q/k) and rectified mask
    q_cos1_ptrs = (
        Cos1 + off_b * stride_qc1b + off_h * stride_qc1h +
        (offs_m[:, None] * stride_qc1m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    q_sin1_ptrs = (
        Sin1 + off_b * stride_qs1b + off_h * stride_qs1h +
        (offs_m[:, None] * stride_qs1m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    k_cos1_ptrs = (
        Cos1 + off_b * stride_kc1b + off_h * stride_kc1h +
        (offs_n[:, None] * stride_kc1n + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    k_sin1_ptrs = (
        Sin1 + off_b * stride_ks1b + off_h * stride_ks1h +
        (offs_n[:, None] * stride_ks1n + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    q_cos2_ptrs = (
        Cos2 + off_b * stride_qc2b + off_h * stride_qc2h +
        (offs_m[:, None] * stride_qc2m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    q_sin2_ptrs = (
        Sin2 + off_b * stride_qs2b + off_h * stride_qs2h +
        (offs_m[:, None] * stride_qs2m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    
    ## initialize bias pointers
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias + off_b * stride_bb + off_h * stride_bh +
            (offs_m[:, None] * stride_bm + offs_n[None, :])
        )
        
    ## initialize pointer to m, l, o
    t_ptrs = TMP + off_hb * seqlen_q_rounded + offs_m
    lse_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # init to -inf for li
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf") # init to -inf for mi
    acc_o = tl.zeros([BLOCK_M, BLOCK_HEADDIM], dtype=tl.float32) # init the empty output, with shpe: [block_sz_m, hd]
    
    ## load q block and its rotate_half with cos/sin rope
    # on different conditions whether the q_len / kv_len can be divided by block_sz_m / block_sz_n 
    # and they will stay in SRAM throughout
    # [2022-10-30] TD: Triton bug - in the case of EVEN_M=True and EVEN_N=False, if we just call
    # tl.load(q_ptrs), we get the wrong output!
    if EVEN_M & EVEN_N:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs)
            rhq = tl.load(rhq_ptrs)
            q_cos1 = tl.load(q_cos1_ptrs)
            q_sin1 = tl.load(q_sin1_ptrs)
            q_cos2 = tl.load(q_cos2_ptrs)
            q_sin2 = tl.load(q_sin2_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            rhq = tl.load(rhq_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_cos1 = tl.load(q_cos1_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_sin1 = tl.load(q_sin1_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_cos2 = tl.load(q_cos2_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_sin2 = tl.load(q_sin2_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            rhq = tl.load(rhq_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_cos1 = tl.load(q_cos1_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_sin1 = tl.load(q_sin1_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_cos2 = tl.load(q_cos2_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_sin2 = tl.load(q_sin2_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            rhq = tl.load(rhq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q_cos1 = tl.load(q_cos1_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q_sin1 = tl.load(q_sin1_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q_cos2 = tl.load(q_cos2_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q_sin2 = tl.load(q_sin2_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
    
    ## apply rope to q1, q2
    q1 = q * q_cos1 + rhq * q_sin1
    q2 = q * q_cos2 + rhq * q_sin2
            
    ## loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # jth block of K,V
        
        # -- load k and its rotate_half with cos/sin rope ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn, mask=offs_d[None, :] < headdim, other=0.0)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n, mask=offs_d[None, :] < headdim, other=0.0)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        
        # -- apply rope to k1, k2 ----
        k1 = k * k_cos1 + rhk * k_sin1
        k2 = k
        
        # -- compute qk1, qk2 ----
        qk1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) # shape: [block_sz_m, block_sz_n]
        qk2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) # shape: [block_sz_m, block_sz_n]
        # qk += tl.dot(q, k, trans_b=True) # Q * K^T => get wrong using triton-nightly 2.1.0
        qk1 += tl.dot(q1, tl.trans(k1)) # Q1 * K1^T
        qk2 += tl.dot(q2, tl.trans(k2)) # Q2 * K2^T
        
        # -- apply rectified mask to get qk --
        reM = tl.abs(offs_m[:, None] - (start_n + offs_n)[None, :]) < WINDOW
        qk = tl.where(reM, qk1, qk2)
        
        # -- apply causal mask --
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        
        # -- compute p and update mij, lij with bias applied if exists --
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0).to(tl.float32)
            # Slightly faster to multiply the softmax_scale in the tl.exp below since the compiler
            # can then fuse the mult and add into an fma instruction. But if we have bias we need to
            # to multiply with softmax_scale here.
            qk = qk * softmax_scale + bias
            m_ij = tl.maximum(tl.max(qk, 1), lse_i)
            p = tl.exp(qk - m_ij[:, None])
        else:
            m_ij = tl.maximum(tl.max(qk, 1) * softmax_scale, lse_i)
            p = tl.exp(qk * softmax_scale - m_ij[:, None])
        l_ij = tl.sum(p, 1) # temporary lij as rowsum(Pij)

        # -- scale acc_o --
        acc_o_scale = tl.exp(m_i - m_ij)
        tl.store(t_ptrs, acc_o_scale) # BUG: have to store
        acc_o_scale = tl.load(t_ptrs) # BUG: and immediately load
        acc_o = acc_o * acc_o_scale[:, None]
        
        # -- load v --
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                v = tl.load(v_ptrs + start_n * stride_vn)
            else:
                v = tl.load(v_ptrs + start_n * stride_vn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=(start_n + offs_n)[:, None] < seqlen_k,
                    other=0.0,
                )
            else:
                v = tl.load(
                    v_ptrs + start_n * stride_vn,
                    mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        
        # -- update acc_o by PV --
        p = p.to(v.dtype)
        acc_o += tl.dot(p, v)

        # -- update mi and li --
        m_i = m_ij
        l_i_new = tl.exp(lse_i - m_ij) + l_ij
        lse_i = m_ij + tl.log(l_i_new)

    ## scale the Oi
    o_scale = tl.exp(m_i - lse_i)
    # BUG: have to store and immediately load
    tl.store(t_ptrs, o_scale)
    o_scale = tl.load(t_ptrs)
    acc_o = acc_o * o_scale[:, None]
    
    ## store Li
    # re-materialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    lse_ptrs = Lse + off_hb * seqlen_q_rounded + offs_m
    tl.store(lse_ptrs, lse_i)
    
    ## initialize pointers to output
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    out_ptrs = (
        Out + off_b * stride_ob + off_h * stride_oh +
        (offs_m[:, None] * stride_om + offs_d[None, :])
    )
    
    ## store Oi
    if EVEN_M:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o)
        else:
            tl.store(out_ptrs, acc_o, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(out_ptrs, acc_o, mask=offs_m[:, None] < seqlen_q)
        else:
            tl.store(
                out_ptrs, acc_o, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim)
            )

def _flash_attn_forward_with_fused_rerope_inner(q, k, v, 
                                        cos, sin, position_ids, window_size,
                                        bias=None, causal=False, softmax_scale=None):
    ## check constraints in shape, dtype, device and index boundary
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    max_position_len, _ = cos.shape
    
    assert k.shape == (batch, seqlen_k, nheads, d)
    assert v.shape == (batch, seqlen_k, nheads, d)
    assert position_ids.shape == (batch, seqlen_q)
    assert d <= 128, "FlashAttention only support head dimensions up to 128"
    assert q.dtype == k.dtype == v.dtype == cos.dtype == sin.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16], "Only support fp16 and bf16"
    assert q.is_cuda and k.is_cuda and v.is_cuda and cos.is_cuda and sin.is_cuda and position_ids.is_cuda
    assert position_ids.max() < max_position_len
    assert window_size <= seqlen_k
    assert window_size <= max_position_len
    
    ## prepare rerope
    # 1. first set is the rope with position ids: [0,1,...,seq_len], with shape: [bs, q_len, nheads, dim]
    # 2. second set is the rope with constant position ids: [w,w,...,w], with shape: [bs, q_len, nheads, dim]
    # 3. the rectified mask to select which set to apply according to the relative distance, with shape: [bs, nh, q_len, kv_len]
    cos1 = cos[position_ids].unsqueeze(2).expand((batch, seqlen_q, nheads, d))  
    sin1 = sin[position_ids].unsqueeze(2).expand((batch, seqlen_q, nheads, d))
    cos2 = cos[position_ids * 0 + window_size].unsqueeze(2).expand((batch, seqlen_q, nheads, d))
    sin2 = sin[position_ids * 0 + window_size].unsqueeze(2).expand((batch, seqlen_q, nheads, d)) 
    # reM = ((position_ids[:, -seqlen_q:, None] - position_ids[:, None]).abs() < window_size).unsqueeze(1).expand(batch, nheads, seqlen_q, seqlen_k)
    
    ## prepare rotate_half q,k
    def rotate_half(x):
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)
    rhq, rhk = rotate_half(q), rotate_half(k)
    cos1, sin1, cos2, sin2, rhq, rhk = [x if x.stride(-1) == 1 else x.contiguous() for x in [cos1, sin1, cos2, sin2, rhq, rhk]]
    
    ## prepare scaling factor
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)

    ## prepare bias
    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        if bias.stride(-1) != 1:
            bias = bias.contiguous()
        if bias.shape[2:] == (1, seqlen_k):
            bias_type = "vector"
        elif bias.shape[2:] == (seqlen_q, seqlen_k):
            bias_type = "matrix"
        else:
            raise RuntimeError(
                "Last 2 dimensions of bias must be (1, seqlen_k)" " or (seqlen_q, seqlen_k)"
            )
        bias = bias.expand(batch, nheads, seqlen_q, seqlen_k)
    bias_strides = (bias.stride(0), bias.stride(1), bias.stride(2)) if has_bias else (0, 0, 0)

    ## prepare O, L, tmp
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128 # the closest q_len which is times of 128
    lse = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    tmp = torch.empty((batch, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    ## prepare block / warp / grid size
    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16) # usually d itself
    ## set block size
    BLOCK = 128
    # BLOCK = 256 # FIXME: out of resource
    
    ## set num of warps
    # num_warps = 4 if d <= 64 else 8
    num_warps = 4 # FIXME: when d = 128, not all close if num_warps = 8 as the line above
    
    ## set grid split
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    
    ## apply forward kernel with fused rope func for all blocks in grid
    _fwd_kernel_with_fused_rerope_inner[grid](
        q, k, v, rhq, rhk, 
        cos1, sin1, cos2, sin2,
        bias, # shape: [bs, nh, q_len, kv_len]
        o, # shape: [bs, q_len, nh, hd]
        lse, tmp, # shape: [bs, nh, q_len_round]
        softmax_scale,
        q.stride(0), q.stride(2), q.stride(1), # bs, nh, sq
        k.stride(0), k.stride(2), k.stride(1), # bs, nh, sk
        v.stride(0), v.stride(2), v.stride(1), # bs, nh, sv
        rhq.stride(0), rhq.stride(2), rhq.stride(1), # bs, nh, sq
        rhk.stride(0), rhk.stride(2), rhk.stride(1), # bs, nh, sk
        # for q1
        cos1.stride(0), cos1.stride(2), cos1.stride(1), # bs, nh, sq
        sin1.stride(0), sin1.stride(2), sin1.stride(1), # bs, nh, sq
        # for k1
        cos1.stride(0), cos1.stride(2), cos1.stride(1), # bs, nh, sk
        sin1.stride(0), sin1.stride(2), sin1.stride(1), # bs, nh, sk
        # for q2
        cos2.stride(0), cos2.stride(2), cos2.stride(1), # bs, nh, sq
        sin2.stride(0), sin2.stride(2), sin2.stride(1), # bs, nh, sq
        # bias
        *bias_strides,
        o.stride(0), o.stride(2), o.stride(1), # bs, nh, sq
        nheads, seqlen_q, seqlen_k, seqlen_q_rounded, d,
        seqlen_q // 32, seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type, causal, window_size,
        BLOCK_HEADDIM, BLOCK_M=BLOCK, BLOCK_N=BLOCK,
        num_warps=num_warps, num_stages=1,
    )
    return o, lse, softmax_scale  # softmax_scale could have been updated


### interface func
def flash_attn_func_with_fused_rerope(
    q, k, v, 
    cos, sin, 
    position_ids,
    window_size,
    bias=None, 
    causal=False, 
    softmax_scale=None, 
    inner=True):
    """
    q: (batch_size, seqlen_q, nheads, headdim)
    k, v: (batch_size, seqlen_k, nheads, headdim)
    cos, sin: (max_seq_len, headdim)
    position_ids: (batch_size, seqlen_q)
    window_size: the inner window size as rerope boundary
    bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
    For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
    ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
    """
    # Make sure that the last dimension is contiguous
    q, k, v, cos, sin = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v, cos, sin]]

    # forward the flash attn auto func
    forward_func = _flash_attn_forward_with_fused_rerope_inner if inner else _flash_attn_forward_with_fused_rerope_outter
    o, _, _ = forward_func(
        q, k, v, 
        cos, sin, position_ids, window_size,
        bias=bias, causal=causal, softmax_scale=softmax_scale
    )
    
    return o