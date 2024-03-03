import os
import argparse

import math

import torch
from transformers.models.llama import modeling_llama

import flash_attn.flash_attn_interface as fi

import triton
import triton.language as tl

# Disabling autotune for now, set num_warps=4 if headdim=64 and num_warps=8 if headdim=128
# @triton.autotune(
#     configs=[
#         triton.Config({"BLOCK_M": 128, "BLOCK_N": 128}, num_warps=4, num_stages=1),
#         # This config has a race condition when EVEN_M == False, disabling it for now.
#         # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64}, num_warps=4, num_stages=1),
#     ],
#     key=['CACHE_KEY_SEQLEN_Q', 'CACHE_KEY_SEQLEN_K', 'BIAS_TYPE', 'IS_CAUSAL', 'BLOCK_HEADDIM']
# )

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


### FIXME: fixed the rectified mask problem
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _debug_fwd_kernel_with_fused_rerope(
    Q, K, V, rhQ, rhK,
    Cos1, Sin1, Cos2, Sin2,
    IdQ, IdK,
    ReM, Bias, Out,
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
    stride_idqb, stride_idqh, stride_idqm, # q_len
    stride_idkb, stride_idkh, stride_idkn, # kv_len
    stride_remb, stride_remh, stride_remm, # q_len
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
    
    ## Initialize pointers to Q, K, V, Cos(q/k), Sin(q/k)
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
    q_cos_ptrs = (
        Cos1 + off_b * stride_qc1b + off_h * stride_qc1h +
        (offs_m[:, None] * stride_qc1m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    q_sin_ptrs = (
        Sin1 + off_b * stride_qs1b + off_h * stride_qs1h +
        (offs_m[:, None] * stride_qs1m + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    k_cos_ptrs = (
        Cos1 + off_b * stride_kc1b + off_h * stride_kc1h +
        (offs_n[:, None] * stride_kc1n + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    k_sin_ptrs = (
        Sin1 + off_b * stride_ks1b + off_h * stride_ks1h +
        (offs_n[:, None] * stride_ks1n + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    
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
    
    idq_ptrs = (
        IdQ + off_b * stride_idqb + off_h * stride_idqh +
        (offs_m[:, None] * stride_idqm + offs_d[None, :]) # shape of [block_sz_m, hd]
    )
    idk_ptrs = (
        IdK + off_b * stride_idkb + off_h * stride_idkh +
        (offs_n[:, None] * stride_idkn + offs_d[None, :]) # shape of [block_sz_n, hd]
    )
    
    reM_ptrs = (
        ReM + off_b * stride_remb + off_h * stride_remh +
        (offs_m[:, None] * stride_remm + offs_n[None, :]) # shape of [block_sz_m, block_sz_n]
    )
    
    ## initialize bias pointers
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + off_b * stride_bb + off_h * stride_bh + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = (
            Bias
            + off_b * stride_bb
            + off_h * stride_bh
            + (offs_m[:, None] * stride_bm + offs_n[None, :])
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
            q_cos = tl.load(q_cos_ptrs)
            q_sin = tl.load(q_sin_ptrs)
            q_cos1 = tl.load(q_cos1_ptrs)
            q_sin1 = tl.load(q_sin1_ptrs)
            q_cos2 = tl.load(q_cos2_ptrs)
            q_sin2 = tl.load(q_sin2_ptrs)
            idq = tl.load(idq_ptrs)
        else:
            q = tl.load(q_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            rhq = tl.load(rhq_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_cos = tl.load(q_cos_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_sin = tl.load(q_sin_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_cos1 = tl.load(q_cos1_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_sin1 = tl.load(q_sin1_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_cos2 = tl.load(q_cos2_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_sin2 = tl.load(q_sin2_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            idq = tl.load(idq_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            q = tl.load(q_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            rhq = tl.load(rhq_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            q_cos = tl.load(q_cos_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_sin = tl.load(q_sin_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_cos1 = tl.load(q_cos1_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_sin1 = tl.load(q_sin1_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_cos2 = tl.load(q_cos2_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            q_sin2 = tl.load(q_sin2_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
            idq = tl.load(idq_ptrs, mask=offs_m[:, None] < seqlen_q, other=0.0)
        else:
            q = tl.load(
                q_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
            rhq = tl.load(
                rhq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
            q_cos = tl.load(
                q_cos_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
            q_sin = tl.load(
                q_sin_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0
            )
            q_cos1 = tl.load(q_cos1_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q_sin1 = tl.load(q_sin1_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q_cos2 = tl.load(q_cos2_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            q_sin2 = tl.load(q_sin2_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0)
            idq = tl.load(idq_ptrs, mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim), other=0.0) 
            
    ## apply rope to q
    q = q * q_cos + rhq * q_sin 
    q1 = q * q_cos1 + rhq * q_sin1
    q2 = q * q_cos2 + rhq * q_sin2
    # idq = idq[:,0][:,None]
            
    ## loop over k, v and update accumulator
    end_n = seqlen_k if not IS_CAUSAL else tl.minimum((start_m + 1) * BLOCK_M, seqlen_k)
    for start_n in range(0, end_n, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N) # jth block of K,V
        
        # -- load k and its rotate_half with cos/sin rope ----
        if EVEN_N & EVEN_M:  # If we just do "if EVEN_N", there seems to be some race condition
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn)
                k_cos = tl.load(k_cos_ptrs + start_n * stride_kc1n)
                k_sin = tl.load(k_sin_ptrs + start_n * stride_ks1n)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n)
                idk = tl.load(idk_ptrs + start_n * stride_idkn)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=offs_d[None, :] < headdim, other=0.0)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn, mask=offs_d[None, :] < headdim, other=0.0)
                k_cos = tl.load(k_cos_ptrs + start_n * stride_kc1n, mask=offs_d[None, :] < headdim, other=0.0)
                k_sin = tl.load(k_sin_ptrs + start_n * stride_ks1n, mask=offs_d[None, :] < headdim, other=0.0)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n, mask=offs_d[None, :] < headdim, other=0.0)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n, mask=offs_d[None, :] < headdim, other=0.0)
                idk = tl.load(idk_ptrs + start_n * stride_idkn, mask=offs_d[None, :] < headdim, other=0.0)
        else:
            if EVEN_HEADDIM:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                k_cos = tl.load(k_cos_ptrs + start_n * stride_kc1n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                k_sin = tl.load(k_sin_ptrs + start_n * stride_ks1n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
                idk = tl.load(idk_ptrs + start_n * stride_idkn, mask=(start_n + offs_n)[:, None] < seqlen_k, other=0.0)
            else:
                k = tl.load(k_ptrs + start_n * stride_kn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                rhk = tl.load(rhk_ptrs + start_n * stride_rhkn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                k_cos = tl.load(k_cos_ptrs + start_n * stride_kc1n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                k_sin = tl.load(k_sin_ptrs + start_n * stride_ks1n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                k_cos1 = tl.load(k_cos1_ptrs + start_n * stride_kc1n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                k_sin1 = tl.load(k_sin1_ptrs + start_n * stride_ks1n, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
                idk = tl.load(idk_ptrs + start_n * stride_idkn, mask=((start_n + offs_n)[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0)
        
        # -- apply rope to k ----
        k1 = k * k_cos1 + rhk * k_sin1
        k2 = k
        k = k * k_cos + rhk * k_sin 
        
        # -- compute qk ----
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) # shape: [block_sz_m, block_sz_n]
        # qk += tl.dot(q, k, trans_b=True) # Q * K^T => get wrong using triton-nightly 2.1.0
        qk += tl.dot(q, tl.trans(k)) # Q * K^T
        
        # -- compute qk1, qk2 ----
        qk1 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) # shape: [block_sz_m, block_sz_n]
        qk2 = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32) # shape: [block_sz_m, block_sz_n]
        # qk += tl.dot(q, k, trans_b=True) # Q * K^T => get wrong using triton-nightly 2.1.0
        qk1 += tl.dot(q1, tl.trans(k1)) # Q1 * K1^T
        qk2 += tl.dot(q2, tl.trans(k2)) # Q2 * K2^T
        
        # -- apply rectified mask to get qk --
        # if EVEN_M & EVEN_N:
        #     reM = tl.load(reM_ptrs + start_n)
        # else:
        #     reM = tl.load(reM_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0)
        # qk = tl.where(reM, qk1, qk2)
        
        # reM = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        # reM = tl.where(
        #     (offs_m[:, None] < (start_n + offs_n)[None, :] + WINDOW) | \
        #     (offs_m[:, None] > (start_n + offs_n)[None, :] - WINDOW), 
        #     1, 0)
        # qk = qk1 * reM + qk2 * (1. - reM)
        
        reM = (offs_m[:, None] < (start_n + offs_n)[None, :] + WINDOW) | (offs_m[:, None] > (start_n + offs_n)[None, :] - WINDOW)
        qk = tl.where(reM, qk1, qk2)
        
        # -- apply mask --
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk += tl.where((start_n + offs_n)[None, :] < seqlen_k, 0, float("-inf"))
        if IS_CAUSAL:
            qk += tl.where(offs_m[:, None] >= (start_n + offs_n)[None, :], 0, float("-inf"))
        
        # -- compute p and update mij, lij --
        if BIAS_TYPE != "none":
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n, mask=(start_n + offs_n) < seqlen_k, other=0.0
                    ).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs + start_n).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs + start_n,
                        mask=(offs_m[:, None] < seqlen_q)
                        & ((start_n + offs_n)[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
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

### FIXME: Segmentation fault (core dumped)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _wrong_fwd_kernel_with_fused_rerope(
    Q, K, V, rhQ, rhK,
    Cos1, Sin1, Cos2, Sin2,
    ReM, Bias, Out,
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
    stride_remb, stride_remh, stride_remm, # q_len
    stride_bb, stride_bh, stride_bm, # q_len
    stride_ob, stride_oh, stride_om, # q_len
    nheads, seqlen_q, seqlen_k, seqlen_q_rounded, headdim,
    CACHE_KEY_SEQLEN_Q, CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr, IS_CAUSAL: tl.constexpr,
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
    reM_ptrs = (
        ReM + off_b * stride_remb + off_h * stride_remh +
        (offs_m[:, None] * stride_remm + offs_n[None, :]) # shape of [block_sz_m, block_sz_n]
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
        if EVEN_M & EVEN_N:
            reM = tl.load(reM_ptrs + start_n)
        else:
            reM = tl.load(reM_ptrs + start_n, mask=(offs_m[:, None] < seqlen_q) & ((start_n + offs_n)[None, :] < seqlen_k), other=0.0)
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


@triton.jit
def _bwd_preprocess_do_o_dot(
    Out,
    DO,
    Delta,
    stride_ob,
    stride_oh,
    stride_om,
    stride_dob,
    stride_doh,
    stride_dom,
    nheads,
    seqlen_q,
    seqlen_q_rounded,
    headdim,
    BLOCK_M: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
):
    start_m = tl.program_id(0)
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # load
    o = tl.load(
        Out + off_b * stride_ob + off_h * stride_oh + offs_m[:, None] * stride_om + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    do = tl.load(
        DO
        + off_b * stride_dob
        + off_h * stride_doh
        + offs_m[:, None] * stride_dom
        + offs_d[None, :],
        mask=(offs_m[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
        other=0.0,
    ).to(tl.float32)
    delta = tl.sum(o * do, axis=1)
    # write-back
    tl.store(Delta + off_hb * seqlen_q_rounded + offs_m, delta)


@triton.jit
def _bwd_store_dk_dv(
    dk_ptrs,
    dv_ptrs,
    dk,
    dv,
    offs_n,
    offs_d,
    seqlen_k,
    headdim,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
):
    # [2022-11-01] TD: Same bug. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.store(dv_ptrs), there's a race condition
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv)
            tl.store(dk_ptrs, dk)
        else:
            tl.store(dv_ptrs, dv, mask=offs_d[None, :] < headdim)
            tl.store(dk_ptrs, dk, mask=offs_d[None, :] < headdim)
    else:
        if EVEN_HEADDIM:
            tl.store(dv_ptrs, dv, mask=offs_n[:, None] < seqlen_k)
            tl.store(dk_ptrs, dk, mask=offs_n[:, None] < seqlen_k)
        else:
            tl.store(dv_ptrs, dv, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))
            tl.store(dk_ptrs, dk, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim))


@triton.jit
def _bwd_kernel_one_col_block(
    start_n,
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qm,
    stride_kn,
    stride_vn,
    stride_bm,
    stride_dom,
    stride_dqm,
    stride_dkn,
    stride_dvn,
    seqlen_q,
    seqlen_k,
    headdim,
    ATOMIC_ADD: tl.constexpr,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # We need to make sure begin_m is a multiple of BLOCK_M (not BLOCK_N)
    begin_m = 0 if not IS_CAUSAL else ((start_n * BLOCK_N) // BLOCK_M) * BLOCK_M
    # initialize row/col offsets
    offs_qm = begin_m + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_HEADDIM)
    # initialize pointers to value-like data
    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :])
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_d[None, :])
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_d[None, :])
    do_ptrs = DO + (offs_qm[:, None] * stride_dom + offs_d[None, :])
    dq_ptrs = DQ + (offs_qm[:, None] * stride_dqm + offs_d[None, :])
    if BIAS_TYPE == "vector":
        b_ptrs = Bias + offs_n
    elif BIAS_TYPE == "matrix":
        b_ptrs = Bias + (offs_qm[:, None] * stride_bm + offs_n[None, :])
    # initialize dv and dk
    dv = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    dk = tl.zeros([BLOCK_N, BLOCK_HEADDIM], dtype=tl.float32)
    # There seems to be some problem with Triton pipelining that makes results wrong for
    # headdim=64, seqlen=(113, 255), bias_type='matrix'. In this case the for loop
    # may have zero step, and pipelining with the bias matrix could screw it up.
    # So we just exit early.
    if begin_m >= seqlen_q:
        dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
        dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
        _bwd_store_dk_dv(
            dk_ptrs,
            dv_ptrs,
            dk,
            dv,
            offs_n,
            offs_d,
            seqlen_k,
            headdim,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
        )
        return
    # k and v stay in SRAM throughout
    # [2022-10-30] TD: Same bug as the fwd. In the case of EVEN_N=True and EVEN_M=False,
    # if we just call tl.load(k_ptrs), we get the wrong output!
    if EVEN_N & EVEN_M:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs)
            v = tl.load(v_ptrs)
        else:
            k = tl.load(k_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
            v = tl.load(v_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
    else:
        if EVEN_HEADDIM:
            k = tl.load(k_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
            v = tl.load(v_ptrs, mask=offs_n[:, None] < seqlen_k, other=0.0)
        else:
            k = tl.load(
                k_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
            v = tl.load(
                v_ptrs, mask=(offs_n[:, None] < seqlen_k) & (offs_d[None, :] < headdim), other=0.0
            )
    # loop over rows
    num_block_m = tl.cdiv(seqlen_q, BLOCK_M)
    for start_m in range(begin_m, num_block_m * BLOCK_M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m_curr = start_m + offs_m
        # load q, k, v, do on-chip
        # Same bug as below. Otherwise gives wrong result for headdim=40, seqlen=(128, 117)
        if EVEN_M & EVEN_HEADDIM:
            q = tl.load(q_ptrs)
        else:
            if EVEN_HEADDIM:
                q = tl.load(q_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
            else:
                q = tl.load(
                    q_ptrs,
                    mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    other=0.0,
                )
        # recompute p = softmax(qk, dim=-1).T
        # qk = tl.dot(q, k, trans_b=True) => get wrong using triton-nightly 2.1.0
        qk = tl.dot(q, tl.trans(k))
        # Trying to combine the two masks seem to make the result wrong
        if not EVEN_N:  # Need to mask out otherwise the softmax is wrong
            qk = tl.where(offs_n[None, :] < seqlen_k, qk, float("-inf"))
        if IS_CAUSAL:
            qk = tl.where(offs_m_curr[:, None] >= (offs_n[None, :]), qk, float("-inf"))
        if BIAS_TYPE != "none":
            tl.debug_barrier()  # Race condition otherwise
            if BIAS_TYPE == "vector":
                if EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(b_ptrs, mask=offs_n < seqlen_k, other=0.0).to(tl.float32)
                bias = bias[None, :]
            elif BIAS_TYPE == "matrix":
                if EVEN_M & EVEN_N:
                    bias = tl.load(b_ptrs).to(tl.float32)
                else:
                    bias = tl.load(
                        b_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_n[None, :] < seqlen_k),
                        other=0.0,
                    ).to(tl.float32)
            qk = qk * softmax_scale + bias
        # There seems to be a race condition when headdim=48/96, and dq, dk, dv are wrong.
        # Also wrong for headdim=64.
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        lse_i = tl.load(LSE + offs_m_curr)
        if BIAS_TYPE == "none":
            p = tl.exp(qk * softmax_scale - lse_i[:, None])
        else:
            p = tl.exp(qk - lse_i[:, None])
        # compute dv
        # [2022-10-30] TD: A Triton bug: if EVEN_M=True and EVEN_HEADDIM=False, if we call
        # do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0), we get wrong outputs
        # in the case of headdim=48/96, seqlen_q & seqlen_k >= 512. If headdim=40 or seqlen < 512,
        # the output is correct.
        if EVEN_M & EVEN_HEADDIM:
            do = tl.load(do_ptrs)
        else:
            # [2022-11-01] TD: Triton bug, there's a race condition if we just use m_mask and not d_mask.
            do = tl.load(
                do_ptrs,
                mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                other=0.0,
            )
        # if EVEN_M:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs)
        #     else:
        #         do = tl.load(do_ptrs, mask=offs_d[None, :] < headdim, other=0.0)
        # else:
        #     if EVEN_HEADDIM:
        #         do = tl.load(do_ptrs, mask=offs_m_curr[:, None] < seqlen_q, other=0.0)
        #     else:
        #         do = tl.load(do_ptrs, mask=(offs_m_curr[:, None] < seqlen_q)
        #                                    & (offs_d[None, :] < headdim), other=0.0)
        
        # dv += tl.dot(p.to(do.dtype), do, trans_a=True) => get wrong using triton-nightly 2.1.0
        dv += tl.dot(tl.trans(p.to(do.dtype)), do)
        # compute dp = dot(v, do)
        # There seems to be a race condition when headdim=48/96, and dq, dk are wrong.
        # Also wrong for headdim=128, seqlen=(108, 256), and ATOMIC_ADD=True
        # Also wrong for headdim=64, seqlen=(1023, 1024), and ATOMIC_ADD=False
        if not (EVEN_M & EVEN_HEADDIM):
            tl.debug_barrier()
        # dp = tl.dot(do, v, trans_b=True) => get wrong using triton-nightly 2.1.0
        dp = tl.dot(do, tl.trans(v))
        # There's a race condition for headdim=48
        if not EVEN_HEADDIM:
            tl.debug_barrier()
            
        # compute ds = p * (dp - delta[:, None])
        # Putting the subtraction after the dp matmul (instead of before) is slightly faster
        Di = tl.load(D + offs_m_curr)
        
        # Converting ds to q.dtype here reduces register pressure and makes it much faster
        # for BLOCK_HEADDIM=128
        ds = (p * (dp - Di[:, None]) * softmax_scale).to(q.dtype)
        
        # compute dk = dot(ds.T, q)
        # dk += tl.dot(ds, q, trans_a=True) => get wrong using triton-nightly 2.1.0
        dk += tl.dot(tl.trans(ds), q)
        
        # compute dq
        if not (
            EVEN_M & EVEN_HEADDIM
        ):  # Otherewise there's a race condition when BIAS_TYPE='matrix'
            tl.debug_barrier()
        if not ATOMIC_ADD:
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                dq = tl.load(dq_ptrs, eviction_policy="evict_last")
                dq += tl.dot(ds, k)
                tl.store(dq_ptrs, dq, eviction_policy="evict_last")
            else:
                if EVEN_HEADDIM:
                    dq = tl.load(
                        dq_ptrs,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=offs_m_curr[:, None] < seqlen_q,
                        eviction_policy="evict_last",
                    )
                else:
                    dq = tl.load(
                        dq_ptrs,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        other=0.0,
                        eviction_policy="evict_last",
                    )
                    dq += tl.dot(ds, k)
                    tl.store(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                        eviction_policy="evict_last",
                    )
        else:  # If we're parallelizing across the seqlen_k dimension
            dq = tl.dot(ds, k)
            if EVEN_M & EVEN_HEADDIM:  # Race condition if we just do EVEN_M
                tl.atomic_add(dq_ptrs, dq)
            else:
                if EVEN_HEADDIM:
                    tl.atomic_add(dq_ptrs, dq, mask=offs_m_curr[:, None] < seqlen_q)
                else:
                    tl.atomic_add(
                        dq_ptrs,
                        dq,
                        mask=(offs_m_curr[:, None] < seqlen_q) & (offs_d[None, :] < headdim),
                    )
        # increment pointers
        dq_ptrs += BLOCK_M * stride_dqm
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        if BIAS_TYPE == "matrix":
            b_ptrs += BLOCK_M * stride_bm
    # write-back
    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_d[None, :])
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_d[None, :])
    _bwd_store_dk_dv(
        dk_ptrs,
        dv_ptrs,
        dk,
        dv,
        offs_n,
        offs_d,
        seqlen_k,
        headdim,
        EVEN_M=EVEN_M,
        EVEN_N=EVEN_N,
        EVEN_HEADDIM=EVEN_HEADDIM,
    )


def init_to_zero(name):
    return lambda nargs: nargs[name].zero_()

@triton.autotune(
    configs=[
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": False},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        triton.Config(
            {"BLOCK_M": 128, "BLOCK_N": 128, "SEQUENCE_PARALLEL": True},
            num_warps=8,
            num_stages=1,
            pre_hook=init_to_zero("DQ"),
        ),
        # Other configs seem to give wrong results when seqlen_q % 128 != 0, disabling them for now
        # # Kernel is buggy (give wrong result) if we set BLOCK_m=128, BLOCK_n=64, num_warps=*4*
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=8, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": False}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
        # triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "SEQUENCE_PARALLEL": True}, num_warps=4, num_stages=1, pre_hook=init_to_zero('DQ')),
    ],
    key=["CACHE_KEY_SEQLEN_Q", "CACHE_KEY_SEQLEN_K", "BIAS_TYPE", "IS_CAUSAL", "BLOCK_HEADDIM"],
)
@triton.heuristics(
    {
        "EVEN_M": lambda args: args["seqlen_q"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["seqlen_k"] % args["BLOCK_N"] == 0,
        "EVEN_HEADDIM": lambda args: args["headdim"] == args["BLOCK_HEADDIM"],
    }
)
@triton.jit
def _bwd_kernel(
    Q,
    K,
    V,
    Bias,
    DO,
    DQ,
    DK,
    DV,
    LSE,
    D,
    softmax_scale,
    stride_qb,
    stride_qh,
    stride_qm,
    stride_kb,
    stride_kh,
    stride_kn,
    stride_vb,
    stride_vh,
    stride_vn,
    stride_bb,
    stride_bh,
    stride_bm,
    stride_dob,
    stride_doh,
    stride_dom,
    stride_dqb,
    stride_dqh,
    stride_dqm,
    stride_dkb,
    stride_dkh,
    stride_dkn,
    stride_dvb,
    stride_dvh,
    stride_dvn,
    nheads,
    seqlen_q,
    seqlen_k,
    seqlen_q_rounded,
    headdim,
    CACHE_KEY_SEQLEN_Q,
    CACHE_KEY_SEQLEN_K,
    BIAS_TYPE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    BLOCK_HEADDIM: tl.constexpr,
    SEQUENCE_PARALLEL: tl.constexpr,
    EVEN_M: tl.constexpr,
    EVEN_N: tl.constexpr,
    EVEN_HEADDIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    off_hb = tl.program_id(1)
    off_b = off_hb // nheads
    off_h = off_hb % nheads
    # offset pointers for batch/head
    Q += off_b * stride_qb + off_h * stride_qh
    K += off_b * stride_kb + off_h * stride_kh
    V += off_b * stride_vb + off_h * stride_vh
    DO += off_b * stride_dob + off_h * stride_doh
    DQ += off_b * stride_dqb + off_h * stride_dqh
    DK += off_b * stride_dkb + off_h * stride_dkh
    DV += off_b * stride_dvb + off_h * stride_dvh
    if BIAS_TYPE != "none":
        Bias += off_b * stride_bb + off_h * stride_bh
    # pointer to row-wise quantities in value-like data
    D += off_hb * seqlen_q_rounded
    LSE += off_hb * seqlen_q_rounded
    if not SEQUENCE_PARALLEL:
        num_block_n = tl.cdiv(seqlen_k, BLOCK_N)
        for start_n in range(0, num_block_n):
            _bwd_kernel_one_col_block(
                start_n,
                Q,
                K,
                V,
                Bias,
                DO,
                DQ,
                DK,
                DV,
                LSE,
                D,
                softmax_scale,
                stride_qm,
                stride_kn,
                stride_vn,
                stride_bm,
                stride_dom,
                stride_dqm,
                stride_dkn,
                stride_dvn,
                seqlen_q,
                seqlen_k,
                headdim,
                ATOMIC_ADD=False,
                BIAS_TYPE=BIAS_TYPE,
                IS_CAUSAL=IS_CAUSAL,
                BLOCK_HEADDIM=BLOCK_HEADDIM,
                EVEN_M=EVEN_M,
                EVEN_N=EVEN_N,
                EVEN_HEADDIM=EVEN_HEADDIM,
                BLOCK_M=BLOCK_M,
                BLOCK_N=BLOCK_N,
            )
    else:
        start_n = tl.program_id(0)
        _bwd_kernel_one_col_block(
            start_n,
            Q,
            K,
            V,
            Bias,
            DO,
            DQ,
            DK,
            DV,
            LSE,
            D,
            softmax_scale,
            stride_qm,
            stride_kn,
            stride_vn,
            stride_bm,
            stride_dom,
            stride_dqm,
            stride_dkn,
            stride_dvn,
            seqlen_q,
            seqlen_k,
            headdim,
            ATOMIC_ADD=True,
            BIAS_TYPE=BIAS_TYPE,
            IS_CAUSAL=IS_CAUSAL,
            BLOCK_HEADDIM=BLOCK_HEADDIM,
            EVEN_M=EVEN_M,
            EVEN_N=EVEN_N,
            EVEN_HEADDIM=EVEN_HEADDIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
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


def _flash_attn_backward(
    do, q, k, v, o, lse, dq, dk, dv, bias=None, causal=False, softmax_scale=None
):
    # Make sure that the last dimension is contiguous
    if do.stride(-1) != 1:
        do = do.contiguous()
    batch, seqlen_q, nheads, d = q.shape
    _, seqlen_k, _, _ = k.shape
    # assert d in {16, 32, 64, 128}
    assert d <= 128
    seqlen_q_rounded = math.ceil(seqlen_q / 128) * 128
    assert lse.shape == (batch, nheads, seqlen_q_rounded)
    assert q.stride(-1) == k.stride(-1) == v.stride(-1) == o.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == dv.stride(-1) == 1
    softmax_scale = softmax_scale or 1.0 / math.sqrt(d)
    # dq_accum = torch.zeros_like(q, dtype=torch.float32)
    dq_accum = torch.empty_like(q, dtype=torch.float32)
    delta = torch.empty_like(lse)
    # delta = torch.zeros_like(lse)

    BLOCK_HEADDIM = max(triton.next_power_of_2(d), 16)
    grid = lambda META: (triton.cdiv(seqlen_q, META["BLOCK_M"]), batch * nheads)
    _bwd_preprocess_do_o_dot[grid](
        o,
        do,
        delta,
        o.stride(0),
        o.stride(2),
        o.stride(1),
        do.stride(0),
        do.stride(2),
        do.stride(1),
        nheads,
        seqlen_q,
        seqlen_q_rounded,
        d,
        BLOCK_M=128,
        BLOCK_HEADDIM=BLOCK_HEADDIM,
    )

    has_bias = bias is not None
    bias_type = "none"
    if has_bias:
        assert bias.dtype in [q.dtype, torch.float]
        assert bias.is_cuda
        assert bias.dim() == 4
        assert bias.stride(-1) == 1
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

    # BLOCK_M = 128
    # BLOCK_N = 64
    # num_warps = 4
    grid = lambda META: (
        triton.cdiv(seqlen_k, META["BLOCK_N"]) if META["SEQUENCE_PARALLEL"] else 1,
        batch * nheads,
    )
    _bwd_kernel[grid](
        q,
        k,
        v,
        bias,
        do,
        dq_accum,
        dk,
        dv,
        lse,
        delta,
        softmax_scale,
        q.stride(0),
        q.stride(2),
        q.stride(1),
        k.stride(0),
        k.stride(2),
        k.stride(1),
        v.stride(0),
        v.stride(2),
        v.stride(1),
        *bias_strides,
        do.stride(0),
        do.stride(2),
        do.stride(1),
        dq_accum.stride(0),
        dq_accum.stride(2),
        dq_accum.stride(1),
        dk.stride(0),
        dk.stride(2),
        dk.stride(1),
        dv.stride(0),
        dv.stride(2),
        dv.stride(1),
        nheads,
        seqlen_q,
        seqlen_k,
        seqlen_q_rounded,
        d,
        seqlen_q // 32,
        seqlen_k // 32,  # key for triton cache (limit number of compilations)
        # Can't use kwargs here because triton autotune expects key to be args, not kwargs
        # IS_CAUSAL=causal, BLOCK_HEADDIM=d,
        bias_type,
        causal,
        BLOCK_HEADDIM,
        # SEQUENCE_PARALLEL=False,
        # BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        # num_warps=num_warps,
        # num_stages=1,
    )
    dq.copy_(dq_accum)


class FlashAttnFuncWithFusedReRoPE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, 
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
        o, lse, ctx.softmax_scale = forward_func(
                q, k, v, 
                cos, sin, position_ids, window_size,
                bias=bias, causal=causal, softmax_scale=softmax_scale
            )
        # save context for backward
        ctx.save_for_backward(q, k, v, o, lse, bias)
        ctx.causal = causal
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, lse, bias = ctx.saved_tensors
        assert not ctx.needs_input_grad[3], "FlashAttention does not support bias gradient yet"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)
            _flash_attn_backward(
                do,
                q,
                k,
                v,
                o,
                lse,
                dq,
                dk,
                dv,
                bias=bias,
                causal=ctx.causal,
                softmax_scale=ctx.softmax_scale,
            )
        return dq, dk, dv, None, None, None


flash_attn_func_with_fused_rerope = FlashAttnFuncWithFusedReRoPE.apply

def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    
    # NOTE patch here:
    # the position_ids no longer the compressed one with only the last token id
    # but always the whole one containing from 0 to kv_seq_len
    # so q only needs to apply the last q_len tokens
    q_embed = (q * cos[:, :, -q.shape[2]:]) + (modeling_llama.rotate_half(q) * sin[:, :, -q.shape[2]:]) if q is not None else None
    # k is also always the whole one from position 0 to position kv_seq_len
    k_embed = (k * cos) + (modeling_llama.rotate_half(k) * sin) if k is not None else None
    return q_embed, k_embed

def check_diff(out1, out2):
    out1 = out1[0,0,:,:10]
    out2 = out2[0,0,:,:10]
    cnt = []
    for i in range(out1.shape[-2]):
        if torch.allclose(out1[i,:], out2[i,:], atol=1e-2, rtol=2):
            continue
        # print(f"The {i}th token's output is different: out1:{out1[i,:]} | out2:{out2[i,:]}\n")
        cnt.append(i)
    print()
    print(f"The unmatched indexs (total {len(cnt)}) are: {cnt}")
    return len(cnt)
    

def test_all_close_rerope(Z, H, N_CTX, D_HEAD, WINDOW, causal, inner, dtype=torch.bfloat16, device='cuda'):
    ## generate qkv data
    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, D_HEAD), dtype=dtype, device=device)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 1.0 / math.sqrt(q.size(-1)) # default scaling factor: 1 / sqrt(d)
    
    ## generat rotary embedding
    position_ids = torch.arange(N_CTX).unsqueeze(0).repeat((Z, 1)).to(device) # shape: (batch_size, q_len)
    rope = modeling_llama.LlamaRotaryEmbedding(
            dim=D_HEAD, 
            max_position_embeddings=N_CTX, 
            base=10000, 
            device=device
    )
    cos, sin = rope(v, seq_len=max(N_CTX, WINDOW + 1)) 
    
    ## generate mask
    M = torch.tril(torch.ones((N_CTX, N_CTX), device=device)) # causal mask
    reM = ((position_ids[:, -N_CTX:, None] - position_ids[:, None]).abs() < WINDOW).unsqueeze(1) # rectified mask
    
    ## 1. pytorch naive attention impl with separated rerope
    q1, k1 = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    q2, _ = apply_rotary_pos_emb(q, None, cos, sin, position_ids * 0 + WINDOW)
    k2 = k

    p1 = torch.matmul(q1, k1.transpose(2,3)) * sm_scale
    p2 = torch.matmul(q2, k2.transpose(2,3)) * sm_scale
    p = torch.where(reM, p1, p2)

    if causal: p[:, :, M == 0] = float("-inf")
    p = torch.softmax(p, dim=-1)
    torch_out = torch.matmul(p, v)
    
    ## 2. triton implementation with fused rerope 
    tri_out = flash_attn_func_with_fused_rerope(
            q.transpose(1,2), k.transpose(1,2), v.transpose(1,2), 
            cos.squeeze(1).squeeze(0), sin.squeeze(1).squeeze(0), # [q_len, dim]
            position_ids, WINDOW,
            None, causal, sm_scale, inner
        ).transpose(1,2)
    
    # compare
    
    return torch.allclose(torch_out, tri_out, atol=1e-2, rtol=2)


def torch_naive_attn_func_with_rerope(q, k, v, cos, sin, position_ids, window, 
                                    attn_mask, causal, sm_scale):
    reM = ((position_ids[:, -q.shape[-2]:, None] - position_ids[:, None]).abs() < window).unsqueeze(1) # rectified mask
    
    ## 1. pytorch naive attention impl with separated rerope
    q1, k1 = apply_rotary_pos_emb(q, k, cos, sin, position_ids)
    q2, _ = apply_rotary_pos_emb(q, None, cos, sin, position_ids * 0 + window)
    k2 = k

    p1 = torch.matmul(q1, k1.transpose(2,3)) * sm_scale
    p2 = torch.matmul(q2, k2.transpose(2,3)) * sm_scale
    p = torch.where(reM, p1, p2)

    if causal: p[:, :, attn_mask == 0] = float("-inf")
    p = torch.softmax(p.float(), dim=-1).half()
    o = torch.matmul(p, v)
    
    return o

BATCH, N_HEADS, D_HEAD, WINDOW = 1, 4, 128, 256
# vary seq length for fixed head and batch
rerope_configs = [
    triton.testing.Benchmark(
        x_names=["N_CTX"],
        x_vals=[2**i for i in range(12, 15)],
        line_arg="provider",
        line_vals=["triton-rerope-inner", "triton-rerope-outter", "torch-rerope"],
        line_names=["triton-rerope-inner", "triton-rerope-outter", "torch-rerope"],
        styles=[("red", "-"), ("orange", "-"), ("purple", "-")],
        ylabel="flops",
        plot_name=f"fused-rerope-batch{BATCH}-head{N_HEADS}-d{D_HEAD}-w{WINDOW}-{mode}",
        args={
            "H": N_HEADS,
            "BATCH": BATCH,
            "D_HEAD": D_HEAD,
            "WINDOW": WINDOW,
            "dtype": torch.float16,
            "mode": mode,
            "causal": causal,
        },
    )
    for mode in ["fwd"]
    for causal in [True]
]

@triton.testing.perf_report(rerope_configs)
def bench_flash_attention_with_rerope(
    BATCH, H, N_CTX, D_HEAD, WINDOW, causal, 
    mode, provider, dtype=torch.float16, device="cuda"
):
    assert mode in ["fwd", "bwd"]
    warmup = 25
    rep = 100
    
    if provider == "triton-rerope-inner":
        ## generat rotary embedding
        position_ids = torch.arange(N_CTX).unsqueeze(0).repeat((BATCH, 1)).to(device) # shape: (batch_size, q_len)
        rope = modeling_llama.LlamaRotaryEmbedding(
                dim=D_HEAD, 
                max_position_embeddings=N_CTX, 
                base=10000, 
                device=device
        )
        
        q = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        # rope
        cos, sin = rope(v.transpose(1,2), seq_len=max(N_CTX, WINDOW + 1)) 
        cos, sin = cos.squeeze(1).squeeze(0), sin.squeeze(1).squeeze(0)
        
        fn = lambda: flash_attn_func_with_fused_rerope(q, k, v, 
                                                    cos, sin, position_ids, WINDOW,
                                                    None, causal, sm_scale, True)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "triton-rerope-outter":
        ## generat rotary embedding
        position_ids = torch.arange(N_CTX).unsqueeze(0).repeat((BATCH, 1)).to(device) # shape: (batch_size, q_len)
        rope = modeling_llama.LlamaRotaryEmbedding(
                dim=D_HEAD, 
                max_position_embeddings=N_CTX, 
                base=10000, 
                device=device
        )
        
        q = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, N_CTX, H, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        # rope
        cos, sin = rope(v.transpose(1,2), seq_len=max(N_CTX, WINDOW + 1)) 
        cos, sin = cos.squeeze(1).squeeze(0), sin.squeeze(1).squeeze(0)
        
        fn = lambda: flash_attn_func_with_fused_rerope(q, k, v, 
                                                    cos, sin, position_ids, WINDOW,
                                                    None, causal, sm_scale, False)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    if provider == "torch-rerope":
        ## generat rotary embedding
        position_ids = torch.arange(N_CTX).unsqueeze(0).repeat((BATCH, 1)).to(device) # shape: (batch_size, q_len)
        rope = modeling_llama.LlamaRotaryEmbedding(
                dim=D_HEAD, 
                max_position_embeddings=N_CTX, 
                base=10000, 
                device=device
        )
        
        q = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        k = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        v = torch.randn((BATCH, H, N_CTX, D_HEAD), dtype=dtype, device=device, requires_grad=True)
        sm_scale = 1.3
        M = torch.tril(torch.ones((N_CTX, N_CTX), device=device)) # causal mask
        
        # rerope
        cos, sin = rope(v, seq_len=max(N_CTX, WINDOW + 1)) 
        fn = lambda: torch_naive_attn_func_with_rerope(q, k, v, cos, sin, position_ids, 
                                                    WINDOW, M, causal, sm_scale)
        if mode == "bwd":
            o = fn()
            do = torch.randn_like(o)
            fn = lambda: o.backward(do, retain_graph=True)
        ms = triton.testing.do_bench(fn, warmup=warmup, rep=rep)
    
    flops_per_matmul = 2.0 * BATCH * H * N_CTX * N_CTX * D_HEAD
    total_flops = 2 * flops_per_matmul
    if causal:
        total_flops *= 0.5
    if mode == "bwd":
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)
    return total_flops / ms * 1e-9


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test flash rerope by precision and flops")
    parser.add_argument("--test", '-t', type=str, default="precision", choices=["precision", "flops"], help="test mode, either precision or flops for now")
    parser.add_argument("--save_dir", '-s', type=str, required=True, help="path to save test results for flops")
    
    args = parser.parse_args()
    
    if args.test == "precision":
        for l in range(1000, 32_000, 549):
            all_close = test_all_close_rerope(1, 2, l, 128, 512, True, False)
            if not all_close:
                print(f"They are NOT all closed when length={l}")
    elif args.test == "flops":
        bench_flash_attention_with_rerope.run(save_path=args.save_dir, print_data=True)




