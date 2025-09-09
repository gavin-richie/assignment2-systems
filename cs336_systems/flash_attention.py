from typing import Any

import numpy as np
import torch
from einops import einsum, rearrange
import triton
import triton.language as tl


class FlashAttentionTorch(torch.autograd.Function):
    @staticmethod
    def forward(context, Q, K, V, is_causal=False):

        # Define block sizes for query and key (used for chunking sequences into smaller blocks)
        Bq = 16  # Block size for Query
        Bk = 16  # Block size for Key

        # Ensure inputs are at least 3D: [batch, seq_len, d]
        # If input is 2D [seq_len, d], unsqueeze to [1, seq_len, d]
        if len(Q.shape) == 2: Q = Q.unsqueeze(0)
        if len(K.shape) == 2: K = K.unsqueeze(0)
        if len(V.shape) == 2: V = V.unsqueeze(0)

        # Now the shape is expected to be: [batch, seq_len, d]
        d = Q.shape[-1]  # Feature dimension (embedding size)
        device = Q.device  # Device (cuda or cpu)

        # 🔹 Rearrange Q, K, V into blocked format to process in smaller chunks.
        # Format: [batch, seq_len, d] --> [Tq, batch, Bq, d], where Tq = seq_len // Bq
        # This splits the sequence length into blocks of size Bq (16) for queries, same for keys.
        Q = rearrange(Q, "batch (Tq Bq) d -> Tq batch Bq d", Bq=Bq)  # Shape: [Tq, batch, Bq, d]
        K = rearrange(K, "batch (Tk Bk) d -> Tk batch Bk d", Bk=Bk)  # Shape: [Tk, batch, Bk, d]
        V = rearrange(V, "batch (Tk Bk) d -> Tk batch Bk d", Bk=Bk)  # Shape: [Tk, batch, Bk, d]

        d = Q.shape[-1]  # Feature dimension (again, for clarity)
        Tq = Q.shape[0]  # Number of query blocks
        Tk = K.shape[0]  # Number of key blocks
        batch = Q.shape[1]  # Batch size

        # Initialize the output tensor O [Tq, batch, Bq, d]
        # This will hold the attention output for each query block
        O = torch.zeros(Tq, batch, Bq, d, device=device)

        # L will store logsumexp values for numerical stability (used in backward pass)
        L = torch.zeros(Tq, batch, Bq, device=device)

        # Mi keeps track of the maximum attention score in each block (for logsumexp)
        Mi = torch.full((batch, Bq), float('-inf'), device=device)

        # =============================================
        # 🔁 Outer Loop: Iterate over each Query Block
        # =============================================
        for i in range(Tq):  # Loop over each query block
            Qi = Q[i]  # Current query block: [batch, Bq, d]

            # Initialize variables to accumulate results across key blocks
            Oi_prev = torch.zeros(batch, Bq, d, device=device)  # Previous output contribution
            Li_prev = torch.zeros(batch, Bq, device=device)  # Previous logsumexp
            Mi_prev = torch.full((batch, Bq), float('-inf'), device=device)  # Previous max score

            # =============================================
            # 🔁 Inner Loop: Iterate over each Key Block
            # =============================================
            for j in range(Tk):  # Loop over each key block
                Kj = K[j]  # Current key block: [batch, Bk, d]
                Vj = V[j]  # Current value block: [batch, Bk, d]

                # Compute attention scores: Q_i * K_j^T / sqrt(d)
                # Resulting shape: [batch, Bq, Bk]
                Sij = einsum(Qi, Kj, "batch Bq d, batch Bk d -> batch Bq Bk") / np.sqrt(d)

                # Compute max score along the key dimension (for numerical stability)
                rowmax = torch.max(Sij, dim=2, keepdim=False)[0]  # [batch, Bq]
                Mij = torch.maximum(Mi_prev, rowmax)  # Update max: global within the block

                # Compute softmax numerator: exp(Sij - Mij)
                Pij = torch.exp(Sij - Mij.unsqueeze(-1))  # [batch, Bq, Bk]

                # Sum of softmax numerators -> denominator for softmax
                rowsum_Pij = torch.sum(Pij, dim=2, keepdim=False)  # [batch, Bq]

                # Compute Lij: logsumexp trick component for stable softmax
                # Formula: (exp(Mi_prev - Mij) * Li_prev) + rowsum_Pij
                Lij = (torch.exp(Mi_prev - Mij) * Li_prev) + rowsum_Pij

                # ================================
                # Compute current output contribution: Oij
                # ================================
                # diag_scale: only keep diagonal elements (could be related to causal masking)
                # Here it's a per-query-block diagonal matrix based on exp(Mi_prev - Mij)
                diag_scale = torch.diag_embed(torch.exp(Mi_prev - Mij))  # [batch, Bq, Bq]

                # Oij = diag_scale * Oi_prev  +  Pij * Vj
                # Meaning: combine previous output with current key/value contribution
                Oij = einsum(diag_scale, Oi_prev, "batch Bq Bq, batch Bq d -> batch Bq d") + \
                      einsum(Pij, Vj, "batch Bq Bk, batch Bk d -> batch Bq d")

                # Update running values for next key block
                Oi_prev = Oij
                Li_prev = Lij
                Mi_prev = Mij

            # =========================================
            # After processing all key blocks for this query block:
            # Compute final output O[i] and log value L[i]
            # =========================================

            # inv_Li_diag: inverse of the summed weights (for normalization)
            inv_Li_diag = torch.diag_embed(1.0 / Li_prev)  # [batch, Bq, Bq]

            # Final output for this query block
            O[i] = einsum(inv_Li_diag, Oi_prev, "batch Bq Bq, batch Bq d -> batch Bq d")

            # Save logsumexp-like value for backward pass
            L[i] = Mi_prev + torch.log(Li_prev)

        # =========================================
        # After all blocks: restore original tensor shapes
        # =========================================

        # Merge blocks back into full sequence: [Tq, batch, Bq, d] -> [batch, Tq*Bq, d]
        O = rearrange(O, 'Tq batch Bq d -> batch (Tq Bq) d')
        L = rearrange(L, 'Tq batch Bq -> batch (Tq Bq)')
        Q = rearrange(Q, 'Tq batch Bq d -> batch (Tq Bq) d')
        K = rearrange(K, 'Tk batch Bk d -> batch (Tk Bk) d')
        V = rearrange(V, 'Tk batch Bk d -> batch (Tk Bk) d')

        # Save tensors needed for backward pass
        context.save_for_backward(Q, K, V, O, L)
        context.is_causal = is_causal  # Save flag (not used currently)

        # Return the attention output
        return O

    @staticmethod
    def backward(context, *dO):
        Q, K, V, O, L = context.saved_tensors
        is_causal = context.is_causal
        # return compiled_backward(Q, K, V, O, L, dO, is_causal)
        return backward_pass_recomp(Q, K, V, O, L, dO, is_causal)
        # raise NotImplementedError(
        #     "You need to implement the backward pass (or import a function like compiled_backward) "
        #     "that computes grad_Q, grad_K, grad_V from Q, K, V, O, L, grad_output, is_causal."
        # )


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(context, Q, K, V, is_causal=False):
        Bq = 16
        Bk = 16

        device = torch.device("cuda")
        Q = Q.to(device)
        K = K.to(device)
        V = V.to(device)

        if len(Q.shape) == 2: Q = Q.unsqueeze(0)
        if len(K.shape) == 2: K = K.unsqueeze(0)
        if len(V.shape) == 2: V = V.unsqueeze(0)

        d = Q.shape[-1]  # [batch,seq_len,d]
        batch = Q.shape[0]
        N_Queries = Q.shape[-2]
        N_KEYS = K.shape[-2]
        T_q = N_Queries // Bq

        O = torch.empty((batch, N_Queries, d), dtype=torch.float32).to(device)
        L = torch.empty((batch, N_Queries), dtype=torch.float32).to(device)

        launch_grid = (T_q, batch)
        scale = 1 / np.sqrt(d)

        flash_attention_kernel[launch_grid](
            Q, K, V,
            O, L,
            Q.stride(0), Q.stride(1), Q.stride(2),
            K.stride(0), K.stride(1), K.stride(2),
            V.stride(0), V.stride(1), V.stride(2),
            O.stride(0), O.stride(1), O.stride(2),
            L.stride(0), L.stride(1),
            N_QUERIES=N_Queries, N_KEYS=N_KEYS,
            scale=scale,
            D=d, Q_TILE_SIZE=Bq, K_TILE_SIZE=Bk, is_causal=is_causal,
        )
        context.save_for_backward(Q, K, V, O, L)
        context.is_causal = is_causal
        return O

    @staticmethod
    def backward(context: Any, *dO: Any) -> Any:
        Q, K, V, O, L = context.saved_tensors
        is_causal = context.is_causal
        # raise NotImplementedError(
        #     "You need to implement the backward pass (or import a function like compiled_backward) "
        #     "that computes grad_Q, grad_K, grad_V from Q, K, V, O, L, grad_output, is_causal."
        # )
        # return compiled_backward(Q, K, V, O, L, dO, is_causal)
        return backward_pass_recomp(Q, K, V, O, L, dO, is_causal)


@triton.jit
def flash_attention_kernel(
        Q_ptr, K_ptr, V_ptr,
        O_ptr, L_ptr,
        stride_qb, stride_qq, stride_qd,
        stride_kb, stride_kk, stride_kd,
        stride_vb, stride_vk, stride_vd,
        stride_ob, stride_oq, stride_od,
        stride_lb, stride_lq,
        N_QUERIES, N_KEYS,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        is_causal: tl.constexpr,
    ):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)

    Q_block_ptr = tl.make_block_ptr(Q_ptr + batch_index * stride_qb,
                                    shape=(N_QUERIES, D),
                                    strides=(stride_qq, stride_qd),
                                    offsets=(query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0),
                                    ),
    K_block_ptr = tl.make_block_ptr(K_ptr + batch_index * stride_kb,
                                    shape=(N_KEYS, D),
                                    strides=(stride_kk, stride_kd),
                                    offsets=(0, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0), )
    V_block_ptr = tl.make_block_ptr(V_ptr + batch_index * stride_vb,
                                    shape=(N_KEYS, D),
                                    strides=(stride_vk, stride_vd),
                                    offsets=(0, 0),
                                    block_shape=(K_TILE_SIZE, D),
                                    order=(1, 0), )
    O_block_ptr = tl.make_block_ptr(O_ptr + batch_index * stride_ob,
                                    shape=(N_QUERIES, D),
                                    strides=(stride_oq, stride_od),
                                    offsets=(query_tile_index * Q_TILE_SIZE, 0),
                                    block_shape=(Q_TILE_SIZE, D),
                                    order=(1, 0), )
    L_block_ptr = tl.make_block_ptr(L_ptr + batch_index * stride_lb,
                                    shape=(N_QUERIES,),
                                    strides=(stride_lq,),
                                    offsets=(query_tile_index * Q_TILE_SIZE,),
                                    block_shape=(Q_TILE_SIZE,),
                                    order=(0,), )
    Q = tl.load(Q_block_ptr, boundary_check=(0, 1), padding_option="zero")
    Tk = tl.cdiv(N_KEYS, K_TILE_SIZE)

    oi_prev = tl.zeros((Q_TILE_SIZE,D), dtype=tl.float32)
    mi_prev = tl.full((Q_TILE_SIZE,), float('-inf'), dtype=tl.float32)
    li_prev = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

    for j in range(Tk):
        Kj = tl.load(K_block_ptr, boundary_check=(0, 1), padding_option="zero")
        Vj = tl.load(V_block_ptr, boundary_check=(0, 1), padding_option="zero")

        Sij = tl.dot(Q, tl.trans(Kj)) * scale

        if is_causal:
            query_offsets = query_tile_index * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
            keys_offsets = j * K_TILE_SIZE + tl.arange(0, K_TILE_SIZE)
            mask = query_offsets[:, None] < keys_offsets[None, :]
            Sij = tl.where(mask, float('-inf'), Sij)

        sij_max = tl.max(Sij, axis=1)
        mij = tl.maximum(mi_prev, sij_max)
        pij = tl.exp(Sij - mij[:, None])

        lij = li_prev * tl.exp(mi_prev - mij) + tl.sum(pij, axis=-1)
        diag = tl.exp(mi_prev - mij)
        scaled_oi_prev = oi_prev * diag[:, None]

        tmp = tl.dot(pij.to(tl.float32), Vj.to(tl.float32))
        oij = tmp + scaled_oi_prev

        oi_prev = oij
        mi_prev = mij
        li_prev = lij

        K_block_ptr = K_block_ptr.advance((K_TILE_SIZE, 0))
        V_block_ptr = V_block_ptr.advance((K_TILE_SIZE, 0))

    inv_Li_diag = 1.0 / li_prev[:, None]
    oi = inv_Li_diag * oi_prev
    li = mi_prev + tl.log(li_prev)

    # cast oi to black pointer's element type before storing
    # same with li
    oi = oi.to(O_block_ptr.type.element_ty)
    li = li.to(L_block_ptr.type.element_ty)

    tl.store(O_block_ptr, oi, boundary_check=(0, 1))
    tl.store(L_block_ptr, li, boundary_check=(0,))



def backward_pass_recomp(Q, K, V, O, L, dO, is_causal = False):
    """
    Q (batch_size=4, n_queries=128, D=64) K,V,
    O ((batch_size, n_queries, D)
    dO[1 batch seq d],
    L[batch seq_len]
    Nq = n_queries =128, Nk= n_keys = 128, d=64

    """
    Nq, Nk, d = Q.shape[-2], K.shape[-2], Q.shape[-1]  # Q [batch seq_len d] [4 128 64] O [batch_size, n_queries, D]
    if is_causal:
        mask = torch.triu(torch.ones(Nq, Nk, device=Q.device, dtype=torch.bool), diagonal=1)
    D = torch.sum(O * dO[0], dim=-1)
    S = einsum(Q, K, "... Nq d, ... Nk d -> ... Nq Nk") / np.sqrt(d)  # [... Nq Nk]
    if is_causal:
        S = S.masked_fill(mask, float('-inf'))
    if L.shape != S.shape[:-1]:  # From the last element towards the beginning
        # Try to reshape L to match S's batch dims
        L = L.view(*S.shape[:-1])
    P = torch.exp(S - L.unsqueeze(-1))
    dV = einsum(P, dO[0], "... Nq Nk, ... Nq d -> ... Nk d")
    dP = einsum(dO[0], V, "... Nq d, ... Nk d -> ... Nq Nk")
    dS = P * (dP - D[..., None])
    dQ = einsum(dS, K, "... Nq Nk, ... Nk d -> ... Nq d") / np.sqrt(d)
    dK = einsum(dS, Q, "... Nq Nk, ... Nq d -> ... Nk d") / np.sqrt(d)
    return dQ, dK, dV, None

compiled_backward = torch.compile(backward_pass_recomp)