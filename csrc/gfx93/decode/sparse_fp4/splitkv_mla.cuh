#pragma once

#include "hytlass_ext.h"

#include "splitkv_mla.h"

#include <kerutils/kerutils.cuh>

#include "utils.h"
#include "components/dequant.h"
#include "config.h"
#include "softmax.h"
using namespace hute;

namespace gfx93::decode::sparse_fp4 {
#define CUDART_L2E_F            1.442695041F

// ---- KernelTemplate 的 FP4 辅助函数（static，可直接内联，不依赖 this）----

// FP4 解包：16 个 nibble 打包的 e2m1 值（payload.data_64，8 字节，偶数下标在低半字节）
// × 本 lane 组负责的 e4m3 scale → 4×bf16。纯软件实现，见 components/dequant.h。
template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ __forceinline__ auto
KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::dequant_fp4_to_bf16(const Fp4_storage& data0, uint8_t scale, int idx)
    -> std::tuple<Element, Element, Element, Element> {
    auto bits4 = fp4x16_to_bf16x4(data0.data_64, scale, idx);  // 同源软件解包（host 单测覆盖）
    return {hytlass::bfloat16_t::bitcast(std::get<0>(bits4)),
            hytlass::bfloat16_t::bitcast(std::get<1>(bits4)),
            hytlass::bfloat16_t::bitcast(std::get<2>(bits4)),
            hytlass::bfloat16_t::bitcast(std::get<3>(bits4))};
}

// 取一个 topk block 的 FP4 载荷：行布局＝[256 B nibble 打包的 e2m1 raw][32 B e4m3 scale]
// （token-interleaved，upstream kv_cache_format.h:18-19）。每个 64 维段（k_idx）的四个
// 16 值 group 各占一个 scale 字节；lane 组 col_idx 拥有 dims [64k+16c, 64k+16c+16)。
// 上游等价语义：TMA gather 整行（raw+scale）见 kernel.cuh:654 与
// kv_cache_utils.cuh:41-62；此处为 das 的向量读版。
template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ __forceinline__ auto
KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::issue_kv_loads(const SparseAttnDecodeParams& params, int token_index, int col_idx, bool is_extra)
    -> PfKV {
    PfKV pf;
    uint8_t* k_ptr = is_extra ? (uint8_t*)params.extra_kv : (uint8_t*)params.kv;
    int page_block_size = is_extra ? params.extra_page_block_size : params.page_block_size;
    int64_t k_block_stride = is_extra ? params.stride_extra_kv_block : params.stride_kv_block;
    int64_t k_row_stride = is_extra ? params.stride_extra_kv_row : params.stride_kv_row;
    int block_index = token_index == -1 ? 0 : (int)((uint32_t)token_index / (uint32_t)page_block_size);
    int rel_idx_in_block = (uint32_t)token_index % (uint32_t)page_block_size;
    const int64_t offset_k = (int64_t)(block_index) * k_block_stride;
    (void)k_row_stride;  // 行距即 BYTES_PER_TOKEN，见 run() 断言
    // FP4 token 行：raw 256 B + inline scale 32 B（上游 layout，kernel.cuh:5-9 注释）
    uint8_t* gK_base = k_ptr + offset_k + rel_idx_in_block * BYTES_PER_TOKEN;
    if (token_index == -1) {
        for (int i = 0; i < 4; i++) {
            pf.kA[i].data_64 = 0ull;
            pf.kB[i].data_64 = 0ull;
        }
        for (int i = 0; i < 8; i++) pf.scale_words[i] = 0u;
    } else {
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            // kB[k] 覆盖 dims [64k, 64k+64)；lane 组取其中连续 16 维 = 8 字节
            pf.kB[k].data_64 = *reinterpret_cast<const uint64_t*>(gK_base + 32 * k + 8 * col_idx);
            // 该段四个 group 的 scale 连续存放于行内偏移 256+4k
            pf.scale_words[k] = *reinterpret_cast<const uint32_t*>(gK_base + 256 + 4 * k);
        }
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            pf.kA[k].data_64 = *reinterpret_cast<const uint64_t*>(gK_base + 32 * (4 + k) + 8 * col_idx);
            pf.scale_words[4 + k] = *reinterpret_cast<const uint32_t*>(gK_base + 256 + 4 * (4 + k));
        }
    }
    return pf;
}

template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ __forceinline__ void
KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::lds_direct_copy_q(const Element* q_ptr, const SparseAttnDecodeParams& params, Element* s_q,
                                                                          int head_block_idx, int lane_idx, int warp_idx, int k_idx, int offset_k) {
    struct PtrWrapper {
        uint32_t former;
        uint32_t latter;
    };
    PtrWrapper glob_ptr;
    *(uint64_t*)&glob_ptr = reinterpret_cast<uint64_t>(q_ptr);
    uint32x4_t global_addr = {0};
    global_addr[0] = (glob_ptr.former);
    global_addr[1] = (glob_ptr.latter);
    global_addr[2] = 0x80000000;
    global_addr[3] = 0x00020000;
    constexpr int bytes_per_warp = 64 * 8 * 2;
    constexpr int bytes_per_block = bytes_per_warp * 4;
    const int row_idx = lane_idx % 16;
    const int col_idx = lane_idx / 16;
    const int row_offset = row_idx;
    const int col_offset = k_idx * 256 + warp_idx * 64 + col_idx * 16 + offset_k * 8;
    int offset_v = (row_offset * params.stride_q_h_q + col_offset) * 2;
    if (head_block_idx * BLOCK_M + row_idx >= params.h_q) {
        offset_v = -1;
    }
    const int offset_s = 0;
    int ldsAddrPerWave = reinterpret_cast<size_t>(s_q) + warp_idx * bytes_per_warp + k_idx * bytes_per_block
        + offset_k * 2 * bytes_per_block;
    __builtin_amdgcn_sched_barrier(0);
    asm volatile(
        "s_mov_b32 m0, %1 \n\t"
        "s_nop 0 \n\t"
        "buffer_load_dwordx4 %0, %2, %3 ,offen  offset:0, lds \n" ::"v"(offset_v),
        "s"(ldsAddrPerWave), "s"(global_addr), "s"(offset_s)
    :);
    __builtin_amdgcn_sched_barrier(0);
}

// 解析本次请求的调度信息（block 范围、是否 no-split、topk_length 等）
template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ __forceinline__ auto
KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::get_cur_req_info(const SparseAttnDecodeParams& params, const DecodingSchedMeta& sched_meta, int batch_idx)
    -> MainloopArgs {
    MainloopArgs args;
    int total_topk_padded;
    if constexpr (HAVE_TOPK_LENGTH) {
        // 动态 topk_length 截断
        int topk_length = __ldg(params.topk_length + batch_idx);
        total_topk_padded = max(ku::ceil(topk_length, (int)TOPK_BLOCK_SIZE), (int)TOPK_BLOCK_SIZE);
        args.topk_length = topk_length;
        args.num_orig_kv_blocks = total_topk_padded / TOPK_BLOCK_SIZE;
    } else {
        // 静态 topk：编译期不生成 topk_length 相关代码，与基线等价
        total_topk_padded = params.topk;
        args.num_orig_kv_blocks = params.topk / TOPK_BLOCK_SIZE;
    }
    // extra_kv（R1-a：fp4 extra cache 挂在 fp8 主 cache 旁）与 topk_length 相互独立，
    // 静态/动态都需截断 extra_topk_length（对齐 sparse_fp8 的 MODEL1 行为）。
    int extra_topk_length = params.extra_topk_length ? __ldg(params.extra_topk_length + batch_idx) : params.extra_topk;
    total_topk_padded += ku::ceil(extra_topk_length, (int)TOPK_BLOCK_SIZE);
    args.extra_topk_length = extra_topk_length;

    args.start_block_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_block_idx : 0;
    args.end_block_idx = batch_idx == sched_meta.end_req_idx ? sched_meta.end_block_idx : total_topk_padded / TOPK_BLOCK_SIZE;
    args.is_no_split = batch_idx == sched_meta.begin_req_idx ? !sched_meta.is_first_req_splitted : (batch_idx == sched_meta.end_req_idx ? !sched_meta.is_last_req_splitted : true);

    return args;
}

template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ __forceinline__ uint16_t
KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::float2bf16(float s) {
    uint32_t x32 = reinterpret_cast<uint32_t const &>(s);
    #ifndef FLASH_MLA_BF16_TYPE
    #define FLASH_MLA_BF16_TYPE 0
    #endif
    #if FLASH_MLA_BF16_TYPE == 1
    x32 += 0x8000u;
    #endif
    return uint16_t(x32 >> 16);
}

template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::compute_attn_1rowblock_splitkv_sparse_mla_fp4(const SparseAttnDecodeParams &params, const DecodingSchedMeta& sched_meta, int batch_idx) {
    using Element = hytlass::bfloat16_t;
    using index_t = int64_t;
    const int tidx = threadIdx.x;
    const int lane_idx = tidx % 64;
    const int warp_off = lane_idx / 16;   // 本 lane 组的 16 维在段内的偏移（0..3）
    const int warp_idx = __builtin_amdgcn_readfirstlane(tidx / 64);
    const int head_block_idx = NUM_M_BLOCKS == 1 ? 0 : blockIdx.x;
    const int s_q_idx = blockIdx.y;
    extern __shared__ char shared_memory[];
    SharedMemoryPlan &plan = *reinterpret_cast<SharedMemoryPlan*>(shared_memory);

    const index_t row_offset_q = batch_idx * params.stride_q_b + head_block_idx * BLOCK_M * params.stride_q_h_q + s_q_idx * params.stride_q_s_q;
    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q) + row_offset_q),
                            Shape<Int<BLOCK_M>, Int<HEAD_DIM_K>>{},
                            make_stride(params.stride_q_h_q, _1{}));
    TiledMMA tiled_mma = TiledMma{};
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma.partition_fragment_A(gQ);
    // 需要的最大空间为 16 * 576 * 2
    Element* s_q = reinterpret_cast<Element *>(shared_memory);

    const auto q_ptr = gQ.data().get();
    lds_direct_copy_q(q_ptr, params, s_q, head_block_idx, lane_idx, warp_idx, 0, 0);
    lds_direct_copy_q(q_ptr, params, s_q, head_block_idx, lane_idx, warp_idx, 1, 0);
    lds_direct_copy_q(q_ptr, params, s_q, head_block_idx, lane_idx, warp_idx, 0, 1);
    lds_direct_copy_q(q_ptr, params, s_q, head_block_idx, lane_idx, warp_idx, 1, 1);

    Element* s_q_read_ptr = s_q + lane_idx * 8;
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(3) \n s_barrier");
    __builtin_amdgcn_sched_barrier(0);
    for (int k = 0; k < 4; k++) {
        for (int i = 0; i < 8; i++) {
            tSrQ(i, 0, k) = s_q_read_ptr[i];
        }
        s_q_read_ptr += 16 * 32;
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(2) \n s_barrier");
    __builtin_amdgcn_sched_barrier(0);
    for (int k = 4; k < 8; k++) {
        for (int i = 0; i < 8; i++) {
            tSrQ(i, 0, k) = s_q_read_ptr[i];
        }
        s_q_read_ptr += 16 * 32;
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(1) \n s_barrier");
    __builtin_amdgcn_sched_barrier(0);
    for (int k = 0; k < 4; k++) {
        for (int i = 8; i < 16; i++) {
            tSrQ(i, 0, k) = s_q_read_ptr[i - 8];
        }
        s_q_read_ptr += 16 * 32;
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt vmcnt(0) \n s_barrier");
    __builtin_amdgcn_sched_barrier(0);
    for (int k = 4; k < 8; k++) {
        for (int i = 8; i < 16; i++) {
            tSrQ(i, 0, k) = s_q_read_ptr[i - 8];
        }
        s_q_read_ptr += 16 * 32;
    }
    __builtin_amdgcn_sched_barrier(0);
    asm volatile("s_waitcnt lgkmcnt(0) \n s_barrier");
    __builtin_amdgcn_sched_barrier(0);

    Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<uint8_t *>(params.kv)),
                            Shape<Int<TOPK_BLOCK_SIZE>, Int<HEAD_DIM_K>>{},
                            make_stride(params.stride_kv_row, _1{}));
    Tensor tSrK  = thr_mma.partition_fragment_B(gK);
    TiledMMA tiled_mma_16x16x32 = TiledMma_16_16_32{};
    auto smem_tiled_copy_K = make_tiled_copy_B(Copy_Atom<DefaultCopy, Element>{}, tiled_mma_16x16x32);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor sK = make_tensor(make_smem_ptr(plan.smem_v.data()), SmemLayoutK{});
    Tensor tOsV = smem_thr_copy_K.partition_S(sK);

    TiledMMA tiled_mma_o = TiledMma_O{};
    auto thr_mma_o = tiled_mma_o.get_thread_slice(tidx);
    auto smem_tiled_copy_V = make_tiled_copy_B(Copy_Atom<flash_mla::hytlass_ext::GFX928_DS_READ_DS_M32x16_B16, Element>{}, tiled_mma_o);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor sV = make_tensor(make_smem_ptr(plan.smem_v.data()), SmemLayoutV{});
    Tensor sVt = make_tensor(sV.data(), SmemLayoutVtransposed{});
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);
    Tensor tOrVt  = thr_mma_o.partition_fragment_B(sVt);
    Tensor tOrVt_copy_view = smem_thr_copy_V.retile_D(tOrVt);

    Tensor acc_o = partition_fragment_C(tiled_mma_o, Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{});
    clear(acc_o);
    flash::Softmax<size<1>(acc_o)> softmax;
    MainloopArgs args = get_cur_req_info(params, sched_meta, batch_idx);

    // idx_of / consume_block_pf 依赖的上下文（indices 与 smem 视图）
    const index_t row_offset_topk =  batch_idx * params.stride_indices_b + s_q_idx * params.stride_indices_s_q;
    int* gIndices = reinterpret_cast<int *>(params.indices) + row_offset_topk;
    int* gExtraIndices = params.extra_indices + batch_idx*params.stride_extra_indices_b + s_q_idx*params.stride_extra_indices_s_q; // (extra_topk) : (1)
    Tensor sP = make_tensor(make_smem_ptr(plan.smem_p.data()), SmemLayoutP{});
    Tensor sRow_max_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_max.data()), SmemLayoutRow{});
    Tensor sRow_sum_reduce_buffer = make_tensor(make_smem_ptr(plan.smem_row_sum.data()), SmemLayoutRow{});

    // 取 block_idx 对应 lane 的 token：超出 topk_length 的位置按 -1 处理，使 K/V 加载为 0
    // （避免 0 x NaN 在 PV 中传播出 NaN）。
    auto idx_of = [&](int block_idx, bool is_extra) -> int {
        const int off = (lane_idx % 16) + warp_idx * 16;
        int* indices_base = is_extra
            ? gExtraIndices + (block_idx - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE
            : gIndices + block_idx * TOPK_BLOCK_SIZE;
        [[maybe_unused]] int rel_block_idx = is_extra ? (block_idx - args.num_orig_kv_blocks) : block_idx;
        [[maybe_unused]] int topk_length = is_extra ? args.extra_topk_length : args.topk_length;
        int token = indices_base[off];
        if (is_extra || HAVE_TOPK_LENGTH) {
            if (rel_block_idx * TOPK_BLOCK_SIZE + off >= topk_length) token = -1;
        }
        return token;
    };

    struct IsOrigBlock {};
    struct IsExtraBlock {};

    // 消费一个已预取到寄存器的 block：QK -> mask -> softmax -> PV
    auto consume_block_pf = [&](int block_idx, const PfKV& pf, auto is_extra_block_t) {
        static constexpr bool IS_EXTRA_BLOCK = std::is_same_v<decltype(is_extra_block_t), IsExtraBlock>;
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
        clear(acc_s);
        int* indices_base = IS_EXTRA_BLOCK
            ? gExtraIndices + (block_idx - args.num_orig_kv_blocks) * TOPK_BLOCK_SIZE
            : gIndices + block_idx * TOPK_BLOCK_SIZE;
        [[maybe_unused]] int topk_length = IS_EXTRA_BLOCK ? args.extra_topk_length : args.topk_length;
        [[maybe_unused]] int rel_block_idx = IS_EXTRA_BLOCK ? (block_idx - args.num_orig_kv_blocks) : block_idx;

        // QK A: k_idx 4..7 (dims 256..511)，FP4 下全部量化（含 RoPE/V 维，上游 D_FP4=512）
        for (int k_idx = 4; k_idx < 8; ++k_idx) {
            for (int j = 0; j < 16; j += 4) {
                const uint8_t scale_byte = (uint8_t)((pf.scale_words[k_idx] >> (8 * warp_off)) & 0xFFu);
                auto [rst0, rst1, rst2, rst3] = dequant_fp4_to_bf16(pf.kA[k_idx - 4], scale_byte, j);
                tSrK(j, 0, k_idx) = rst0;
                tSrK(j + 1, 0, k_idx) = rst1;
                tSrK(j + 2, 0, k_idx) = rst2;
                tSrK(j + 3, 0, k_idx) = rst3;
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                tOsV(j, 0, (k_idx - 4) * 2) = tSrK(j, 0, k_idx);
            }
            #pragma unroll
            for (int j = 8; j < 16; j++) {
                tOsV(j - 8, 0, (k_idx - 4) * 2 + 1) = tSrK(j, 0, k_idx);
            }
            hute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        }
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 0, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 1, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 2, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 3, 2>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<1, 0, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<1, 1, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<1, 2, 3>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<1, 3, 3>(tOsVt, tOrVt_copy_view);
        __builtin_amdgcn_sched_barrier(0);
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        // QK B: k_idx 0..3 (dims 0..255)
        for (int k_idx = 0; k_idx < 4; ++k_idx) {
            for (int j = 0; j < 16; j += 4) {
                const uint8_t scale_byte = (uint8_t)((pf.scale_words[k_idx] >> (8 * warp_off)) & 0xFFu);
                auto [rst0, rst1, rst2, rst3] = dequant_fp4_to_bf16(pf.kB[k_idx], scale_byte, j);
                tSrK(j, 0, k_idx) = rst0;
                tSrK(j + 1, 0, k_idx) = rst1;
                tSrK(j + 2, 0, k_idx) = rst2;
                tSrK(j + 3, 0, k_idx) = rst3;
            }
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                tOsV(j, 0, k_idx * 2) = tSrK(j, 0, k_idx);
            }
            #pragma unroll
            for (int j = 8; j < 16; j++) {
                tOsV(j - 8, 0, k_idx * 2 + 1) = tSrK(j, 0, k_idx);
            }
            hute::gemm(tiled_mma, tSrQ(_, _, k_idx), tSrK(_, _, k_idx), acc_s);
        }
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 0, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 1, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 2, 0>(tOsVt, tOrVt_copy_view);
        flash::__ds_read_m32x16_row_col_rrow_alt<0, 3, 0>(tOsVt, tOrVt_copy_view);
        __builtin_amdgcn_sched_barrier(0);
        asm volatile("s_waitcnt lgkmcnt(0) \n\t s_barrier\n\t");
        __builtin_amdgcn_sched_barrier(0);

        // mask + softmax
        Tensor cS = make_identity_tensor(Shape<Int<BLOCK_M>, Int<TOPK_BLOCK_SIZE>>{});
        Tensor tScS = thr_mma.partition_C(cS);
        for (int i = 0; i < size(acc_s); ++i) {
            if constexpr (IS_EXTRA_BLOCK || HAVE_TOPK_LENGTH) {
                int col = int(get<1>(tScS(i)));
                if (indices_base[col] == -1 || (rel_block_idx * TOPK_BLOCK_SIZE + col >= topk_length)) acc_s(i) = -INFINITY;
            } else {
                if (indices_base[int(get<1>(tScS(i)))] == -1) acc_s(i) = -INFINITY;
            }
        }
        block_idx == args.start_block_idx
        ? softmax.template softmax_rescale_o_prefill</*Is_first=*/true,  /*Check_inf=*/Is_causal>(acc_s, acc_o, sRow_max_reduce_buffer, params.sm_scale_div_log2)
        :   softmax.template softmax_rescale_o_prefill</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, sRow_max_reduce_buffer, params.sm_scale_div_log2);
        Tensor rP = flash::convert_type<Element>(acc_s);
        Tensor tOrP = flash::convert_layout_acc_Aregs(tiled_mma, tiled_mma_o, rP, sP);

        // PV
        {
            flash::__ds_read_m32x16_row_col_alt<1, 0>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col_alt<1, 1>(tOsVt, tOrVt_copy_view);
            hute::gemm(tiled_mma_o, tOrP(_, _, 0), tOrVt(_, _, 0), acc_o);
            hute::gemm(tiled_mma_o, tOrP(_, _, 1), tOrVt(_, _, 1), acc_o);
            flash::__ds_read_m32x16_row_col_alt<1, 2>(tOsVt, tOrVt_copy_view);
            flash::__ds_read_m32x16_row_col_alt<1, 3>(tOsVt, tOrVt_copy_view);
            hute::gemm(tiled_mma_o, tOrP(_, _, 2), tOrVt(_, _, 2), acc_o);
            hute::gemm(tiled_mma_o, tOrP(_, _, 3), tOrVt(_, _, 3), acc_o);
        }
    };

    const int col_idx_pf = lane_idx / 16;
    for (int block_idx = args.start_block_idx; block_idx < min(args.num_orig_kv_blocks, args.end_block_idx); ++block_idx) {
        PfKV pf = issue_kv_loads(params, idx_of(block_idx, false), col_idx_pf, false);
        consume_block_pf(block_idx, pf, IsOrigBlock{});
    }
    for (int block_idx = max(args.start_block_idx, args.num_orig_kv_blocks); block_idx < args.end_block_idx; ++block_idx) {
        PfKV pf = issue_kv_loads(params, idx_of(block_idx, true), col_idx_pf, true);
        consume_block_pf(block_idx, pf, IsExtraBlock{});
    }

    if (args.is_no_split) {
        int start_head_idx = head_block_idx*BLOCK_M;
        Tensor lse = softmax.template normalize_softmax_lse<false>(acc_o, sRow_sum_reduce_buffer, params.sm_scale);
        const index_t row_offset_o = batch_idx * params.stride_o_b + start_head_idx * params.stride_o_h_q + s_q_idx * params.stride_o_s_q ;
        Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.out) + row_offset_o),
                                        Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
                                        make_stride(params.stride_o_h_q, _1{}));
        if (params.attn_sink != nullptr) {
            float rAttn_sink = __ldg((float*)params.attn_sink + start_head_idx + lane_idx % 16);
            if (flash::is_positive_infinity(rAttn_sink)) {
                for (int i = 0; i < size(acc_o); i++) {
                    acc_o(i) = 0.0f;
                }
            }
            else
            {
                if (!flash::is_positive_infinity(lse(0)))
                {
                    // Stable sigmoid(LSE - sink): never exponentiate a positive value.
                    float sink_delta = (lse[0] - rAttn_sink) * CUDART_L2E_F;
                    float sink_exp_neg_abs = __builtin_amdgcn_exp2f(-fabsf(sink_delta));
                    float o_scale = (sink_delta >= 0.0f ? 1.0f : sink_exp_neg_abs)
                        / (1.0f + sink_exp_neg_abs);
                    for (int i = 0; i < size(acc_o); i++)
                    {
                        acc_o(i) *= o_scale;
                    }
                }
            }

        }
        float* gSoftmaxLse = (float*)params.lse + batch_idx * params.stride_lse_b + start_head_idx + s_q_idx * params.stride_lse_s_q;	// (BLOCK_M) : (1)
        {
            using result_type = hytlass::Array<Element, 2>;
            int row, col;
            const int warpId = tidx / 64;
            const int laneId = tidx % 64;
            for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                row = mi * BLOCK_M + laneId % 16;
                if (row < params.h_q) {
                    for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                        // 为了使用global_loadx4指令, V矩阵吸入lds的时候 N方向发生了了交换
                        /*
                        ------------------- N 方向----------------------
                        |0 1 ... 7 16 ... 31 40 ... 47 56... 64 8 .. 15 32 ... 39
                        |
                        |
                        k
                        方向
                        |
                        |
                        |
                        */
                        col = (laneId / 16) * 2 + ni * 128 + (warpId % 2) * 8 + (warpId / 2) * 64;
                        for (int i = 0; i < 4; i ++) {

                            #if defined(__gfx938__)
                            auto d =  __builtin_hcu_cvt_pk_bf16_f32(0, acc_o(i, mi, ni), 0, acc_o(i + 4, mi, ni), 0);
                            auto res = reinterpret_cast<result_type const &>(d);
                            #else
                            result_type res;
                            Element e0, e1;
                            e0.storage = float2bf16(acc_o(i, mi, ni));
                            e1.storage = float2bf16(acc_o(i + 4, mi, ni));
                            res[0] = e0;
                            res[1] = e1;
                            #endif
                            *(result_type*)(&gO(row, col)) = res;
                            col += 16;
                        }
                    }
                    if (tidx / 16 == 0)
                        gSoftmaxLse[row] = lse(mi);
                }
            }
        }
    } else {
        int start_head_idx = head_block_idx*BLOCK_M;
        Tensor lse = softmax.template normalize_softmax_lse<false, true>(acc_o, sRow_sum_reduce_buffer, params.sm_scale);
        int n_split_idx = batch_idx == sched_meta.begin_req_idx ? sched_meta.begin_split_idx : 0;
        int split_idx = params.use_fixed_split
            ? batch_idx * params.fixed_num_splits + n_split_idx
            : __ldg(params.num_splits_ptr + batch_idx) + n_split_idx;
        float* oaccum_ptr = (float*)params.o_accum + split_idx*params.stride_o_accum_split + s_q_idx*params.stride_o_accum_s_q + start_head_idx*params.stride_o_accum_h_q;	// (BLOCK_M, HEAD_DIM_V) : (params.stride_o_accum_h_q, 1)
        Tensor gOaccum = make_tensor(make_gmem_ptr(oaccum_ptr), make_layout(
            Shape<Int<BLOCK_M>, Int<HEAD_DIM_V>>{},
            make_stride(params.stride_o_accum_h_q, _1{})
        ));
        float* gSoftmaxLseAccum = (float*)params.lse_accum + split_idx*params.stride_lse_accum_split + s_q_idx*params.stride_lse_accum_s_q + start_head_idx;	// (BLOCK_M) : (1)
        {
            int row, col;
            const int warpId = tidx / 64;
            const int laneId = tidx % 64;
            for (int mi = 0; mi < size<1>(acc_o); ++mi) {
                row = mi * BLOCK_M + laneId % 16;
                if (row < params.h_q) {
                    for (int ni = 0; ni < size<2>(acc_o); ++ni) {
                        col = (laneId / 16) * 2 + ni * 128 + (warpId % 2) * 8 + (warpId / 2) * 64;
                        for (int i = 0; i < 4; i ++) {
                            gOaccum(row, col) = acc_o(i, mi, ni);
                            gOaccum(row, col + 1) = acc_o(i + 4, mi, ni);
                            col += 16;
                        }
                    }
                    if (tidx / 16 == 0)
                        gSoftmaxLseAccum[row] = lse(mi);
                }
            }
        }
    }

}


template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::devfunc(const SparseAttnDecodeParams &params) {
    const int partition_idx = blockIdx.z;
    DecodingSchedMeta sched_meta = params.tile_scheduler_metadata_ptr[partition_idx];

    if (sched_meta.begin_req_idx >= params.b) return;
    for (int batch_idx = sched_meta.begin_req_idx; batch_idx <= sched_meta.end_req_idx; ++batch_idx) {
        if (batch_idx > sched_meta.begin_req_idx) {
            __syncthreads();
        }
        compute_attn_1rowblock_splitkv_sparse_mla_fp4(params, sched_meta, batch_idx);
    }
}

template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
__device__ void KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::devfunc_fixed_split(
    const SparseAttnDecodeParams &params) {
    const int split_linear_idx = blockIdx.z;
    const int split_idx = split_linear_idx % params.fixed_num_splits;
    const int batch_idx = split_linear_idx / params.fixed_num_splits;
    if (batch_idx >= params.b) return;

    // 逐请求有效 token 数：动态 topk 时按 topk_length 截断，否则回到全量 topk。
    // max(...,1) 保证零有效 token 也至少调度 1 个 block，由 lane 掩码产出 lse=+inf。
    int total_blocks;
    if constexpr (HAVE_TOPK_LENGTH) {
        const int req_topk = std::min((int)params.topk_length[batch_idx], params.topk);
        total_blocks = std::max(hute::ceil_div(req_topk, TOPK_BLOCK_SIZE), 1);
    } else {
        total_blocks = params.topk / TOPK_BLOCK_SIZE;
    }
    const int blocks_per_split = hute::ceil_div(total_blocks, params.fixed_num_splits);

    DecodingSchedMeta fixed_meta{};
    fixed_meta.begin_req_idx = batch_idx;
    fixed_meta.end_req_idx = batch_idx;
    fixed_meta.begin_block_idx = split_idx * blocks_per_split;
    fixed_meta.end_block_idx = min(total_blocks, fixed_meta.begin_block_idx + blocks_per_split);
    fixed_meta.begin_split_idx = split_idx;
    fixed_meta.is_first_req_splitted = params.fixed_num_splits > 1;
    fixed_meta.is_last_req_splitted = params.fixed_num_splits > 1;

    compute_attn_1rowblock_splitkv_sparse_mla_fp4(params, fixed_meta, batch_idx);
}

template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
flash_fwd_splitkv_mla_fp4_sparse_kernel(const SparseAttnDecodeParams params) {
#if defined(__gfx936__) || defined(__gfx938__)
    Kernel::devfunc(params);
#endif
}

template<typename Kernel>
__global__ void __launch_bounds__(Kernel::NUM_THREADS, 1)
flash_fwd_fixed_splitkv_mla_fp4_sparse_kernel(const SparseAttnDecodeParams params) {
#if defined(__gfx936__) || defined(__gfx938__)
    Kernel::devfunc_fixed_split(params);
#endif
}

template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
void KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::run(const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    // FP4 行布局：256 B nibble 打包 e2m1 + 32 B e4m3 inline scales（上游 V41_FP4，288 B/token）
    constexpr int BYTES_PER_TOKEN_FP4 = 288;
    KU_ASSERT(params.stride_kv_row == BYTES_PER_TOKEN_FP4, "Each page block in KV cache must be contiguous for head64 sparse fp4 decoding attention");
    if (params.extra_kv != nullptr) {
        KU_ASSERT(params.stride_extra_kv_row == BYTES_PER_TOKEN_FP4, "Each page block in extra KV cache must be contiguous for head64 sparse fp4 decoding attention");
    }
    auto mla_kernel = &flash_fwd_splitkv_mla_fp4_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>>;
    constexpr size_t smem_size = 32768; // lds复用
    mla_kernel<<<dim3(NUM_M_BLOCKS, params.s_q, params.num_sm_parts), NUM_THREADS, smem_size, params.stream>>>(params);
}

template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
void KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>::run_fixed_split(
    const SparseAttnDecodeParams &params) {
    KU_ASSERT(params.h_kv == 1);
    KU_ASSERT(params.topk % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.d_qk == HEAD_DIM_K);
    KU_ASSERT(params.d_v == HEAD_DIM_V);
    KU_ASSERT(params.use_fixed_split);
    KU_ASSERT(params.fixed_num_splits >= 1);
    KU_ASSERT(params.fixed_partition_size >= TOPK_BLOCK_SIZE);
    KU_ASSERT(params.fixed_partition_size % TOPK_BLOCK_SIZE == 0);
    KU_ASSERT(params.stride_kv_row == 288);
    if (params.extra_kv != nullptr) {
        KU_ASSERT(params.stride_extra_kv_row == 288);
    }

    auto mla_kernel =
        &flash_fwd_fixed_splitkv_mla_fp4_sparse_kernel<KernelTemplate<MODEL_TYPE, NUM_HEADS, HAVE_TOPK_LENGTH>>;
    constexpr size_t smem_size = 32768;
    mla_kernel<<<dim3(NUM_M_BLOCKS, params.s_q, params.b * params.fixed_num_splits),
                 NUM_THREADS, smem_size, params.stream>>>(params);
    CHECK_CUDA_KERNEL_LAUNCH();
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_splitkv_mla_fp4_sparse_kernel(const SparseAttnDecodeParams &params) {
    // 与 sparse_fp8 相同的分发习惯：有 topk_length 走动态特化，否则走静态特化
    if (params.topk_length != nullptr) {
        KernelTemplate<MODEL_TYPE, NUM_HEADS, true>::run(params);
    } else {
        KernelTemplate<MODEL_TYPE, NUM_HEADS, false>::run(params);
    }
}

template<ModelType MODEL_TYPE, int NUM_HEADS>
void run_flash_fixed_splitkv_mla_fp4_sparse_kernel(const SparseAttnDecodeParams &params) {
    if (params.topk_length != nullptr) {
        KernelTemplate<MODEL_TYPE, NUM_HEADS, true>::run_fixed_split(params);
    } else {
        KernelTemplate<MODEL_TYPE, NUM_HEADS, false>::run_fixed_split(params);
    }
}

}