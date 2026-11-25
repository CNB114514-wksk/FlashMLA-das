#pragma once

#include "hytlass_ext.h"

#include <hytlass/numeric_types.h>
#include <hytlass/arch/barrier.h>
#include <hute/tensor.hpp>
#include <kerutils/kerutils.cuh>
#include <tuple>

#include "defines.h"
#include "params.h"

using namespace hute;

namespace gfx93::decode::sparse_fp4 {

// FP4 KV sparse decode MLA kernel (V4.1 fp4 extra-KV format, ported from upstream
// sm100/decode/sparse/head64 with Config{V41, V41_FP4}). Skeleton copied from
// gfx93::decode::sparse_fp8 with three increments:
//   (a) FP4 token layout (288 B/token = 256 B nibble-packed e2m1 + 32 B e4m3 scales,
//       token-interleaved: scales sit right after the raw row, upstream
//       kv_cache_format.h:18-19);
//   (b) issue_kv_loads FP4 branch (vector reads of the raw+inline-scale row);
//   (c) dequant_fp4_to_bf16 replacing the fp8 scalar path (components/dequant.h).
template<ModelType MODEL_TYPE, int NUM_HEADS, bool HAVE_TOPK_LENGTH>
class KernelTemplate {
public:

    static_assert(NUM_HEADS == 64 || NUM_HEADS == 128 || NUM_HEADS == 16);
    // todo only support tp8
    static constexpr int BLOCK_M = 16;
    static constexpr int NUM_M_BLOCKS = NUM_HEADS / BLOCK_M;
    static constexpr bool Is_causal = false;

    // FP4 V4.1: all 512 dims are quantized (D_FP4 = 512, upstream kv_cache_format.h:14),
    // including the RoPE dims that act as V.
    static_assert(MODEL_TYPE != ModelType::V32, "FP4 kernel keeps the MODEL1 (d_qk=512) layout");
    static constexpr int HEAD_DIM_K = 512;
    static constexpr int HEAD_DIM_V = 512;
    static constexpr int HEAD_DIM_ROPE = 64;
    static constexpr int HEAD_DIM_NOPE = HEAD_DIM_K - HEAD_DIM_ROPE;

    // ---- FP4 format constants (upstream kv_cache_format.h:10-19, V41_FP4 column) ----
    static constexpr bool IS_FP4 = true;
    static constexpr int QUANT_TILE_SIZE = 16;                 // dims sharing one e4m3 scale
    static constexpr int NUM_SCALES_EACH_TOKEN = 32;           // 512 / 16
    static constexpr int QUANT_BYTES = 256;                   // D_FP4 / 2 (nibble packed)
    static constexpr int BYTES_PER_TOKEN = 288;                // QUANT_BYTES + NUM_SCALES_EACH_TOKEN

    static constexpr int NUM_THREADS = 256;
    static constexpr int TOPK_BLOCK_SIZE = 64;
    // One scale byte per k_idx per warp lane-group (warp owns 16 consecutive dims = one tile).
    static constexpr int NUM_WARPS = NUM_THREADS / 64;
    static constexpr int NUM_SCALE_BYTES = 8 * 4;

    // Nibbled data payload: 8 bytes holding 16 consecutive e2m1 values (element i of a
    // row is in byte i//2, even elements in the low nibble; flash_mla_interface.py:119).
    union Fp4_storage {
        uint64_t data_64;
        uint32_t data_array_u32[2];
        uint16_t data_array[8];
        uint8_t nibble[8];
    };

    union bf16_storage {
        uint32x4_t data_128;
        uint16_t data_array[8];
    };

    // One topk block's K/V payload in registers (per thread, per warp offset):
    // 8 segments of 16 nibble-pair values + 32 inline e4m3 scale bytes.
    // Scale words hold the four e4m3 bytes of a k_idx segment contiguously
    // (byte g of word k = scale of dims [64k+16g, 64k+16g+16)).
    struct PfKV {
        Fp4_storage kA[4];      // k_idx 4..7 (dims 256..511)
        Fp4_storage kB[4];      // k_idx 0..3 (dims 0..255)
        uint32_t scale_words[8];
    };

    // This request's scheduling info
    struct MainloopArgs {
        int start_block_idx, end_block_idx;
        bool is_no_split;

        // The following fields are only valid for MODEL1
        int topk_length, extra_topk_length, num_orig_kv_blocks;
    };

    using elem_type = hytlass::bfloat16_t;
    using MMA_Atom_Arch = std::conditional_t<
        std::is_same_v<elem_type, hytlass::half_t>,
        MMA_Atom<flash_mla::hytlass_ext::GFX928_16x16x64_F32F16F16F32_NT>,
        MMA_Atom<flash_mla::hytlass_ext::GFX928_16x16x64_F32BF16BF16F32_NT>
    >;
    static constexpr int kNWarps = 4;
    using ValLayoutMNK = Layout<Shape<_1, _1, _1>>;
    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<_1, Int<kNWarps>, _1>>>;

    using MMA_Atom_Arch_16_16_32 = std::conditional_t<
        std::is_same_v<elem_type, hytlass::half_t>,
        MMA_Atom<flash_mla::hytlass_ext::GFX928_16x16x32_F32F16F16F32_NN>,
        MMA_Atom<flash_mla::hytlass_ext::GFX928_16x16x32_F32BF16BF16F32_NN>
    >;
    using TiledMma_16_16_32 = TiledMMA<
        MMA_Atom_Arch_16_16_32,
        Layout<Shape<_1, Int<kNWarps>, _1>>>;

    using MMA_Atom_Arch_16x32_NT = std::conditional_t<
        std::is_same_v<elem_type, hytlass::half_t>,
        MMA_Atom<flash_mla::hytlass_ext::GFX928_16x32x16_F32F16F16F32_NT>,
        MMA_Atom<flash_mla::hytlass_ext::GFX928_16x32x16_F32BF16BF16F32_NT>
    >;
    using TiledMma_O = TiledMMA<
        MMA_Atom_Arch_16x32_NT,
        Layout<Shape<_1, Int<kNWarps>, _1>>>;

    using SmemLayoutAtomK = decltype(composition(
        Swizzle<3, 3, 3>{},
        Layout<Shape<Int<8>, Int<32>>, Stride<Int<32>, _1>>{}));
    using SmemLayoutK = decltype(tile_to_shape(
        SmemLayoutAtomK{},
        Shape<Int<TOPK_BLOCK_SIZE>, Int<8 * 32>>{}));

    using SmemLayoutAtomV = SmemLayoutAtomK;
    using SmemLayoutV = decltype(tile_to_shape(
        SmemLayoutAtomV{},
        Shape<Int<TOPK_BLOCK_SIZE>, Int<512>>{}));

    using SmemLayoutVtransposed = decltype(
        composition(SmemLayoutV{}, make_layout(Shape<Int<512>, Int<TOPK_BLOCK_SIZE>>{}, GenRowMajor{})));

    using SmemLayoutAtomP = Layout<Shape<Int<4*16*16>>, Stride<Int<1>>>;
    using SmemLayoutP = decltype(tile_to_shape(
        SmemLayoutAtomP{},
        Shape<Int<4*16*16>>{}));
    using SmemLayoutRow = Layout<Shape<_128>, Stride<_1>>;

    using Element = hytlass::bfloat16_t;
    using ElementAccum = float;
    struct SharedMemoryPlan {
        union {
            struct {
                hute::array_aligned<Element, hute::cosize_v<SmemLayoutV>> smem_v;
            };
            struct {
                // cute::array_aligned<typename Kernel_traits::Element, cute::cosize_v<typename Kernel_traits::SmemLayoutV_tmp>> smem_v_tmp;  // Double buffer
                hute::array_aligned<Element, hute::cosize_v<SmemLayoutP>> smem_p;
                hute::array_aligned<ElementAccum, hute::cosize_v<SmemLayoutRow>> smem_row_sum;
                hute::array_aligned<ElementAccum, hute::cosize_v<SmemLayoutRow>> smem_row_max;

            };
        };
    };

    static __device__ __forceinline__ std::tuple<Element, Element, Element, Element>
    dequant_fp4_to_bf16(const Fp4_storage& data0, uint8_t scale, int idx);

    static __device__ __forceinline__ PfKV
    issue_kv_loads(const SparseAttnDecodeParams& params, int token_index, int col_idx, bool is_extra);

    static __device__ __forceinline__ void
    lds_direct_copy_q(const Element* q_ptr, const SparseAttnDecodeParams& params, Element* s_q,
                    int head_block_idx, int lane_idx, int warp_idx, int k_idx, int offset_k);

    static __device__ __forceinline__ MainloopArgs
    get_cur_req_info(const SparseAttnDecodeParams& params, const DecodingSchedMeta& sched_meta, int batch_idx);

    static __device__ __forceinline__ uint16_t
    float2bf16(float s);

    static __device__ __forceinline__ void
    compute_attn_1rowblock_splitkv_sparse_mla_fp4(const SparseAttnDecodeParams &params, const DecodingSchedMeta& sched_meta, int batch_idx);

    static __device__ __forceinline__ void
    devfunc(const SparseAttnDecodeParams &params);

    static __device__ __forceinline__ void
    devfunc_fixed_split(const SparseAttnDecodeParams &params);

    static void run(const SparseAttnDecodeParams &params);

    static void run_fixed_split(const SparseAttnDecodeParams &params);

};

}  // namespace gfx93::decode::sparse_fp4