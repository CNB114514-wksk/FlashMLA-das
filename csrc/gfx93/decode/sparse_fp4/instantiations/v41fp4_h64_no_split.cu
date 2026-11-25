#include "../splitkv_mla.cuh"

namespace gfx93::decode::sparse_fp4 {

// V41 main cache (fp8) + fp4 extra_kv, fixed-split path.
// Upstream counterpart: instantiations/v41fp4_h64_no_split.cu (Config{V41, V41_FP4, false}).
template void run_flash_splitkv_mla_fp4_sparse_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams &params);
template void run_flash_fixed_splitkv_mla_fp4_sparse_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams &params);

}