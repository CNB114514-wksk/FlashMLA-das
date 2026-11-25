#include "../splitkv_mla.cuh"

namespace gfx93::decode::sparse_fp4 {

// V41 main cache (fp8) + fp4 extra_kv, split-KV path.
// Upstream counterpart: instantiations/v41fp4_h64.cu (Config{V41, V41_FP4, true}).
// HAVE_TOPK_LENGTH is instantiated transitively by the wrapper's dispatch below
// (same mechanism as sparse_fp8/instantiations/model1_persistent_h64.cu).
template void run_flash_splitkv_mla_fp4_sparse_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams &params);
template void run_flash_fixed_splitkv_mla_fp4_sparse_kernel<ModelType::MODEL1, 64>(const SparseAttnDecodeParams &params);

}