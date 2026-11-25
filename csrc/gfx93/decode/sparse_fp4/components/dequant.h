#pragma once

// FP4 (e2m1 data + e4m3 scale) software dequantization for gfx936/gfx928.
//
// Pure software implementation: DCU (Hygon gfx936/gfx928) has NO native fp4 support,
// so no fp4 hardware intrinsics are used anywhere in this port (the A0 probe from
// fp4_port_plan.md is skipped per user constraint). All decoding is done with plain
// integer / float ops, which also makes these functions callable from host code for
// unit testing (see test_fp4_dequant.cu).
//
// Reference semantics come from upstream tests/quant.py:
//   - nibble packing: element i of a row is in byte i//2, even elements in the low
//     nibble (upstream kv_cache_format.h V41_FP4 + flash_mla_interface.py:119);
//   - e2m1 magnitudes indexed by the 3 low bits, bit 3 is the sign
//     (_E2M1_MAGNITUDES = [0, 0.5, 1, 1.5, 2, 3, 4, 6], quant.py:29);
//   - e2m1 x e4m3 has at most 2 + 4 significant bits, so the product is exact in bf16
//     (quant.py:187).

#include <cstdint>
#include <tuple>

namespace gfx93::decode::sparse_fp4 {

// ---- helpers ---------------------------------------------------------------

inline __host__ __device__ float fp4_bits_to_float(uint32_t bits) {
    union { uint32_t as_bits; float as_value; } u;
    u.as_bits = bits;
    return u.as_value;
}

inline __host__ __device__ uint32_t fp4_float_to_bits(float f) {
    union { uint32_t as_bits; float as_value; } u;
    u.as_value = f;
    return u.as_bits;
}

// Pack an fp32 value into bf16 bit patterns with round-toward-zero. Used instead of
// round-to-nearest because the products are exact in bf16 anyway (see header comment),
// which makes rtz == rne bit-for-bit and matches the repo precedent
// hytlass::NumericConverter<..., round_toward_zero> in sparse_fp8/splitkv_mla.cuh:44.
inline __host__ __device__ uint16_t fp4_pack_bf16_rtz(float f) {
    const uint32_t b = fp4_float_to_bits(f);
    if ((b & 0x80000000u) && (b & 0xFFFFu)) {
        return static_cast<uint16_t>((b + 0x10000u) >> 16);  // negative: nudge up to truncate toward zero
    }
    return static_cast<uint16_t>(b >> 16);
}

// ---- e4m3 scale expansion ----------------------------------------------------

// Expand one e4m3 byte to fp32.
// Matches torch.float8_e4m3fn semantics (verified against torch 2.14 CPU, see
// /root/logs/fp4_e4m3_table.json / fp4_e4m3_probe.py): 1 sign + 4 exp (bias 7) + 3
// mantissa; exp field 15 with mantissa 0..6 carries the extended finite range up to
// 448 (the format has no infinities), and only 0x7F is NaN.
inline __host__ __device__ float fp4_e4m3_to_fp32(uint8_t b) {
    const uint32_t e = (static_cast<uint32_t>(b) >> 3) & 0xFu;
    const uint32_t m = static_cast<uint32_t>(b) & 7u;
    const uint32_t sign = (static_cast<uint32_t>(b) & 0x80u) ? 0x80000000u : 0u;

    uint32_t bits;
    if (e == 0u) {
        // Subnormal code: value = m * 2^-9 (m in 0..7). Since 2^-9 >> 2^-126 these are
        // all normal fp32 numbers; the table keeps it exact and host-safe.
        static constexpr uint32_t SUBNORMAL_BITS[8] = {
            0x00000000u,                   // m=0 (+0)
            0x3B000000u,                   // m=1: 1.0 x 2^-9
            0x3B800000u,                   // m=2: 1.0 x 2^-8
            0x3BC00000u,                   // m=3: 1.5 x 2^-8
            0x3C000000u,                   // m=4: 1.0 x 2^-7
            0x3C200000u,                   // m=5: 1.25 x 2^-7
            0x3C400000u,                   // m=6: 1.5 x 2^-7
            0x3C600000u,                   // m=7: 1.75 x 2^-7
        };
        bits = SUBNORMAL_BITS[m] | sign;
    } else {
        // Normal codes, INCLUDING the extended e=15 zone (256..448, m<=6); only
        // e=15 & m=7 (0x7F) is NaN (quiet-nan bit set).
        if (e == 15u && m == 7u) {
            bits = 0x7FC00000u | sign;  // quiet NaN
        } else {
            bits = (e + 120u) << 23 | m << 20 | sign;  // (e - 7 + 127) << 23 | mantissa
        }
    }
    return fp4_bits_to_float(bits);
}

// ---- nibble unpack + scaled product ------------------------------------------

// 16 nibble-packed e2m1 values (one 64-B-less-than-... warp payload: 8 bytes, element i
// in byte i//2, even elements in the low nibble) -> 4 bf16 bit patterns at positions
// [idx, idx+4) multiplied by the e4m3 `scale` byte. Pure software.
inline __host__ __device__ std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>
fp4x16_to_bf16x4(uint64_t packed, uint8_t scale, int idx) {
    static constexpr float E2M1_MAGNITUDES[8] =
        {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};  // quant.py:29
    const float s = fp4_e4m3_to_fp32(scale);

    auto code_at = [packed](int i) -> uint8_t {
        const uint8_t byte = static_cast<uint8_t>((packed >> ((i >> 1) * 8)) & 0xFFu);
        return static_cast<uint8_t>((byte >> ((i & 1) * 4)) & 0xFu);
    };

    float out[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const uint8_t code = code_at(idx + i);
        out[i] = E2M1_MAGNITUDES[code & 7u]
               * ((code & 8u) ? -1.0f : 1.0f)   // bit 3 is the sign
               * s;
    }
    return {fp4_pack_bf16_rtz(out[0]), fp4_pack_bf16_rtz(out[1]),
            fp4_pack_bf16_rtz(out[2]), fp4_pack_bf16_rtz(out[3])};
}

}  // namespace gfx93::decode::sparse_fp4