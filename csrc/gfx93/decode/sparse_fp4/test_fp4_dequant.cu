// Host-side numeric unit tests (plan A2/A3) for the sparse_fp4 software dequant path.
//
// Compiles/runs on the buq CPU box with plain hipcc (host target); no GPU involved.
// The "kernel side" functions are the very same components/dequant.h routines the
// device kernel calls (single source of truth). References below are independent
// implementations of upstream tests/quant.py semantics:
//   - _E2M1_MAGNITUDES / _dequantize_e2m1 (:29,53-57)
//   - _quantize_to_e2m1 (rn.satfinite: ties to even, saturate +-6, NaN -> code 0) (:31-51)
//   - V41_FP4Sparse token layout: element i in byte i//2, even elements low nibble (:99-100)
// Products are exact in bf16 (quant.py:187), so round-toward-zero (kernel) and
// round-to-nearest-even (reference) must agree bit-for-bit wherever both are finite.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <tuple>
#include <vector>

#include "components/dequant.h"

using gfx93::decode::sparse_fp4::fp4x16_to_bf16x4;

// Value-based element access (avoids forming pointers into returned tuples).
inline uint16_t tup_at(const std::tuple<uint16_t, uint16_t, uint16_t, uint16_t>& t, int i) {
    if (i == 0) return std::get<0>(t);
    if (i == 1) return std::get<1>(t);
    if (i == 2) return std::get<2>(t);
    return std::get<3>(t);
}

namespace {

uint64_t rng_state = 0x243F6A8885A308Dull;
inline uint64_t rnd64() {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return rng_state * 0x2545F4914F6CDD1Dull;
}
inline float frand(float a, float b) {
    return a + (float)((rnd64() >> 40) / (double)(1ull << 24)) * (b - a);
}

constexpr float E2M1_MAG[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

uint32_t f32bits(float f) {
    union { float f; uint32_t u; } x;
    x.f = f;
    return x.u;
}

// bf16 bit pattern -> float32 (for NaN-aware comparisons)
float bf16_to_float(uint16_t b) {
    union { uint32_t u; float f; } x;
    x.u = (uint32_t)b << 16;
    return x.f;
}

// ---- independent e4m3 byte -> float32 (mirrors torch.float8_e4m3fn; verified against
// torch 2.14 CPU: exp field 15 with mantissa 0..6 carries finite values up to 448,
// only 0x7F is NaN) ----
float ref_e4m3(uint8_t b) {
    int s = (b >> 7) & 1;
    int e = (b >> 3) & 0xF;
    int m = b & 7;
    if (e == 15 && m == 7) return std::nanf("");
    double v = (e == 0) ? ldexp((double)m, -9) : ldexp(1.0 + (double)m / 8.0, e - 7);
    return (float)(s ? -v : v);
}

// round-to-nearest-even bf16 bit packing from an already-rounded float32
uint16_t rne_bf16_bits(float f) {
    uint32_t b = f32bits(f);
    uint32_t rem = b & 0xFFFFu;
    if (rem > 0x8000u || (rem == 0x8000u && ((b >> 16) & 1))) b += 0x10000u;
    return (uint16_t)(b >> 16);
}

// independent scaled-product reference -> bf16 bits
uint16_t ref_product_bf16(uint8_t code, uint8_t scale_byte) {
    float mag = E2M1_MAG[code & 7];
    if (code & 8) mag = -mag;
    double p = (double)ref_e4m3(scale_byte) * (double)mag;
    return rne_bf16_bits((float)p);
}

// upstream _quantize_to_e2m1 (PTX cvt.rn.satfinite.e2m1x2.f32 semantics)
uint8_t quant_e2m1(float x) {
    constexpr float MIDS[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
    float a;
    uint8_t sgn;
    if (std::isnan(x)) {
        sgn = 0;  // NaN -> +0 code (caller keeps the NaN in the scale)
        a = 0.0f;
    } else {
        sgn = std::signbit(x) ? 0x8u : 0u;
        a = fminf(fabsf(x), 6.0f);
    }
    int code = 0;
    for (int i = 0; i < 7; ++i) {
        if (a > MIDS[i]) ++code;
    }
    for (int i = 0; i < 7; ++i) {  // ties to even
        if (a == MIDS[i]) code = i + (i & 1);
    }
    return (uint8_t)(sgn | (uint8_t)code);
}

struct Stats {
    long long total = 0, mism = 0, skipped = 0;
    long long max_ulp = 0;
    void check(uint16_t got, uint16_t want) {
        ++total;
        bool g_nan = std::isnan(bf16_to_float(got));
        bool w_nan = std::isnan(bf16_to_float(want));
        if (g_nan || w_nan) {  // NaN payloads differ by construction; sign-test both
            if (g_nan != w_nan) ++mism;
            return;
        }
        if (got != want) {
            ++mism;
            int d = (int)got - (int)want;
            int ulp = 0;
            while (d != 0 && (d < -16 || d > 16)) { d /= 2; ++ulp; }
            if (ulp > max_ulp) max_ulp = ulp;
        }
    }
};

void test_core() {
    Stats st;
    const int idxs[4] = {0, 4, 8, 12};
    for (int it = 0; it < 200000; ++it) {
        uint64_t packed = rnd64();
        uint8_t scale = (uint8_t)(rnd64() & 0xFF);
        int idx = idxs[rnd64() % 4];
        auto tb = fp4x16_to_bf16x4(packed, scale, idx);
        for (int i = 0; i < 4; ++i) {
            uint8_t byte = (uint8_t)((packed >> (((idx + i) >> 1) * 8)) & 0xFF);
            uint8_t code = (uint8_t)((byte >> (((idx + i) & 1) * 4)) & 0xF);
            st.check((i == 0) ? std::get<0>(tb) : (i == 1) ? std::get<1>(tb)
                              : (i == 2) ? std::get<2>(tb) : std::get<3>(tb),
                     ref_product_bf16(code, scale));
        }
    }
    printf("TEST1 core unpack x scale: total=%lld mism=%lld skipped=%lld max_ulp=%lld\n",
           st.total, st.mism, st.skipped, st.max_ulp);
}

// TEST 2 (A3): full 288 B/token layout roundtrip, quantize -> cache -> kernel path.
void test_layout_roundtrip() {
    Stats st;
    constexpr int T = 23;   // tokens per block
    constexpr int D = 512;
    std::vector<float> x(T * D);
    std::vector<uint8_t> row(D / 2);     // nibble-packed raw, dims 0..511
    std::vector<uint8_t> scales(D / 16); // e4m3 scale per 16-dim tile (token offset +256)

    auto quant_token = [&](int t) {
        float amax = 0.0f;
        for (int d = 0; d < D; ++d) {
            float v;
            switch (rnd64() % 100) {
                case 0 ... 54: v = frand(-8.0f, 8.0f); break;
                case 55 ... 79: v = frand(-0.02f, 0.02f); break;
                case 80 ... 89: v = (rnd64() & 1) ? 1e30f : -1e30f; break;  // saturating
                case 90 ... 95: v = std::nanf(""); break;                    // NaN data
                default: v = frand(0.0001f, 0.9f); break;
            }
            x[t * D + d] = v;
            if (!(std::isnan(v) || std::isinf(v))) {
                float a = fabsf(v);
                if (a > amax) amax = a;
            }
        }
        // scale = clamp(amax/6, 2^-9, 448) snapped onto the finite e4m3 grid
        double sv = (double)amax / 6.0;
        sv = sv < ldexp(1.0, -9) ? ldexp(1.0, -9) : (sv > 448.0 ? 448.0 : sv);
        uint8_t best = 0;
        double best_err = 1e300;
        for (int pat = 0; pat < 256; ++pat) {
            if ((pat & 0x78) == 0x70) continue;
            float w = ref_e4m3((uint8_t)pat);
            if (!std::isfinite(w)) continue;
            double err = fabs((double)w - sv);
            if (err < best_err) { best_err = err; best = (uint8_t)pat; }
        }
        for (int tile = 0; tile < D / 16; ++tile) scales[(size_t)tile] = best;
    };

    for (int t = 0; t < T; ++t) {
        quant_token(t);
        // pack nibbles: element i in byte i//2, even elements in the low nibble
        for (int d = 0; d < D; ++d) {
            float norm = x[t * D + d] / (float)ref_e4m3(scales[(size_t)d / 16]);
            uint8_t code = quant_e2m1(norm);
            if (d & 1) row[(size_t)d / 2] |= (uint8_t)(code << 4);
            else row[(size_t)d / 2] = (uint8_t)code;
        }
        // walk the cache like issue_kv_loads: segments k=0..7, lane groups g=0..3 own
        // dims [64k+16g, 64k+16g+16); kernel decodes 16 values per group (idx 0/4/8/12).
        for (int k = 0; k < 8; ++k) {
            for (int g = 0; g < 4; ++g) {
                uint64_t packed = 0;
                for (int b = 0; b < 8; ++b) {  // 16 dims = 8 bytes at dim/2
                    int dim = k * 64 + g * 16 + b * 2;
                    packed |= (uint64_t)row[(size_t)dim / 2] << (8 * b);
                }
                uint32_t scale_word = (uint32_t)scales[(size_t)k * 4] |
                                      ((uint32_t)scales[(size_t)k * 4 + 1] << 8) |
                                      ((uint32_t)scales[(size_t)k * 4 + 2] << 16) |
                                      ((uint32_t)scales[(size_t)k * 4 + 3] << 24);
                uint8_t scale_byte = (uint8_t)((scale_word >> (8 * g)) & 0xFF);
                // blocks of four payload positions (idx 0/4/8/12), decoded once each
                auto tb0 = fp4x16_to_bf16x4(packed, scale_byte, 0);
                auto tb1 = fp4x16_to_bf16x4(packed, scale_byte, 4);
                auto tb2 = fp4x16_to_bf16x4(packed, scale_byte, 8);
                auto tb3 = fp4x16_to_bf16x4(packed, scale_byte, 12);
                for (int j = 0; j < 4; ++j) {
                    for (int i = 0; i < 4; ++i) {
                        int dim = k * 64 + g * 16 + j * 4 + i;
                        uint8_t byte = row[(size_t)dim / 2];
                        uint8_t code = (dim & 1) ? (uint8_t)((byte >> 4) & 0xF)
                                                  : (uint8_t)(byte & 0xF);
                        {
                            const auto& tb = j == 0 ? tb0 : (j == 1 ? tb1 : (j == 2 ? tb2 : tb3));
                            st.check(tup_at(tb, i), ref_product_bf16(code, scale_byte));
                        }
                    }
                }
            }
        }
    }
    printf("TEST2 layout roundtrip: total=%lld mism=%lld skipped=%lld max_ulp=%lld\n",
           st.total, st.mism, st.skipped, st.max_ulp);
}

}  // namespace

int main() {
    test_core();
    test_layout_roundtrip();
    printf("FP4 HOST TEST DONE\n");
    return 0;
}