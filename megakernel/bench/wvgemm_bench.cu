// wvgemm_bench: a flat int8 tensor-core verify GEMM ("wv": mma.m16n8k32, tokens on N) for the main-pass roles, against
// today's dp4a GEMV (the engine's own rows_* / accR code, copied verbatim into wv_ref.cuh by wv_extract.py).
//
//   python3 wv_extract.py && nvcc -O3 -std=c++17 -arch=sm_86 -lineinfo -Xptxas -v wvgemm_bench.cu -o wvgemm_bench
//   ./wvgemm_bench check            exactness: T-invariance (T = 1..16 bit-identical per row) + integer / fp64 checks
//   ./wvgemm_bench time [roles] [types] [reps]
//
// Weights are random bytes in the engine's on-device layouts (native Q4_K / Q5_K, the load-time repacks of IQ4_XS, IQ4_NL,
// Q8_0, Q6_K, Q3_K, IQ3_S), so both kernels read exactly the bytes they would read in mk_cycle. Activations are random
// q8 rows in the engine's unit layout (QBuf: per-32 scales like q32_store, or per-16 like q16_store for ffn_down).
// Both kernels run like a mk_cycle phase: 82 blocks x 512 threads (1 block / SM, 128 registers), the block fills the
// activations into shared memory, computes, and a grid barrier ends the phase; NLAY layer copies defeat L2.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#define CK(x)                                                                                                    \
    do {                                                                                                         \
        cudaError_t e_ = (x);                                                                                    \
        if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(1); } \
    } while (0)

constexpr int NT = 512, NW = 16;
#include "wv_ref.cuh"

// ------------------------------------------------------------------------------------------------------------------
// grid barrier (one cooperative launch per measurement; the counter is cleared before each launch)
__device__ unsigned g_bar;
__device__ __forceinline__ void gsync(unsigned & epoch) {
    __syncthreads();
    if (threadIdx.x == 0) {
        epoch += gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(&g_bar) : "memory");
        unsigned v;
        do { asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(&g_bar) : "memory"); } while (v < epoch);
    }
    __syncthreads();
}
__device__ __forceinline__ void cp16(void * dst, const void * src) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_wait() { asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory"); }

// Activations in the engine's unit layout (QBuf): unit u = (super-block b, 64-group p, half h) holds A = elements
// 256b + 64p + 16h + [0, 16) and B = 256b + 64p + 32 + 16h + [0, 16); meta per unit: half2 (scale of A / B), short2
// (int sums of A / B).
struct QB {
    const int8_t * xa;
    const int8_t * xb;
    const uint2 * meta;
    int U; // units per token row
};

// ==================================================================================================================
// today's GEMV: gemv_pairs (2-row warp items over all warps) with the engine's mrows<T, IT, 2, TYPE>
// ==================================================================================================================
struct DpArgs {
    const Mat * mats;
    int nlay;
    QB q;
    float * out;
    int M;
};
template <int T, int IT, int TYPE>
__global__ void __launch_bounds__(NT, 1) k_dp4a(DpArgs A) {
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int U = 32 * IT;
    SX sx;
    sx.xa = reinterpret_cast<int4 *>(sm);
    sx.xb = sx.xa + T * U;
    sx.xm = reinterpret_cast<uint2 *>(sx.xb + T * U);
    sx.U = U;
    const int lane = threadIdx.x & 31;
    unsigned epoch = 0;
    for (int L = 0; L < A.nlay; L++) {
        for (int i = threadIdx.x; i < T * U; i += NT) {
            cp16(sx.xa + i, A.q.xa + (size_t) i * 16);
            cp16(sx.xb + i, A.q.xb + (size_t) i * 16);
        }
        for (int i = threadIdx.x; i < T * U / 2; i += NT) cp16(sx.xm + 2 * i, A.q.meta + 2 * i);
        cp_wait();
        __syncthreads();
        const Mat m = A.mats[L];
        for (int it = gwarp(); it < A.M / 2; it += nwarps()) {
            float a[2][T];
            zero_acc<T, 2>(a);
            mrows<T, IT, 2, TYPE>(m, 2 * it, sx, a);
            wsum_acc<T, 2>(a);
#pragma unroll
            for (int t = 0; t < T; t++)
                if ((lane >> 1) == t) A.out[(size_t) t * A.M + 2 * it + (lane & 1)] = (lane & 1) ? a[1][t] : a[0][t];
        }
        gsync(epoch);
    }
}

// ==================================================================================================================
// wv: int8 mma.m16n8k32, a warp computes a 16-row tile (rows g, g + 8 per thread) x 8 tokens per MMA over a K range.
//
// K permutation (the same for A and B, so the integer dot is unchanged): MMA k-slot 4 tq + i <-> element 8 tq + i and
// k-slot 16 + 4 tq + i <-> element 8 tq + 4 + i of the 32-element block, i.e. each thread supplies 8 consecutive
// elements: one 8-byte load from the weight row and one 8-byte LDS of the activations.
// Integer -> float without I2F: the accumulator input is MAGIC = 0x4B400000 (12582912.0f), so the MMA returns
// MAGIC + S, whose float value is exactly 12582912 + S (|S| < 2^22); fmaf(dx, that, -12582912 dx) = dx * S rounded once
// (the product -12582912 dx is exact).
// Per-16 scales (Q6_K / Q3_K weights, or the q16 activations of ffn_down): "split" MMAs. Threads tq 0-1 hold chunk 0
// (elements 0-15), tq 2-3 chunk 1. B column g carries token (g & 3) of the group, and only the k-slots of chunk g >> 2
// (the other slots are zero), so C column n = the 16-element dot of token n & 3 over chunk n >> 2: thread tq owns
// chunk tq >> 1 of tokens 2 tq & 3, + 1. The two chunk sums of a token meet in one shuffle after the K loop.
// Offset types (Q6_K, Q3_K, IQ3_S feed q + 32 like the dp4a path): the accumulator input is MAGIC - 32 * qsum(chunk),
// so the MMA returns MAGIC + sum (q + 32 - 32) x = the signed dot, with no float min term.
// ==================================================================================================================
#ifndef WV_L2PF
#define WV_L2PF 0 // L2 prefetch distance in super-blocks (0: off)
#endif
constexpr int MAGIC = 0x4B400000;
constexpr float FMAGIC = 12582912.0f;

__device__ __forceinline__ void mma16832(int (&d)[4], const int (&a)[4], int b0, int b1, const int (&c)[4]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
// float(byte k of w) + 2^23 (k compile-time after unrolling)
__device__ __forceinline__ float mag23(unsigned w, int k) { return __int_as_float(__byte_perm(w, 0x4b000000u, 0x7540u | k)); }

// Per weight type: Raw = one super-block of one row as this thread loads it; Sc = its decoded per-super-block scale
// state; frag(s) = the two int8x4 words of elements 8 tq .. 8 tq + 7 of sub-block s; wsc(s, h) = the float scale of
// sub-block s (chunk h for per-16 types) and, for MIN == 1, the min coefficient (weight = fA q - gA).
// MIN: 0 none, 1 per-row min (Q4_K / Q5_K: one FFMA per output), 2 constant offset 32 (via the accumulator input).
template <int TYPE> struct WV;

// weight loads of the wv kernel: the engine's evict-first non-coherent loads, optionally with the L2::256B fetch-size
// hint (WV_LD256=1: an L2 miss fetches the aligned 256 bytes around the access, a prefetch of the row's next block)
#ifndef WV_LD256
#define WV_LD256 0
#endif
#if WV_LD256
__device__ __forceinline__ int4 wld4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int2 wld2(const void * p) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.v2.s32 {%0,%1}, [%2], %3;" : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int wld1(const void * p) {
    int r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.L2::256B.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(l2_evict_first()));
    return r;
}
#else
__device__ __forceinline__ int4 wld4(const void * p) { return ldw4(p); }
__device__ __forceinline__ int2 wld2(const void * p) { return ldw2(p); }
__device__ __forceinline__ int wld1(const void * p) { return ldw1(p); }
#endif
__device__ __forceinline__ void pf_l2n(const void * p) { asm volatile("prefetch.global.L2 [%0];" ::"l"(p)); }

template <> struct WV<T_Q4K> { // native 144 B: d, dmin | scales[12] | qs[128]
    static constexpr bool S16 = false;
    static constexpr int MIN = 1;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { unsigned s0, s1, m0, m1; float d, nd, dm, ndm; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = m.w + (size_t) row * m.rb + sb * 144;
        r.h = wld4(blk);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(blk + 16 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&r.h.x));
        const unsigned y = r.h.y, z = r.h.z, w = r.h.w;
        s.s0 = y & 0x3f3f3f3fu;
        s.m0 = z & 0x3f3f3f3fu;
        s.s1 = (w & 0x0f0f0f0fu) | ((y >> 2) & 0x30303030u);
        s.m1 = ((w >> 4) & 0x0f0f0f0fu) | ((z >> 2) & 0x30303030u);
        s.d = dm.x; s.nd = -8388608.0f * dm.x; s.dm = dm.y; s.ndm = -8388608.0f * dm.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        fA = fmaf(s.d, mag23(k < 4 ? s.s0 : s.s1, k & 3), s.nd);
        gA = fmaf(s.dm, mag23(k < 4 ? s.m0 : s.m1, k & 3), s.ndm);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = (r.q[k >> 1].x >> sh) & M4;
        w1 = (r.q[k >> 1].y >> sh) & M4;
    }
};

template <> struct WV<T_Q5K> { // native 176 B: d, dmin | scales[12] | qh[32] | qs[128]
    static constexpr bool S16 = false;
    static constexpr int MIN = 1;
    struct Raw { int4 h; int2 qh; int2 q[4]; };
    using Sc = WV<T_Q4K>::Sc;
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = m.w + (size_t) row * m.rb + sb * 176;
        r.h = wld4(blk);
        r.qh = wld2(blk + 16 + 8 * tq);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(blk + 48 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        WV<T_Q4K>::Raw t;
        t.h = r.h;
        WV<T_Q4K>::scales(t, s);
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int h, float & fA, float & gA) { WV<T_Q4K>::wsc(s, k, h, fA, gA); }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | ((((unsigned) r.qh.x >> k) & 0x01010101u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | ((((unsigned) r.qh.y >> k) & 0x01010101u) << 4);
    }
};

template <> struct WV<T_IQ4X> { // repacked pairs of blocks, 272 B: hdr0 | hdr1 (d | scales_h | scales_l[4]) | qs0 | qs1
    static constexpr bool S16 = false;
    static constexpr int MIN = 0;
    struct Raw { int2 h; int2 q[4]; };
    struct Sc { unsigned sl, sh; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * pr = m.w + (size_t) row * m.rb + (sb >> 1) * 272;
        r.h = wld2(pr + 8 * (sb & 1));
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(pr + 16 + 128 * (sb & 1) + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.d = __half2float(__ushort_as_half((unsigned short) (r.h.x & 0xffff)));
        s.nd = -FMAGIC * s.d;
        s.sh = (unsigned) r.h.x >> 16;
        s.sl = (unsigned) r.h.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        const int ls = (int) (((s.sl >> (4 * k)) & 0xfu) | (((s.sh >> (2 * k)) & 3u) << 4)); // k < 8: nibble k of sl
        fA = fmaf(s.d, __int_as_float(MAGIC - 32 + ls), s.nd);
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_IQ4NL> { // repacked 144 B: d[8] (fp16, one per 32) | qs[128] in Q4_K order
    static constexpr bool S16 = false;
    static constexpr int MIN = 0;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { int4 h; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = m.w + (size_t) row * m.rb + sb * 144;
        r.h = wld4(blk);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = wld2(blk + 16 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.h = r.h; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        const int w = k == 0 || k == 1 ? s.h.x : k == 2 || k == 3 ? s.h.y : k == 4 || k == 5 ? s.h.z : s.h.w;
        fA = __half2float(__ushort_as_half((unsigned short) ((unsigned) w >> (16 * (k & 1)))));
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_Q8_0> { // repacked: per unit 32 B (A 16 | B 16); fp16 d per natural 32-block in Mat::d
    static constexpr bool S16 = false;
    static constexpr int MIN = 0;
    struct Raw { int4 d; int2 q[8]; };
    struct Sc { int4 d; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * rp = m.w + (size_t) row * m.rb + sb * 256;
        r.d = wld4(m.d + (size_t) row * m.nd + 8 * sb);
#pragma unroll
        for (int k = 0; k < 8; k++) r.q[k] = wld2(rp + 32 * (2 * (k >> 1) + (tq >> 1)) + 16 * (k & 1) + 8 * (tq & 1));
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.d = r.d; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        WV<T_IQ4NL>::Sc t;
        t.h = s.d;
        WV<T_IQ4NL>::wsc(t, k, 0, fA, gA);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int, int & w0, int & w1) { w0 = r.q[k].x; w1 = r.q[k].y; }
};

// Q6_K repack: values 192 B per block: qlo[128] (Q4_K-order nibbles) | qhi[64] (8 B per unit u' = 2p + h: A's highs | B's,
// byte m bits 2j = element 4j + m of the 16-chunk); meta (Mat::d) 18 B per block, in pairs: sc[16] even | sc[16] odd |
// d even | d odd. Values are q + 32 in [0, 63] (offset 32).
template <> struct WV<T_Q6K> {
    static constexpr bool S16 = true;
    static constexpr int MIN = 2;
    struct Raw { int sc[4]; int dd; int2 q[4]; int2 hb[4]; };
    struct Sc { int sc[4]; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = m.w + (size_t) row * m.rb + sb * Q6_BLK;
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + (sb >> 1) * (2 * Q6_META);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = wld1(mp + 16 * (sb & 1) + 4 * k); // (meta is only 4-byte aligned)
        r.dd = wld1(mp + 32);
        r.dd = (int) ((unsigned) r.dd >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            r.q[p] = wld2(blk + 32 * p + 8 * tq);
            r.hb[p] = wld2(blk + 128 + 8 * (2 * p + (tq >> 1)));
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -FMAGIC * s.d;
    }
    // chunk c = 4p + 2 part + h of the block: byte c of sc[16] (h = tq >> 1 is not compile-time: select the word pair)
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int h, float & fA, float & gA) {
        const int c0 = 4 * (k >> 1) + 2 * (k & 1); // even: bytes c0, c0 + 1 lie in word c0 >> 2
        const unsigned w = (unsigned) s.sc[c0 >> 2] >> (8 * h);
        const int v = (int) (signed char) (w >> (8 * (c0 & 3)));
        fA = fmaf(s.d, __int_as_float(MAGIC + v), s.nd);
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int tq, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        const unsigned hs = (k & 1) ? (unsigned) r.hb[k >> 1].y : (unsigned) r.hb[k >> 1].x;
        const int j0 = 2 * (tq & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | (int) (((hs >> (2 * j0)) & 0x03030303u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | (int) (((hs >> (2 * j0 + 2)) & 0x03030303u) << 4);
    }
};

// Q3_K repack: values 96 B per block: unit u' = 2p + h: 2-bit lows at 8u' (A word | B word; byte m bits 2j = element 4j + m),
// highs at 64 + 4u' (byte m: bit j = A element 4j + m, bit 4 + j = B); meta as Q6_K (sc - 32 as int8, d). Value fed:
// u3 + 28 = the Q6_K value (offset 32).
template <> struct WV<T_Q3K> {
    static constexpr bool S16 = true;
    static constexpr int MIN = 2;
    struct Raw { int sc[4]; int dd; int2 lo[4]; int hi[4]; };
    using Sc = WV<T_Q6K>::Sc;
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = m.w + (size_t) row * m.rb + sb * Q3_BLK;
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + (sb >> 1) * (2 * Q6_META);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = wld1(mp + 16 * (sb & 1) + 4 * k);
        r.dd = (int) ((unsigned) wld1(mp + 32) >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.lo[p] = wld2(blk + 8 * u);
            r.hi[p] = wld1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int h, float & fA, float & gA) { WV<T_Q6K>::wsc(s, k, h, fA, gA); }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int tq, int & w0, int & w1) {
        const unsigned x = (k & 1) ? (unsigned) r.lo[k >> 1].y : (unsigned) r.lo[k >> 1].x, hh = (unsigned) r.hi[k >> 1];
        const int j0 = 2 * (tq & 1), hs = 4 * (k & 1);
        w0 = (int) ((((x >> (2 * j0)) & 0x03030303u) | (((hh >> (j0 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
        w1 = (int) ((((x >> (2 * j0 + 2)) & 0x03030303u) | (((hh >> (j0 + 1 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
    }
};

// IQ3_S repack: values 96 B per block: unit u' = 2p + h: the low bytes of A's 4 grid indices then B's at 8u', signs at
// 64 + 4u' (byte m: bit c = A element 4c + m, bit 4 + c = B); meta 16 B per block: qh[8] (byte u': index bit 8, A in
// bits 0-3, B in 4-7) | scales[4] (byte p: sub-block 2p | 2p + 1 << 4) | fp16 d | pad. Value fed: 32 +- grid (offset 32).
template <> struct WV<T_IQ3S> {
    static constexpr bool S16 = false;
    static constexpr int MIN = 2;
    struct Raw { int2 qh; int2 sd; int2 ix[4]; int sg[4]; };
    struct Sc { unsigned sc; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int row, int sb, int tq) {
        const uint8_t * blk = m.w + (size_t) row * m.rb + sb * Q3_BLK;
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + sb * IQ3S_META;
        r.qh = wld2(mp);
        r.sd = wld2(mp + 8);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.ix[p] = wld2(blk + 8 * u);
            r.sg[p] = wld1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.sc = (unsigned) r.sd.x;
        s.d = __half2float(__ushort_as_half((unsigned short) (r.sd.y & 0xffff)));
        s.nd = -FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, int, float & fA, float & gA) {
        const int v = 1 + 2 * (int) ((s.sc >> (4 * k)) & 0xfu); // nibble k = sub-block k
        fA = fmaf(s.d, __int_as_float(MAGIC + v), s.nd);
        gA = 0.f;
    }
    // (the qh byte of unit u' is per thread: u' = 2p + (tq >> 1))
    static __device__ __forceinline__ void frag(const Raw & r, int k, int tq, int & w0, int & w1) {
        const int p = k >> 1, part = k & 1, u = 2 * p + (tq >> 1);
        const unsigned qhw = (unsigned) (u < 4 ? r.qh.x : r.qh.y) >> (8 * (u & 3));
        const unsigned x = part ? (unsigned) r.ix[p].y : (unsigned) r.ix[p].x, sv = (unsigned) r.sg[p];
        const int c0 = 2 * (tq & 1), hs = 4 * part;
        int w[2];
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const int c = c0 + i;
            const unsigned g = iq3s_grid(((x >> (8 * c)) & 0xffu) | (((qhw >> (c + hs)) & 1u) << 8));
            const unsigned mk = ((sv >> (c + hs)) & 0x01010101u) * 0xffu;
            w[i] = (int) (0x20202020u + (g ^ mk) - mk);
        }
        w0 = w[0];
        w1 = w[1];
    }
};

// shared-memory activations of the block's K chunk: xq[token][ld] int8 in natural element order (ld = Kc + 32, so the
// 8-byte B loads of a half-warp hit distinct banks); md[(j * NG + n) * 4 + tq] = {dx_a, -12582912 dx_a, dx_b, -..dx_b}
// for the two tokens (a, b) of the thread's C columns in group n and 32-block j (split mode: of its chunk); mx = the
// matching (sq_a, sq_b) = dx * qsum for MIN 1, or (MAGIC - 32 qsum) bits for MIN 2.
struct XSm {
    const int8_t * xq;
    int ld;
    const float4 * md;
    const float2 * mx;
};

template <int TYPE, int T, bool X16>
struct Cfg {
    static constexpr bool SPLIT = WV<TYPE>::S16 || X16;
    static constexpr int TPG = SPLIT ? 4 : 8;
    static constexpr int NG = (T + TPG - 1) / TPG;
    static constexpr int NTOKP = NG * TPG;
};

// one super-block (8 sub-blocks) of the tile from the thread's raw rows r0 (row g) and r1 (row g + 8); j0 = 8 * the
// super-block's index within the shared-memory chunk
template <int TYPE, int T, bool X16>
__device__ __forceinline__ void wv_sb(const typename WV<TYPE>::Raw & r0, const typename WV<TYPE>::Raw & r1, int j0,
                                      const XSm & xs, const int8_t * xrow, bool bon, float (&acc)[Cfg<TYPE, T, X16>::NG][4]) {
    using W = WV<TYPE>;
    using C = Cfg<TYPE, T, X16>;
    const int tq = threadIdx.x & 3, h = tq >> 1;
    typename W::Sc s0, s1;
    W::scales(r0, s0);
    W::scales(r1, s1);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int a[4];
        W::frag(r0, k, tq, a[0], a[2]);
        W::frag(r1, k, tq, a[1], a[3]);
        float fA0, gA0, fA1, gA1;
        W::wsc(s0, k, h, fA0, gA0);
        W::wsc(s1, k, h, fA1, gA1);
        const int j = j0 + k;
#pragma unroll
        for (int n = 0; n < C::NG; n++) {
            int2 bb = make_int2(0, 0);
            if (bon) bb = *reinterpret_cast<const int2 *>(xrow + C::TPG * n * xs.ld + 32 * j);
            const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
            int ci[4] = {MAGIC, MAGIC, MAGIC, MAGIC};
            float2 mx = make_float2(0.f, 0.f);
            if (W::MIN) mx = xs.mx[(j * C::NG + n) * 4 + tq];
            if (W::MIN == 2) {
                ci[0] = ci[2] = __float_as_int(mx.x);
                ci[1] = ci[3] = __float_as_int(mx.y);
            }
            int cc[4];
            mma16832(cc, a, bb.x, bb.y, ci);
            acc[n][0] = fmaf(fA0, fmaf(md.x, __int_as_float(cc[0]), md.y), acc[n][0]);
            acc[n][1] = fmaf(fA0, fmaf(md.z, __int_as_float(cc[1]), md.w), acc[n][1]);
            acc[n][2] = fmaf(fA1, fmaf(md.x, __int_as_float(cc[2]), md.y), acc[n][2]);
            acc[n][3] = fmaf(fA1, fmaf(md.z, __int_as_float(cc[3]), md.w), acc[n][3]);
            if (W::MIN == 1) {
                acc[n][0] = fmaf(-gA0, mx.x, acc[n][0]);
                acc[n][1] = fmaf(-gA0, mx.y, acc[n][1]);
                acc[n][2] = fmaf(-gA1, mx.x, acc[n][2]);
                acc[n][3] = fmaf(-gA1, mx.y, acc[n][3]);
            }
        }
    }
}

#ifndef WV_SBU
#define WV_SBU 1 // super-blocks loaded per batch (PF = 0 only)
#endif
template <int TYPE, int T, bool X16, int PF>
__device__ __forceinline__ void wv_tile(const Mat & m, int row0, int sb0, int nsb, const XSm & xs,
                                        float (&acc)[Cfg<TYPE, T, X16>::NG][4]) {
    using W = WV<TYPE>;
    using C = Cfg<TYPE, T, X16>;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, h = tq >> 1;
#pragma unroll
    for (int n = 0; n < C::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc[n][c] = 0.f;
    const int8_t * xrow = C::SPLIT ? xs.xq + (g & 3) * xs.ld + 8 * tq : xs.xq + g * xs.ld + 8 * tq;
    const bool bon = !C::SPLIT || (g >> 2) == h;
    if (!PF && WV_SBU == 2) {
        typename W::Raw a0, a1, b0, b1;
        int sb = 0;
#pragma unroll 1
        for (; sb + 1 < nsb; sb += 2) {
            W::load(a0, m, row0 + g, sb0 + sb, tq);
            W::load(a1, m, row0 + g + 8, sb0 + sb, tq);
            W::load(b0, m, row0 + g, sb0 + sb + 1, tq);
            W::load(b1, m, row0 + g + 8, sb0 + sb + 1, tq);
            wv_sb<TYPE, T, X16>(a0, a1, 8 * sb, xs, xrow, bon, acc);
            wv_sb<TYPE, T, X16>(b0, b1, 8 * sb + 8, xs, xrow, bon, acc);
        }
        if (sb < nsb) {
            W::load(a0, m, row0 + g, sb0 + sb, tq);
            W::load(a1, m, row0 + g + 8, sb0 + sb, tq);
            wv_sb<TYPE, T, X16>(a0, a1, 8 * sb, xs, xrow, bon, acc);
        }
    } else {
    typename W::Raw r0, r1, n0, n1;
    if (PF) {
        W::load(r0, m, row0 + g, sb0, tq);
        W::load(r1, m, row0 + g + 8, sb0, tq);
    }
#if WV_L2PF
    // L2 prefetch of the main weight stream WV_L2PF super-blocks ahead: lane l -> row l & 15, 128-byte line l >> 4
    const int bps = m.rb / (m.K / 256);
    const uint8_t * pfrow = m.w + (size_t) (row0 + (lane & 15)) * m.rb + 128 * (lane >> 4);
#pragma unroll
    for (int d = 0; d < WV_L2PF; d++)
        if (d < nsb && 128 * (lane >> 4) < bps) pf_l2n(pfrow + (size_t) (sb0 + d) * bps);
#endif
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
#if WV_L2PF
        if (sb + WV_L2PF < nsb && 128 * (lane >> 4) < bps) pf_l2n(pfrow + (size_t) (sb0 + sb + WV_L2PF) * bps);
#endif
        if (PF) {
            if (sb + 1 < nsb) {
                W::load(n0, m, row0 + g, sb0 + sb + 1, tq);
                W::load(n1, m, row0 + g + 8, sb0 + sb + 1, tq);
            }
        } else {
            W::load(r0, m, row0 + g, sb0 + sb, tq);
            W::load(r1, m, row0 + g + 8, sb0 + sb, tq);
        }
        wv_sb<TYPE, T, X16>(r0, r1, 8 * sb, xs, xrow, bon, acc);
        if (PF) {
            r0 = n0;
            r1 = n1;
        }
    }
    }
    if (C::SPLIT) { // chunk 0 (tq 0-1) + chunk 1 (tq 2-3) of the same tokens
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[n][c] += __shfl_xor_sync(0xffffffffu, acc[n][c], 2);
    }
}

// Fill the block's K chunk [k0, k0 + Kc) of T activation rows: values to xq, scales to md / mx.
template <int TYPE, int T, bool X16>
__device__ void wv_fill(const QB & q, int k0, int Kc, uint8_t * sm, XSm & xs) {
    using C = Cfg<TYPE, T, X16>;
    const int ld = Kc + 32, NJ = Kc / 32;
    int8_t * xq = reinterpret_cast<int8_t *>(sm);
    float4 * md = reinterpret_cast<float4 *>(sm + C::NTOKP * ld);
    float2 * mx = reinterpret_cast<float2 *>(md + NJ * C::NG * 4);
    const int u0 = k0 / 32, nu = Kc / 32;
    for (int i = threadIdx.x; i < T * nu * 2; i += NT) {
        const int t = i / (2 * nu), r = i - t * 2 * nu, u = u0 + (r >> 1), part = r & 1;
        const int b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        const int e = 256 * b + 64 * p + 32 * part + 16 * hh - k0;
        cp16(xq + t * ld + e, (part ? q.xb : q.xa) + ((size_t) t * q.U + u) * 16);
    }
    for (int i = threadIdx.x; i < NJ * C::NG * 4; i += NT) {
        const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
        const int jg = u0 + j, b = jg >> 3, s = jg & 7, p = s >> 1, part = s & 1;
        float v[4];
        float2 x2;
        int qs2[2];
        for (int k = 0; k < 2; k++) {
            int t, hh = -1;
            if (C::SPLIT) {
                t = 4 * n + ((2 * tq) & 3) + k;
                hh = tq >> 1;
            } else {
                t = 8 * n + 2 * tq + k;
            }
            float dx = 0.f;
            int qs = 0;
            if (t < T) {
                const uint2 m0 = q.meta[(size_t) t * q.U + 8 * b + 2 * p + (hh < 0 ? 0 : hh)];
                const uint2 m1 = q.meta[(size_t) t * q.U + 8 * b + 2 * p + 1];
                dx = __half2float(reinterpret_cast<const __half *>(&m0.x)[part]);
                const int q0 = reinterpret_cast<const short *>(&m0.y)[part], q1 = reinterpret_cast<const short *>(&m1.y)[part];
                qs = hh < 0 ? q0 + q1 : q0;
            }
            v[2 * k] = dx;
            v[2 * k + 1] = -FMAGIC * dx;
            qs2[k] = qs;
            (k ? x2.y : x2.x) = WV<TYPE>::MIN == 2 ? __int_as_float(MAGIC - 32 * qs) : dx * (float) qs;
        }
        md[i] = make_float4(v[0], v[1], v[2], v[3]);
        mx[i] = x2;
    }
    cp_wait();
    __syncthreads();
    xs.xq = xq;
    xs.ld = ld;
    xs.md = md;
    xs.mx = mx;
}
template <int TYPE, int T, bool X16>
constexpr int wv_smem(int Kc) {
    return Cfg<TYPE, T, X16>::NTOKP * (Kc + 32) + (Kc / 32) * Cfg<TYPE, T, X16>::NG * 4 * 24;
}

struct WvArgs {
    const Mat * mats;
    int nlay;
    QB q;
    float * out;
    float * part; // [S][T][M] split-K partial sums
    unsigned * cnt; // [nlay][M / 16] arrival counters (count mod S)
    int M, K, S;
};
// Block b < S * G (G = gridDim / S) takes K chunk b / G and the tiles [ntile * bi / G, ntile * (bi + 1) / G) of its
// group index bi; its warps take those tiles round-robin. S > 1: each chunk writes its partial sums, the last of the S
// to arrive at a tile adds them in chunk order 0 .. S - 1 (fixed, so the result does not depend on who arrives last).
template <int TYPE, int T, bool X16, int PF>
__global__ void __launch_bounds__(NT, 1) k_wv(WvArgs A) {
    extern __shared__ __align__(16) uint8_t sm[];
    using C = Cfg<TYPE, T, X16>;
    const int S = A.S, G = gridDim.x / S, b = blockIdx.x;
    const bool active = b < S * G;
    const int kc = b / G, bi = b % G, NSB = A.K / 256;
    const int sb0 = NSB * kc / S, sb1 = NSB * (kc + 1) / S;
    const int ntile = A.M / 16, t0 = ntile * bi / G, t1 = ntile * (bi + 1) / G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    unsigned epoch = 0;
    for (int L = 0; L < A.nlay; L++) {
        XSm xs;
        if (active) wv_fill<TYPE, T, X16>(A.q, 256 * sb0, 256 * (sb1 - sb0), sm, xs);
        const Mat m = A.mats[L];
        if (active)
            for (int tile = t0 + warp; tile < t1; tile += NW) {
                float acc[C::NG][4];
                wv_tile<TYPE, T, X16, PF>(m, 16 * tile, sb0, sb1 - sb0, xs, acc);
                const bool st = !C::SPLIT || tq < 2;
                float * dst = S == 1 ? A.out : A.part + (size_t) kc * T * A.M;
#pragma unroll
                for (int n = 0; n < C::NG; n++)
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const int t = C::SPLIT ? 4 * n + 2 * tq + (c & 1) : 8 * n + 2 * tq + (c & 1);
                        if (st && t < T) dst[(size_t) t * A.M + 16 * tile + g + 8 * (c >> 1)] = acc[n][c];
                    }
                if (S > 1) {
                    __threadfence();
                    __syncwarp();
                    unsigned old = 0;
                    if (lane == 0) old = atomicAdd(A.cnt + (size_t) L * ntile + tile, 1u);
                    old = __shfl_sync(0xffffffffu, old, 0);
                    if (old % S == S - 1) {
                        __threadfence();
                        for (int i = lane; i < 16 * T; i += 32) {
                            const int t = i / 16, r = 16 * tile + (i & 15);
                            float s = 0.f;
                            for (int k = 0; k < S; k++) s += __ldcg(A.part + ((size_t) k * T + t) * A.M + r);
                            A.out[(size_t) t * A.M + r] = s;
                        }
                    }
                }
            }
        gsync(epoch);
    }
}

// ==================================================================================================================
// scalar reference decode of one weight in the engine layout: w = fA * v - gA, v = the integer operand both kernels feed
// ==================================================================================================================
__constant__ signed char c_kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__device__ float hf(const uint8_t * p) { return __half2float(*reinterpret_cast<const __half *>(p)); }
__device__ void ref_w(const Mat & m, int row, int e, int & v, double & fA, double & gA) {
    const int b = e >> 8, w = e & 255, p = w >> 6, part = (w >> 5) & 1, l = w & 31, hh = l >> 4, i = l & 15;
    const int sbk = 2 * p + part, c = 4 * p + 2 * part + hh; // sub-block (32), chunk (16)
    const uint8_t * r = m.w + (size_t) row * m.rb;
    if (m.type == T_Q4K || m.type == T_Q5K) {
        const uint8_t * blk = r + b * (m.type == T_Q4K ? 144 : 176);
        const uint8_t * sc = blk + 4;
        int s, mm;
        if (sbk < 4) { s = sc[sbk] & 63; mm = sc[sbk + 4] & 63; }
        else { s = (sc[sbk + 4] & 0xF) | ((sc[sbk - 4] >> 6) << 4); mm = (sc[sbk + 4] >> 4) | ((sc[sbk] >> 6) << 4); }
        const uint8_t qb = blk[(m.type == T_Q4K ? 16 : 48) + 32 * p + l];
        v = part ? qb >> 4 : qb & 15;
        if (m.type == T_Q5K) v |= ((blk[16 + l] >> sbk) & 1) << 4;
        fA = (double) hf(blk) * s;
        gA = (double) hf(blk + 2) * mm;
    } else if (m.type == T_IQ4X) {
        const uint8_t * pr = r + (b >> 1) * 272, * hd = pr + 8 * (b & 1);
        const int sh = hd[2] | (hd[3] << 8), sl = hd[4 + p];
        const int ls = ((sl >> (4 * part)) & 15) | (((sh >> (2 * sbk)) & 3) << 4);
        const uint8_t qb = pr[16 + 128 * (b & 1) + 32 * p + l];
        v = c_kv[part ? qb >> 4 : qb & 15];
        fA = (double) hf(hd) * (ls - 32);
        gA = 0;
    } else if (m.type == T_IQ4NL) {
        const uint8_t * blk = r + b * 144;
        const uint8_t qb = blk[16 + 32 * p + l];
        v = c_kv[part ? qb >> 4 : qb & 15];
        fA = hf(blk + 2 * sbk);
        gA = 0;
    } else if (m.type == T_Q8_0) {
        const int u = 8 * b + 2 * p + hh;
        v = (int8_t) r[32 * u + 16 * part + i];
        fA = __half2float(m.d[(size_t) row * m.nd + (e >> 5)]);
        gA = 0;
    } else if (m.type == T_Q6K || m.type == T_Q3K) {
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + (b >> 1) * 36;
        const int sc = (int8_t) mp[16 * (b & 1) + c];
        const double d = hf(mp + 32 + 2 * (b & 1));
        const int u = 2 * p + hh, jj = i >> 2, mb = i & 3;
        if (m.type == T_Q6K) {
            const uint8_t * blk = r + b * 192;
            const uint8_t qb = blk[32 * p + 16 * hh + i];
            const int lo = part ? qb >> 4 : qb & 15, hi = (blk[128 + 8 * u + 4 * part + mb] >> (2 * jj)) & 3;
            v = lo | (hi << 4);
        } else {
            const uint8_t * blk = r + b * 96;
            const int lo = (blk[8 * u + 4 * part + mb] >> (2 * jj)) & 3, hi = (blk[64 + 4 * u + mb] >> (jj + 4 * part)) & 1;
            v = (lo | (hi << 2)) + 28;
        }
        fA = d * sc;
        gA = 32 * fA;
    } else { // IQ3_S
        const uint8_t * blk = r + b * 96;
        const uint8_t * mp = reinterpret_cast<const uint8_t *>(m.d + (size_t) row * m.nd) + b * 16;
        const int u = 2 * p + hh, cc = i >> 2, mb = i & 3;
        const int idx = blk[8 * u + 4 * part + cc] | (((mp[u] >> (cc + 4 * part)) & 1) << 8);
        const int mag = (g_iq3s_grid[idx] >> (8 * mb)) & 0xff;
        const int sgn = (blk[64 + 4 * u + mb] >> (cc + 4 * part)) & 1;
        v = 32 + (sgn ? -mag : mag);
        const int s4 = (mp[8 + p] >> (4 * part)) & 15;
        fA = (double) hf(mp + 12) * (1 + 2 * s4);
        gA = 32 * fA;
    }
}
// activation element e of token t: integer value, its scale, and the scale-group index (per 16 or per 32)
__device__ void ref_x(const QB & q, int t, int e, int & x, double & dx) {
    const int b = e >> 8, w = e & 255, p = w >> 6, part = (w >> 5) & 1, hh = (w >> 4) & 1, i = w & 15;
    const int u = 8 * b + 2 * p + hh;
    x = (part ? q.xb : q.xa)[((size_t) t * q.U + u) * 16 + i];
    dx = __half2float(reinterpret_cast<const __half *>(&q.meta[(size_t) t * q.U + u].x)[part]);
}
// fp64 reference: out[t][row] = sum_e w_e x_e dx_e; also the integer dot per (row, t) of the operands (sum v x) and
// of the weights' offsets removed (for offset types sum (v - 32) x)
__global__ void k_ref(Mat m, QB q, int T, int K, double * out, long long * idot) {
    const int row = blockIdx.x, t = threadIdx.x / 32, lane = threadIdx.x & 31;
    if (t >= T) return;
    double s = 0;
    long long id = 0;
    for (int e = lane; e < K; e += 32) {
        int v, x;
        double fA, gA, dx;
        ref_w(m, row, e, v, fA, gA);
        ref_x(q, t, e, x, dx);
        s += (fA * v - gA) * (dx * x);
        id += (long long) v * x;
    }
    for (int o = 16; o > 0; o >>= 1) {
        s += __shfl_xor_sync(0xffffffffu, s, o);
        id += __shfl_xor_sync(0xffffffffu, id, o);
    }
    if (lane == 0) {
        out[(size_t) t * m.N + row] = s;
        idot[(size_t) t * m.N + row] = id;
    }
}

// ==================================================================================================================
// host
// ==================================================================================================================
static const char * tname(int ty) {
    switch (ty) {
        case T_Q4K: return "Q4_K";
        case T_Q5K: return "Q5_K";
        case T_Q6K: return "Q6_K";
        case T_Q8_0: return "Q8_0";
        case T_IQ4X: return "IQ4_XS";
        case T_IQ4NL: return "IQ4_NL";
        case T_Q3K: return "Q3_K";
        case T_IQ3S: return "IQ3_S";
    }
    return "?";
}
static int row_bytes(int ty, int K) {
    const int nb = K / 256;
    switch (ty) {
        case T_Q4K: return nb * 144;
        case T_Q5K: return nb * 176;
        case T_Q6K: return nb * Q6_BLK;
        case T_Q8_0: return K;
        case T_IQ4X: return nb * 136;
        case T_IQ4NL: return nb * 144;
        default: return nb * Q3_BLK; // Q3_K, IQ3_S
    }
}
static int row_meta_halves(int ty, int K) {
    const int nb = K / 256;
    switch (ty) {
        case T_Q6K: case T_Q3K: return nb * (Q6_META / 2);
        case T_IQ3S: return nb * (IQ3S_META / 2);
        case T_Q8_0: return K / 32;
        default: return 0;
    }
}
static uint16_t h16(float f) {
    __half h = __float2half(f);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
// random weights in the engine layout; fp16 scale fields set to finite positive values
static void make_weights(int ty, int M, int K, std::vector<uint8_t> & w, std::vector<uint16_t> & d, std::mt19937 & rng) {
    const int rb = row_bytes(ty, K), nd = row_meta_halves(ty, K), nb = K / 256;
    w.resize((size_t) M * rb);
    d.assign((size_t) M * nd, 0);
    for (auto & x : w) x = (uint8_t) rng();
    std::uniform_real_distribution<float> U(0.5f, 1.0f);
    auto dv = [&]() { return h16(U(rng) * (1.0f / 1024)); };
    for (int r = 0; r < M; r++) {
        uint8_t * row = w.data() + (size_t) r * rb;
        uint16_t * dr = d.data() + (size_t) r * nd;
        for (int b = 0; b < nb; b++) {
            if (ty == T_Q4K || ty == T_Q5K) {
                uint16_t a = dv(), c = dv();
                memcpy(row + b * (ty == T_Q4K ? 144 : 176), &a, 2);
                memcpy(row + b * (ty == T_Q4K ? 144 : 176) + 2, &c, 2);
            } else if (ty == T_IQ4X) {
                uint16_t a = dv();
                memcpy(row + (b >> 1) * 272 + 8 * (b & 1), &a, 2);
            } else if (ty == T_IQ4NL) {
                for (int k = 0; k < 8; k++) { uint16_t a = dv(); memcpy(row + b * 144 + 2 * k, &a, 2); }
            } else if (ty == T_Q6K || ty == T_Q3K) {
                uint8_t * mp = reinterpret_cast<uint8_t *>(dr) + (b >> 1) * 36;
                for (int k = 0; k < 16; k++) mp[16 * (b & 1) + k] = (uint8_t) rng();
                uint16_t a = dv();
                memcpy(mp + 32 + 2 * (b & 1), &a, 2);
            } else if (ty == T_IQ3S) {
                uint8_t * mp = reinterpret_cast<uint8_t *>(dr) + b * 16;
                for (int k = 0; k < 12; k++) mp[k] = (uint8_t) rng();
                uint16_t a = dv();
                memcpy(mp + 12, &a, 2);
                mp[14] = mp[15] = 0;
            }
        }
        if (ty == T_Q8_0)
            for (int k = 0; k < nd; k++) dr[k] = dv();
    }
}
// random q8 activations (NTOK rows) in the unit layout; per-32 (q32) or per-16 (q16) scales
static void make_acts(int K, int ntok, bool q16, std::vector<int8_t> & xa, std::vector<int8_t> & xb,
                      std::vector<uint2> & meta, std::mt19937 & rng) {
    const int U = K / 32;
    xa.resize((size_t) ntok * U * 16);
    xb.resize(xa.size());
    meta.resize((size_t) ntok * U);
    std::uniform_int_distribution<int> Q(-127, 127);
    std::uniform_real_distribution<float> S(0.5f, 1.0f);
    for (int t = 0; t < ntok; t++)
        for (int j = 0; j < K / 32; j++) { // natural block j: units (b, p, 0 / 1), part j & 1
            const int b = j >> 3, p = (j >> 1) & 3, part = j & 1;
            const float d32 = S(rng) / 64;
            for (int hh = 0; hh < 2; hh++) {
                const int u = 8 * b + 2 * p + hh;
                int8_t * dst = (part ? xb.data() : xa.data()) + ((size_t) t * U + u) * 16;
                int s = 0;
                for (int i = 0; i < 16; i++) { dst[i] = (int8_t) Q(rng); s += dst[i]; }
                uint2 & m = meta[(size_t) t * U + u];
                const uint16_t dh = h16(q16 ? S(rng) / 64 : d32);
                reinterpret_cast<uint16_t *>(&m.x)[part] = dh;
                reinterpret_cast<short *>(&m.y)[part] = (short) s;
            }
        }
}

template <class F> static float timeit(F f, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    f();
    CK(cudaDeviceSynchronize());
    float best = 1e30f;
    for (int i = 0; i < reps; i++) {
        cudaEventRecord(a);
        f();
        cudaEventRecord(b);
        CK(cudaEventSynchronize(b));
        float ms;
        cudaEventElapsedTime(&ms, a, b);
        best = ms < best ? ms : best;
    }
    return best;
}

static int g_nsm = 82;
static void * g_bar_ptr = nullptr;
template <class K> static void prep(K k) { CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, 99 * 1024)); }
template <class K, class Args> static void coop(K k, int smem, Args & a) {
    CK(cudaMemsetAsync(g_bar_ptr, 0, 4));
    void * args[] = {&a};
    CK(cudaLaunchCooperativeKernel((void *) k, dim3(g_nsm), dim3(NT), args, smem, 0));
}

// roles: rows M, K, dp4a IT (K / 1024), q16 activations, split-K S (fixed per role, for every T)
struct Role {
    const char * name;
    int M, K, IT;
    bool q16;
    int S;
};
static Role ROLES[] = {
    {"ffn_gu", 2 * 17408, 5120, 5, false, 1},   // gate + up rows
    {"dn_proj", 10240 + 6144, 5120, 5, false, 1}, // qkv + z
    {"attn_proj", 12288 + 2048, 5120, 5, false, 1}, // q (+gate) + k + v
    {"out", 5120, 6144, 6, false, 4},            // attn_output / ssm_out
    {"ffn_down", 5120, 17408, 17, true, 4},
};

// ---- dispatch tables ------------------------------------------------------------------------------------------------
template <int T, int TYPE>
static float run_dp4a(int IT, DpArgs & a, int reps) {
    auto go = [&](auto kfun, int U) {
        const int smem = T * U * 40;
        if (smem > 99 * 1024) return -1.f;
        prep(kfun);
        return timeit([&] { coop(kfun, smem, a); }, reps);
    };
    if (IT == 5) return go(k_dp4a<T, 5, TYPE>, 160);
    if (IT == 6) return go(k_dp4a<T, 6, TYPE>, 192);
    return go(k_dp4a<T, 17, TYPE>, 544);
}
template <int TYPE>
static float dp4a_T(int T, int IT, DpArgs & a, int reps) {
    switch (T) {
        case 1: return run_dp4a<1, TYPE>(IT, a, reps);
        case 2: return run_dp4a<2, TYPE>(IT, a, reps);
        case 3: return run_dp4a<3, TYPE>(IT, a, reps);
        case 4: return run_dp4a<4, TYPE>(IT, a, reps);
        case 5: return run_dp4a<5, TYPE>(IT, a, reps);
    }
    return -1.f;
}
template <int TYPE, int T, int PF>
static float run_wv(bool q16, WvArgs & a, int reps) {
    const int NSB = a.K / 256, Kc = 256 * ((NSB + a.S - 1) / a.S);
    auto go = [&](auto kfun, int smem) {
        if (smem > 99 * 1024) return -1.f;
        prep(kfun);
        return timeit([&] { coop(kfun, smem, a); }, reps);
    };
    if (q16) return go(k_wv<TYPE, T, true, PF>, wv_smem<TYPE, T, true>(Kc));
    return go(k_wv<TYPE, T, false, PF>, wv_smem<TYPE, T, false>(Kc));
}
template <int TYPE, int PF>
static float wv_T(int T, bool q16, WvArgs & a, int reps) {
#ifdef WV_QUICK // experiment builds: T = 1, 4, 8 only
    switch (T) {
        case 1: return run_wv<TYPE, 1, PF>(q16, a, reps);
        case 4: return run_wv<TYPE, 4, PF>(q16, a, reps);
        case 8: return run_wv<TYPE, 8, PF>(q16, a, reps);
    }
    return -1.f;
#else
    switch (T) {
        case 1: return run_wv<TYPE, 1, PF>(q16, a, reps);
        case 2: return run_wv<TYPE, 2, PF>(q16, a, reps);
        case 3: return run_wv<TYPE, 3, PF>(q16, a, reps);
        case 4: return run_wv<TYPE, 4, PF>(q16, a, reps);
        case 5: return run_wv<TYPE, 5, PF>(q16, a, reps);
        case 6: return run_wv<TYPE, 6, PF>(q16, a, reps);
        case 7: return run_wv<TYPE, 7, PF>(q16, a, reps);
        case 8: return run_wv<TYPE, 8, PF>(q16, a, reps);
        case 16: return run_wv<TYPE, 16, PF>(q16, a, reps);
    }
    return -1.f;
#endif
}
#ifndef WV_PF
#define WV_PF 0
#endif
static float dp4a_any(int ty, int T, int IT, DpArgs & a, int reps) {
#ifdef WV_QUICK
    if (T != 1) return -1.f;
    switch (ty) {
        case T_Q4K: return dp4a_T<T_Q4K>(T, IT, a, reps);
        case T_Q5K: return dp4a_T<T_Q5K>(T, IT, a, reps);
        case T_IQ4X: return dp4a_T<T_IQ4X>(T, IT, a, reps);
    }
    return -1.f;
#else
    switch (ty) {
        case T_Q4K: return dp4a_T<T_Q4K>(T, IT, a, reps);
        case T_Q5K: return dp4a_T<T_Q5K>(T, IT, a, reps);
        case T_Q6K: return dp4a_T<T_Q6K>(T, IT, a, reps);
        case T_Q8_0: return dp4a_T<T_Q8_0>(T, IT, a, reps);
        case T_IQ4X: return dp4a_T<T_IQ4X>(T, IT, a, reps);
        case T_IQ4NL: return dp4a_T<T_IQ4NL>(T, IT, a, reps);
        case T_Q3K: return dp4a_T<T_Q3K>(T, IT, a, reps);
        default: return dp4a_T<T_IQ3S>(T, IT, a, reps);
    }
#endif
}
static float wv_any(int ty, int T, bool q16, WvArgs & a, int reps) {
#ifdef WV_QUICK
    switch (ty) {
        case T_Q4K: return wv_T<T_Q4K, WV_PF>(T, q16, a, reps);
        case T_Q5K: return wv_T<T_Q5K, WV_PF>(T, q16, a, reps);
        case T_IQ4X: return wv_T<T_IQ4X, WV_PF>(T, q16, a, reps);
    }
    return -1.f;
#else
    switch (ty) {
        case T_Q4K: return wv_T<T_Q4K, WV_PF>(T, q16, a, reps);
        case T_Q5K: return wv_T<T_Q5K, WV_PF>(T, q16, a, reps);
        case T_Q6K: return wv_T<T_Q6K, WV_PF>(T, q16, a, reps);
        case T_Q8_0: return wv_T<T_Q8_0, WV_PF>(T, q16, a, reps);
        case T_IQ4X: return wv_T<T_IQ4X, WV_PF>(T, q16, a, reps);
        case T_IQ4NL: return wv_T<T_IQ4NL, WV_PF>(T, q16, a, reps);
        case T_Q3K: return wv_T<T_Q3K, WV_PF>(T, q16, a, reps);
        default: return wv_T<T_IQ3S, WV_PF>(T, q16, a, reps);
    }
#endif
}

struct Dev {
    std::vector<Mat> mats;
    Mat * dmats = nullptr;
    uint8_t * w = nullptr;
    __half * d = nullptr;
    size_t bytes = 0;
    int nlay = 0;
};
static Dev upload(int ty, int M, int K, int nlay, std::mt19937 & rng) {
    Dev D;
    std::vector<uint8_t> w;
    std::vector<uint16_t> d;
    make_weights(ty, M, K, w, d, rng);
    const size_t mw = w.size(), md = d.size();
    D.nlay = nlay;
    CK(cudaMalloc(&D.w, mw * nlay));
    if (md) CK(cudaMalloc(&D.d, md * 2 * nlay));
    for (int L = 0; L < nlay; L++) { // the same random matrix in every layer copy (distinct addresses defeat L2)
        CK(cudaMemcpy(D.w + mw * L, w.data(), mw, cudaMemcpyHostToDevice));
        if (md) CK(cudaMemcpy(D.d + md * L, d.data(), md * 2, cudaMemcpyHostToDevice));
        Mat m{D.w + mw * L, md ? D.d + md * L : nullptr, ty, K, M, row_bytes(ty, K), row_meta_halves(ty, K)};
        D.mats.push_back(m);
    }
    D.bytes = (mw + md * 2);
    CK(cudaMalloc(&D.dmats, sizeof(Mat) * nlay));
    CK(cudaMemcpy(D.dmats, D.mats.data(), sizeof(Mat) * nlay, cudaMemcpyHostToDevice));
    return D;
}
static void release(Dev & D) {
    cudaFree(D.w);
    if (D.d) cudaFree(D.d);
    cudaFree(D.dmats);
}

struct Acts {
    int8_t * xa, * xb;
    uint2 * meta;
    QB q;
};
static Acts upload_acts(int K, int ntok, bool q16, std::mt19937 & rng) {
    std::vector<int8_t> xa, xb;
    std::vector<uint2> meta;
    make_acts(K, ntok, q16, xa, xb, meta, rng);
    Acts A;
    CK(cudaMalloc(&A.xa, xa.size()));
    CK(cudaMalloc(&A.xb, xb.size()));
    CK(cudaMalloc(&A.meta, meta.size() * 8));
    CK(cudaMemcpy(A.xa, xa.data(), xa.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A.xb, xb.data(), xb.size(), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(A.meta, meta.data(), meta.size() * 8, cudaMemcpyHostToDevice));
    A.q = QB{A.xa, A.xb, A.meta, K / 32};
    return A;
}

static const int ALLTYPES[] = {T_Q4K, T_Q5K, T_IQ4X, T_Q6K, T_IQ4NL, T_Q8_0, T_Q3K, T_IQ3S};

static void init_grid(std::mt19937 & rng) {
    std::vector<uint32_t> grid(512);
    for (auto & gv : grid) {
        uint32_t v = 0;
        for (int k = 0; k < 4; k++) v |= (uint32_t) (1 + 2 * (rng() % 8)) << (8 * k);
        gv = v;
    }
    CK(cudaMemcpyToSymbol(g_iq3s_grid, grid.data(), 512 * 4));
}

// exactness: per type x role: wv at T = 1..8, 16 bit-identical per (row, token) to T = 16's; wv and dp4a vs fp64
static int check(const std::vector<int> & roles, const std::vector<int> & types) {
    std::mt19937 rng(1234);
    init_grid(rng);
    int bad = 0;
    const int TS[] = {1, 2, 3, 4, 5, 6, 7, 8, 16};
    for (int ri : roles) {
        const Role & R = ROLES[ri];
        Acts X = upload_acts(R.K, 16, R.q16, rng);
        for (int ty : types) {
            Dev D = upload(ty, R.M, R.K, 1, rng);
            float * out, * part;
            unsigned * cnt;
            double * ref;
            long long * idot;
            CK(cudaMalloc(&out, sizeof(float) * 16 * R.M));
            CK(cudaMalloc(&part, sizeof(float) * 16 * R.M * R.S));
            CK(cudaMalloc(&cnt, 4 * (R.M / 16)));
            CK(cudaMemset(cnt, 0, 4 * (R.M / 16)));
            CK(cudaMalloc(&ref, sizeof(double) * 16 * R.M));
            CK(cudaMalloc(&idot, sizeof(long long) * 16 * R.M));
            k_ref<<<R.M, 16 * 32>>>(D.mats[0], X.q, 16, R.K, ref, idot);
            CK(cudaGetLastError());
            std::vector<double> href((size_t) 16 * R.M);
            CK(cudaMemcpy(href.data(), ref, href.size() * 8, cudaMemcpyDeviceToHost));
            std::vector<std::vector<float>> res;
            for (int T : TS) {
                CK(cudaMemset(out, 0xff, sizeof(float) * 16 * R.M));
                WvArgs a{D.dmats, 1, X.q, out, part, cnt, R.M, R.K, R.S};
                const float ms = wv_any(ty, T, R.q16, a, 0);
                CK(cudaDeviceSynchronize());
                std::vector<float> h((size_t) 16 * R.M);
                CK(cudaMemcpy(h.data(), out, h.size() * 4, cudaMemcpyDeviceToHost));
                if (ms < 0) h.assign(h.size(), NAN);
                res.push_back(h);
            }
            // T-invariance against the largest T that ran (rows t < T)
            int tinv_bad = 0, tref = -1, ntested = 0;
            for (int ti = (int) res.size() - 1; ti >= 0 && tref < 0; ti--)
                if (!std::isnan(res[ti][0]) || !std::isnan(res[ti][1])) tref = ti;
            for (size_t ti = 0; ti < res.size(); ti++) {
                const int T = TS[ti];
                if (std::isnan(res[ti][0]) && std::isnan(res[ti][1])) continue;
                ntested++;
                for (int t = 0; t < T; t++)
                    if (memcmp(res[ti].data() + (size_t) t * R.M, res[tref].data() + (size_t) t * R.M, 4 * R.M)) tinv_bad++;
            }
            const int Tmax = TS[tref];
            const std::vector<float> & best = res[tref];
            const int Td = 4; // dp4a reference rows
            CK(cudaMemset(out, 0, sizeof(float) * 16 * R.M));
            DpArgs da{D.dmats, 1, X.q, out, R.M};
            dp4a_any(ty, Td, R.IT, da, 0);
            CK(cudaDeviceSynchronize());
            std::vector<float> hd((size_t) 16 * R.M);
            CK(cudaMemcpy(hd.data(), out, hd.size() * 4, cudaMemcpyDeviceToHost));
            double ew = 0, ed = 0, mw = 0, md = 0, rms = 0, dwd = 0;
            size_t n = 0, nd = 0;
            for (int t = 0; t < Tmax; t++)
                for (int r = 0; r < R.M; r++) {
                    const double rv = href[(size_t) t * R.M + r];
                    const double vw = best[(size_t) t * R.M + r];
                    rms += rv * rv;
                    ew += (vw - rv) * (vw - rv);
                    mw = fmax(mw, fabs(vw - rv));
                    n++;
                    if (t < Td) {
                        const double vd = hd[(size_t) t * R.M + r];
                        ed += (vd - rv) * (vd - rv);
                        md = fmax(md, fabs(vd - rv));
                        dwd = fmax(dwd, fabs(vd - (double) best[(size_t) t * R.M + r]));
                        nd++;
                    }
                }
            rms = sqrt(rms / n);
            printf("%-9s %-6s T-inv(%d T's, ref T=%d) %s (%d rows differ) | rel err vs fp64: wv rms %.2e max %.2e, dp4a rms "
                   "%.2e max %.2e | max |wv-dp4a| %.2e\n",
                   R.name, tname(ty), ntested, Tmax, tinv_bad ? "FAIL" : "ok", tinv_bad, sqrt(ew / n) / rms, mw / rms,
                   sqrt(ed / nd) / rms, md / rms, dwd / rms);
            if (tinv_bad || sqrt(ew / n) / rms > 1e-4) bad++;
            cudaFree(out); cudaFree(part); cudaFree(cnt); cudaFree(ref); cudaFree(idot);
            release(D);
        }
        cudaFree(X.xa); cudaFree(X.xb); cudaFree(X.meta);
    }
    printf(bad ? "CHECK FAILED (%d)\n" : "CHECK OK\n", bad);
    return bad;
}

static void timing(const std::vector<int> & roles, const std::vector<int> & types, int reps, const std::vector<int> & wvT,
                   const std::vector<int> & dpT) {
    std::mt19937 rng(77);
    init_grid(rng);
    for (int ri : roles) {
        const Role & R = ROLES[ri];
        Acts X = upload_acts(R.K, 16, R.q16, rng);
        for (int ty : types) {
            const size_t mb = (size_t) R.M * row_bytes(ty, R.K) + (size_t) R.M * row_meta_halves(ty, R.K) * 2;
            int nlay = (int) ((320u << 20) / mb);
            nlay = nlay < 2 ? 2 : nlay > 24 ? 24 : nlay;
            Dev D = upload(ty, R.M, R.K, nlay, rng);
            float * out, * part;
            unsigned * cnt;
            CK(cudaMalloc(&out, sizeof(float) * 16 * R.M));
            CK(cudaMalloc(&part, sizeof(float) * 16 * R.M * R.S));
            CK(cudaMalloc(&cnt, 4 * (R.M / 16) * nlay));
            CK(cudaMemset(cnt, 0, 4 * (R.M / 16) * nlay));
            const double gb = (double) D.bytes * nlay / 1e9;
            printf("%-9s %-6s %5.1f MB/layer x %2d |", R.name, tname(ty), D.bytes / 1e6, nlay);
            DpArgs da{D.dmats, nlay, X.q, out, R.M};
            for (int T : dpT) {
                const float ms = dp4a_any(ty, T, R.IT, da, reps);
                if (ms < 0) printf(" dp4a%d    -    ", T);
                else printf(" dp4a%d %6.1fus %4.0f", T, 1e3 * ms / nlay, gb / ms * 1e3);
            }
            printf(" |");
            WvArgs a{D.dmats, nlay, X.q, out, part, cnt, R.M, R.K, R.S};
            for (int T : wvT) {
                const float ms = wv_any(ty, T, R.q16, a, reps);
                if (ms < 0) printf(" wv%d    -    ", T);
                else printf(" wv%d %6.1fus %4.0f", T, 1e3 * ms / nlay, gb / ms * 1e3);
            }
            printf("\n");
            fflush(stdout);
            cudaFree(out); cudaFree(part); cudaFree(cnt);
            release(D);
        }
        cudaFree(X.xa); cudaFree(X.xb); cudaFree(X.meta);
    }
}

static std::vector<int> parse_list(const char * s, bool roles) {
    std::vector<int> v;
    std::string str(s);
    size_t pos = 0;
    while (pos <= str.size()) {
        size_t e = str.find(',', pos);
        if (e == std::string::npos) e = str.size();
        std::string tok = str.substr(pos, e - pos);
        if (roles) {
            for (int i = 0; i < (int) (sizeof(ROLES) / sizeof(ROLES[0])); i++)
                if (tok == ROLES[i].name || tok == "all") v.push_back(i);
        } else if (tok == "all") {
            v.assign(std::begin(ALLTYPES), std::end(ALLTYPES));
        } else {
            for (int ty : ALLTYPES)
                if (tok == tname(ty)) v.push_back(ty);
        }
        pos = e + 1;
    }
    return v;
}

int main(int argc, char ** argv) {
    cudaDeviceGetAttribute(&g_nsm, cudaDevAttrMultiProcessorCount, 0);
    CK(cudaGetSymbolAddress(&g_bar_ptr, g_bar));
    const std::string mode = argc > 1 ? argv[1] : "check";
    const std::vector<int> roles = parse_list(argc > 2 ? argv[2] : "all", true);
    const std::vector<int> types = parse_list(argc > 3 ? argv[3] : "all", false);
    if (argc > 6) // split-K override for every role (timing experiments; T-invariance needs one S per role for all T)
        for (auto & R : ROLES) R.S = atoi(argv[6]);
    if (mode == "check") return check(roles, types);
    const int reps = argc > 4 ? atoi(argv[4]) : 5;
    std::vector<int> wvT = {1, 2, 4, 8, 16}, dpT = {1, 2, 4, 5};
    auto ints = [](const char * s) {
        std::vector<int> v;
        for (const char * p = s; *p;) {
            v.push_back(atoi(p));
            while (*p && *p != ',') p++;
            if (*p == ',') p++;
        }
        return v;
    };
    if (argc > 5 && strcmp(argv[5], "-")) wvT = ints(argv[5]);
    if (argc > 7) dpT = ints(argv[7]);
    timing(roles, types, reps, wvT, dpT);
    return 0;
}
