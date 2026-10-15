// wv: int8 tensor-core verify GEMM (mma.m16n8k32, tokens on N) for the main-pass GEMV phases (MK_WV builds).
// Included by qwen38_mk.cu inside its anonymous namespace, after the dp4a phases. Port of the bench kernel
// megakernel/bench/wvgemm_bench.cu @ exp-wvgemm 5e653f7 (k_wv<TYPE, T, X16, PF = 0>, -DWV_TILEMAJ=2), which checks
// T = 1..8 bit-identical per row and errors vs fp64 at fp32-rounding level (notes/agent_wvgemm.md A6).
//
// A warp computes a 16-row tile (thread (g, tq): rows g, g + 8) x 8 tokens (one C column per token) per MMA over a K
// range, sequentially in K: every output is the same sequence of integer 32-sums and float FMAs at any T, so the path
// is T-invariant, but not bit-identical to the dp4a GEMV (per-32 integer sums); it must therefore serve every T.
//
// K permutation (the same for A and B): MMA k-slot 4 tq + i <-> element 8 tq + i, k-slot 16 + 4 tq + i <-> element
// 8 tq + 4 + i of the 32-element block: each thread supplies 8 consecutive elements of a weight row (one 8-byte piece of
// the fragment-ordered unit) and of a token row (one 8-byte LDS of the activation units).
// Integer -> float without I2F: the accumulator input is MAGIC = 0x4B400000 (12582912.0f), the MMA returns MAGIC + S
// (|S| < 2^22), fmaf(dx, that, -12582912 dx) = dx * S rounded once.
// Per-16 scales of the q16 activations (ffn_down, X16): "split" MMAs. Threads tq 0-1 hold chunk 0 (elements 0-15),
// tq 2-3 chunk 1; B column g carries token (g & 3) of the group and only the k-slots of chunk g >> 2, so C column n =
// the 16-element dot of token n & 3 over chunk n >> 2; the two chunk sums of a token meet in one shuffle after the K loop.
//
// Weight layout (loader, wv_relayout): every main-pass matrix of a wv type is stored as units (16-row tile, super-block)
// of 16 x BS contiguous bytes. Q4_K / Q5_K / IQ4_XS fragment-ordered ("FM"): the 16 rows' headers paired per g (row g
// | row g + 8; 16 B each for Q4_K / Q5_K, 8 B for IQ4_XS), [Q5_K: qh as 32 lanes x {row g 8 B | row g + 8 8 B}], then the
// 4 qs groups likewise (512 B each); every lane load is one 16-byte LDG and every warp load reads contiguous bytes.
// IQ4_NL / Q8_0: tile-major whole blocks (unit = the 16 rows' blocks of that super-block; Q8_0's scale stream likewise).
#pragma once

constexpr int WV_MAGIC = 0x4B400000;
constexpr float WV_FMAGIC = 12582912.0f;

__device__ __forceinline__ void mma16832(int (&d)[4], const int (&a)[4], int b0, int b1, const int (&c)[4]) {
    asm("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
// float(byte k of w) + 2^23 (k compile-time after unrolling)
__device__ __forceinline__ float mag23(unsigned w, int k) { return __int_as_float(__byte_perm(w, 0x4b000000u, 0x7540u | k)); }

// the weight types the wv path serves (the others stay on dp4a at every T: per tensor T-invariant)
__host__ __device__ constexpr bool wv_type(int t) { return t == T_Q4K || t == T_Q5K || t == T_IQ4X || t == T_IQ4NL || t == T_Q8_0; }
// fragment-ordered (FM) types; the other wv types are tile-major whole blocks
// (MK_WV_FMK=0: Q4_K / Q5_K tile-major whole blocks instead: their prefill GEMMs read whole 16-B chunks, but the
// decode GEMMs lose most of the wv gain (8-B loads at stride 32); notes/agent_wvplumb.md "wvplumb3")
__host__ __device__ constexpr bool wv_fm(int t) { return t == T_IQ4X || (MK_WV_FMK && (t == T_Q4K || t == T_Q5K)); }
// per-16 weight scales (MK_MAXT > 5 ffn_down types): split MMAs only
__host__ __device__ constexpr bool wv_s16(int t) { return t == T_Q6K || t == T_Q3K; }

// unit (tile, super-block sb) of an FM matrix, BS bytes per row and super-block
__device__ __forceinline__ const uint8_t * wv_unit(const Mat & m, int tile, int sb, int BS) {
    return m.w + ((size_t) tile * (m.K >> 8) + sb) * 16 * BS;
}
// tile-major whole blocks: block sb of row r (0..15) of the tile, and the meta stream (MB bytes per super-block)
__device__ __forceinline__ const uint8_t * wv_tblk(const Mat & m, int tile, int r, int sb, int BS) {
    return m.w + (((size_t) tile * (m.K >> 8) + sb) * 16 + r) * BS;
}
__device__ __forceinline__ const uint8_t * wv_tmeta(const Mat & m, int tile, int r, int sb, int MB) {
    return reinterpret_cast<const uint8_t *>(m.d) + (((size_t) tile * (m.K >> 8) + sb) * 16 + r) * MB;
}

// Per weight type: Raw = one super-block of one row as this thread loads it; Sc = its decoded scales; frag(k) = the two
// int8x4 words of elements 8 tq .. 8 tq + 7 of sub-block k; wsc(k) = the float scale of sub-block k and, for MIN == 1, the
// min coefficient (weight = fA q - gA). MIN: 0 none, 1 per-row min (Q4_K / Q5_K: one FFMA per output).
template <int TYPE> struct WV;

template <> struct WV<T_Q4K> { // 144 B: d, dmin | scales[12] | qs[128]
    static constexpr int MIN = 1, BS = 144;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { unsigned s0, s1, m0, m1; float d, nd, dm, ndm; };
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&r.h.x));
        const unsigned y = r.h.y, z = r.h.z, w = r.h.w;
        s.s0 = y & 0x3f3f3f3fu;
        s.m0 = z & 0x3f3f3f3fu;
        s.s1 = (w & 0x0f0f0f0fu) | ((y >> 2) & 0x30303030u);
        s.m1 = ((w >> 4) & 0x0f0f0f0fu) | ((z >> 2) & 0x30303030u);
        s.d = dm.x; s.nd = -8388608.0f * dm.x; s.dm = dm.y; s.ndm = -8388608.0f * dm.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        fA = fmaf(s.d, mag23(k < 4 ? s.s0 : s.s1, k & 3), s.nd);
        gA = fmaf(s.dm, mag23(k < 4 ? s.m0 : s.m1, k & 3), s.ndm);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = (r.q[k >> 1].x >> sh) & M4;
        w1 = (r.q[k >> 1].y >> sh) & M4;
    }
    // tile-major whole blocks (MK_WV_FMK=0)
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, 144);
        r.h = ldw4(blk);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = ldw2(blk + 16 + 32 * p + 8 * tq);
    }
};

template <> struct WV<T_Q5K> { // 176 B: d, dmin | scales[12] | qh[32] | qs[128]
    static constexpr int MIN = 1, BS = 176;
    struct Raw { int4 h; int2 qh; int2 q[4]; };
    using Sc = WV<T_Q4K>::Sc;
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) { // (MK_WV_FMK=0)
        const uint8_t * blk = wv_tblk(m, tile, row, sb, 176);
        r.h = ldw4(blk);
        r.qh = ldw2(blk + 16 + 8 * tq);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = ldw2(blk + 48 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        WV<T_Q4K>::Raw t;
        t.h = r.h;
        WV<T_Q4K>::scales(t, s);
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) { WV<T_Q4K>::wsc(s, k, fA, gA); }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | ((((unsigned) r.qh.x >> k) & 0x01010101u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | ((((unsigned) r.qh.y >> k) & 0x01010101u) << 4);
    }
};

template <> struct WV<T_IQ4X> { // per super-block: 8-B GGUF header (fp16 d | u16 scales_h | u8 scales_l[4]) | qs[128]
    static constexpr int MIN = 0, BS = 136;
    struct Raw { int2 h; int2 q[4]; };
    struct Sc { unsigned sl, sh; float d, nd; };
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.d = __half2float(__ushort_as_half((unsigned short) (r.h.x & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
        s.sh = (unsigned) r.h.x >> 16;
        s.sl = (unsigned) r.h.y;
    }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        const int ls = (int) (((s.sl >> (4 * k)) & 0xfu) | (((s.sh >> (2 * k)) & 3u) << 4)); // k < 8: nibble k of sl
        fA = fmaf(s.d, __int_as_float(WV_MAGIC - 32 + ls), s.nd);
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_IQ4NL> { // repacked 144 B: d[8] (fp16, one per 32) | qs[128] in Q4_K order; tile-major blocks
    static constexpr int MIN = 0, BS = 144;
    struct Raw { int4 h; int2 q[4]; };
    struct Sc { int4 h; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, 144);
        r.h = ldw4(blk);
#pragma unroll
        for (int p = 0; p < 4; p++) r.q[p] = ldw2(blk + 16 + 32 * p + 8 * tq);
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.h = r.h; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        const int w = k == 0 || k == 1 ? s.h.x : k == 2 || k == 3 ? s.h.y : k == 4 || k == 5 ? s.h.z : s.h.w;
        fA = __half2float(__ushort_as_half((unsigned short) ((unsigned) w >> (16 * (k & 1)))));
        gA = 0.f;
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) {
        int lo, hi;
        iq4_lut((unsigned) r.q[k >> 1].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w0 = (k & 1) ? hi : lo;
        iq4_lut((unsigned) r.q[k >> 1].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, lo, hi);
        w1 = (k & 1) ? hi : lo;
    }
};

template <> struct WV<T_Q8_0> { // repacked: per unit 32 B (A 16 | B 16); fp16 d per natural 32-block in Mat::d; tile-major
    static constexpr int MIN = 0, BS = 256;
    struct Raw { int4 d; int2 q[8]; };
    struct Sc { int4 d; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * rp = wv_tblk(m, tile, row, sb, 256);
        r.d = ldw4(wv_tmeta(m, tile, row, sb, 16));
#if MK_K6_LD
        // (stock2 K6_LD) q[k] = the 8 bytes at 32 (2 kk + (tq >> 1)) + 16 kb + 8 (tq & 1), k = 2 kk + kb. The lane pair tq = 2 h,
        // 2 h + 1 shares the 32-B sector 2 kk + h: the even lane loads its bytes 0-15 (its kb 0 | the odd lane's kb 0), the odd
        // lane 16-31 (the even lane's kb 1 | its kb 1); one exchange of 8 bytes completes both: whole sectors, half the loads.
        const bool odd = tq & 1;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const int4 v = ldw4(rp + 32 * (2 * kk + (tq >> 1)) + 16 * (tq & 1));
            const int s0 = __shfl_xor_sync(0xffffffffu, odd ? v.x : v.z, 1);
            const int s1 = __shfl_xor_sync(0xffffffffu, odd ? v.y : v.w, 1);
            r.q[2 * kk] = odd ? make_int2(s0, s1) : make_int2(v.x, v.y);
            r.q[2 * kk + 1] = odd ? make_int2(v.z, v.w) : make_int2(s0, s1);
        }
#else
#pragma unroll
        for (int k = 0; k < 8; k++) r.q[k] = ldw2(rp + 32 * (2 * (k >> 1) + (tq >> 1)) + 16 * (k & 1) + 8 * (tq & 1));
#endif
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.d = r.d; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        WV<T_IQ4NL>::Sc t;
        t.h = s.d;
        WV<T_IQ4NL>::wsc(t, k, fA, gA);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) { w0 = r.q[k].x; w1 = r.q[k].y; }
};

#if MK_MAXT > 5
// Offset types (MK_MAXT > 5: every main-layer ffn_down is on wv, these too), bench WV<..> ported: tile-major whole value
// blocks (16 x BS bytes per unit) and a tile-major meta stream (Q6_K / Q3_K: the loader's 36-B block pairs, IQ3_S: 16 B per
// block). Values are fed with offset 32 (MIN 2: the accumulator input is MAGIC - 32 qsum, wv_fill's min2 mx), so the
// weight is fA q'. Q6_K / Q3_K carry per-16 scales (S16): split MMAs only (X16), where h = tq >> 1 is the thread's chunk.
// frag / wsc take the thread's tq / h (MIN 2 types only: wv_sb).
__host__ __device__ constexpr bool wv_min2(int t) { return t == T_Q6K || t == T_Q3K || t == T_IQ3S; }
// the meta pair (Q6_K / Q3_K) of super-block sb of row r of the tile
__device__ __forceinline__ const uint8_t * wv_tpair(const Mat & m, int tile, int r, int sb) {
    return reinterpret_cast<const uint8_t *>(m.d) + (((size_t) tile * (m.K >> 9) + (sb >> 1)) * 16 + r) * (2 * Q6_META);
}
// Q6_K values 192 B per block: qlo[128] (Q4_K-order nibbles) | qhi[64] (8 B per unit u' = 2p + h: A's highs | B's, byte m bits
// 2j = element 4j + m of the 16-chunk); meta pair: sc[16] even | sc[16] odd | d even | d odd
template <> struct WV<T_Q6K> {
    static constexpr int MIN = 2, BS = Q6_BLK;
    static constexpr bool S16 = true;
    struct Raw { int sc[4]; int dd; int2 q[4]; int2 hb[4]; };
    struct Sc { int sc[4]; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, Q6_BLK);
        const uint8_t * mp = wv_tpair(m, tile, row, sb);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = ldw1(mp + 16 * (sb & 1) + 4 * k); // (meta is only 4-byte aligned)
        r.dd = (int) ((unsigned) ldw1(mp + 32) >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            r.q[p] = ldw2(blk + 32 * p + 8 * tq);
            r.hb[p] = ldw2(blk + 128 + 8 * (2 * p + (tq >> 1)));
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
    }
    // chunk c = 4p + 2 part + h of the block: byte c of sc[16]
    static __device__ __forceinline__ void wsc2(const Sc & s, int k, int h, float & fA) {
        const int c0 = 4 * (k >> 1) + 2 * (k & 1); // even: bytes c0, c0 + 1 lie in word c0 >> 2
        const unsigned w = (unsigned) s.sc[c0 >> 2] >> (8 * h);
        const int v = (int) (signed char) (w >> (8 * (c0 & 3)));
        fA = fmaf(s.d, __int_as_float(WV_MAGIC + v), s.nd);
    }
    static __device__ __forceinline__ void frag2(const Raw & r, int k, int tq, int & w0, int & w1) {
        const int sh = 4 * (k & 1);
        const unsigned hs = (k & 1) ? (unsigned) r.hb[k >> 1].y : (unsigned) r.hb[k >> 1].x;
        const int j0 = 2 * (tq & 1);
        w0 = ((r.q[k >> 1].x >> sh) & M4) | (int) (((hs >> (2 * j0)) & 0x03030303u) << 4);
        w1 = ((r.q[k >> 1].y >> sh) & M4) | (int) (((hs >> (2 * j0 + 2)) & 0x03030303u) << 4);
    }
};
#if MK_WV_Q6F
// ---- FM Q6_K (MK_WV_Q6F): every main-layer Q6_K matrix, fragment-ordered with the meta inline (wv_relayout) ----------
// Unit (16-row tile, super-block) = 3360 B = 16 x (192 + 18):
//   [0, 256)     sc: per g {row g sc[16] | row g + 8 sc[16]} (int8, natural 16-chunk order)
//   [256, 288)   d: per g {row g fp16 | row g + 8 fp16}
//   [288, 2336)  qlo: 4 groups p x 512 B, lane l = 4 g + tq: {row g | row g + 8} x {word A | word B}: A = the Q4_K-order nibble
//                bytes 32 p + 4 tq .. + 3 (chunk 0 elements 4 tq .. + 3 of sub-block 2 p in the low nibbles, 2 p + 1 in the
//                high ones), B = bytes 32 p + 16 + 4 tq .. + 3 (chunk 1)
//   [2336, 3360) qhi: 4 groups p x 256 B, lane l: {row g H | row g + 8 H}; byte m of H = the high 2 bits of element 4 tq + m
//                of (sub-block, chunk) (2p, 0) at bits 4-5, (2p + 1, 0) at 6-7, (2p, 1) at 0-1, (2p + 1, 1) at 2-3
// Thread tq supplies elements 4 tq .. + 3 of both 16-chunks of each sub-block (11 coalesced loads per unit; the TM layout
// took 26 loads of 4-8 B). Two MMA forms on it, bit-identical to each other and to the TM split path (the same exact integer
// 16-chunk sums, the same float sequence per (row, token, chunk), chunk 0's and chunk 1's sums added last):
//   X16 (ffn_down's q16 activations): split m16n8k32, chunk 0 in k-slots 0-15 and chunk 1 in 16-31, B column n = token n & 3
//   of chunk n >> 2 (the split md / mx tables of the X16 fill);
//   q32 phases: m16n8k16 per 16-chunk over 8 tokens: the non-split md table + the fill's per-chunk qsum table (mx4).
// (Prefill: mmh's FM6 stage reads the same units, prefill_gemm.cuh.)
constexpr int Q6F_UNIT = 3360, Q6F_D = 256, Q6F_LO = 288, Q6F_HI = 2336;
struct Q6FRaw { int4 sc[2]; int dd; int2 q[2][4]; int hi[2][4]; };
__device__ __forceinline__ void q6f_load(Q6FRaw & r, const Mat & m, int tile, int sb, int g, int tq) {
    const uint8_t * u = m.w + ((size_t) tile * (m.K >> 8) + sb) * Q6F_UNIT;
    const int l = 4 * g + tq;
    r.sc[0] = ldw4(u + 32 * g);
    r.sc[1] = ldw4(u + 32 * g + 16);
    r.dd = ldw1(u + Q6F_D + 4 * g);
#pragma unroll
    for (int p = 0; p < 4; p++) {
        const int4 v = ldw4(u + Q6F_LO + 512 * p + 16 * l);
        r.q[0][p] = make_int2(v.x, v.y);
        r.q[1][p] = make_int2(v.z, v.w);
        const int2 h = ldw2(u + Q6F_HI + 256 * p + 8 * l);
        r.hi[0][p] = h.x;
        r.hi[1][p] = h.y;
    }
}
// the int8x4 operand (q + 32) of row rr (0: g, 1: g + 8), sub-block k, chunk c: elements 16 c + 4 tq .. + 3
__device__ __forceinline__ int q6f_frag(const Q6FRaw & r, int rr, int k, int c) {
    const int p = k >> 1, part = k & 1;
    const unsigned base = c ? (unsigned) r.q[rr][p].y : (unsigned) r.q[rr][p].x;
    const unsigned lo = (part ? base >> 4 : base) & 0x0f0f0f0fu;
    const unsigned H = (unsigned) r.hi[rr][p];
    const unsigned hp = c == 0 ? (part ? H >> 2 : H) : (part ? H << 2 : H << 4);
    return (int) (lo | (hp & 0x30303030u));
}
// d sc[cn] of row rr, 16-chunk cn = 2 k + c, computed as WV<T_Q6K>::wsc2 (fmaf form: exact)
__device__ __forceinline__ float q6f_fa(const Q6FRaw & r, float d, float nd, int rr, int cn) {
    const int4 s = r.sc[rr];
    const int w = (cn >> 2) == 0 ? s.x : (cn >> 2) == 1 ? s.y : (cn >> 2) == 2 ? s.z : s.w;
    const int v = (int) (signed char) ((unsigned) w >> (8 * (cn & 3)));
    return fmaf(d, __int_as_float(WV_MAGIC + v), nd);
}
__device__ __forceinline__ void mma16816(int (&d)[4], int a0, int a1, int b0, const int (&c)[4]) {
    asm("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0), "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}
#endif
// Q3_K values 96 B per block: unit u' = 2p + h: 2-bit lows at 8u' (A word | B word; byte m bits 2j = element 4j + m), highs
// at 64 + 4u' (byte m: bit j = A element 4j + m, bit 4 + j = B); meta as Q6_K (sc - 32 as int8, d). Fed u3 + 28 (offset 32).
template <> struct WV<T_Q3K> {
    static constexpr int MIN = 2, BS = Q3_BLK;
    static constexpr bool S16 = true;
    struct Raw { int sc[4]; int dd; int2 lo[4]; int hi[4]; };
    using Sc = WV<T_Q6K>::Sc;
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, Q3_BLK);
        const uint8_t * mp = wv_tpair(m, tile, row, sb);
#pragma unroll
        for (int k = 0; k < 4; k++) r.sc[k] = ldw1(mp + 16 * (sb & 1) + 4 * k);
        r.dd = (int) ((unsigned) ldw1(mp + 32) >> (16 * (sb & 1)));
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.lo[p] = ldw2(blk + 8 * u);
            r.hi[p] = ldw1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
#pragma unroll
        for (int k = 0; k < 4; k++) s.sc[k] = r.sc[k];
        s.d = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc2(const Sc & s, int k, int h, float & fA) { WV<T_Q6K>::wsc2(s, k, h, fA); }
    static __device__ __forceinline__ void frag2(const Raw & r, int k, int tq, int & w0, int & w1) {
        const unsigned x = (k & 1) ? (unsigned) r.lo[k >> 1].y : (unsigned) r.lo[k >> 1].x, hh = (unsigned) r.hi[k >> 1];
        const int j0 = 2 * (tq & 1), hs = 4 * (k & 1);
        w0 = (int) ((((x >> (2 * j0)) & 0x03030303u) | (((hh >> (j0 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
        w1 = (int) ((((x >> (2 * j0 + 2)) & 0x03030303u) | (((hh >> (j0 + 1 + hs)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
    }
};
// IQ3_S values 96 B per block: unit u' = 2p + h: the low bytes of A's 4 grid indices then B's at 8u', signs at 64 + 4u' (byte
// m: bit c = A element 4c + m, bit 4 + c = B); meta 16 B per block: qh[8] (byte u': index bit 8, A in bits 0-3, B in 4-7) |
// scales[4] (byte p: sub-block 2p | 2p + 1 << 4) | fp16 d | pad. Fed 32 +- grid (offset 32); per-32 scales (no S16).
template <> struct WV<T_IQ3S> {
    static constexpr int MIN = 2, BS = Q3_BLK;
    static constexpr bool S16 = false;
    struct Raw { int2 qh; int2 sd; int2 ix[4]; int sg[4]; };
    struct Sc { unsigned sc; float d, nd; };
    static __device__ __forceinline__ void load(Raw & r, const Mat & m, int tile, int row, int sb, int tq) {
        const uint8_t * blk = wv_tblk(m, tile, row, sb, Q3_BLK);
        const uint8_t * mp = wv_tmeta(m, tile, row, sb, IQ3S_META);
        r.qh = ldw2(mp);
        r.sd = ldw2(mp + 8);
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int u = 2 * p + (tq >> 1);
            r.ix[p] = ldw2(blk + 8 * u);
            r.sg[p] = ldw1(blk + 64 + 4 * u);
        }
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) {
        s.sc = (unsigned) r.sd.x;
        s.d = __half2float(__ushort_as_half((unsigned short) (r.sd.y & 0xffff)));
        s.nd = -WV_FMAGIC * s.d;
    }
    static __device__ __forceinline__ void wsc2(const Sc & s, int k, int, float & fA) {
        const int v = 1 + 2 * (int) ((s.sc >> (4 * k)) & 0xfu); // nibble k = sub-block k
        fA = fmaf(s.d, __int_as_float(WV_MAGIC + v), s.nd);
    }
    // (the qh byte of unit u' is per thread: u' = 2p + (tq >> 1))
    static __device__ __forceinline__ void frag2(const Raw & r, int k, int tq, int & w0, int & w1) {
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
#endif

// the thread's rows g (r0) and g + 8 (r1) of super-block sb of the tile
template <int TYPE>
__device__ __forceinline__ void wv_load2(typename WV<TYPE>::Raw & r0, typename WV<TYPE>::Raw & r1, const Mat & m, int tile, int sb,
                                         int g, int tq) {
    if constexpr (wv_fm(TYPE)) {
        constexpr int BS = WV<TYPE>::BS;
        const uint8_t * u = wv_unit(m, tile, sb, BS);
        const int l = 4 * g + tq;
        int qo;
        if constexpr (TYPE == T_IQ4X) {
            const int4 h = ldw4(u + 16 * g);
            r0.h = make_int2(h.x, h.y);
            r1.h = make_int2(h.z, h.w);
            qo = 128;
        } else {
            r0.h = ldw4(u + 32 * g);
            r1.h = ldw4(u + 32 * g + 16);
            qo = 256;
        }
        if constexpr (TYPE == T_Q5K) {
            const int4 v = ldw4(u + 256 + 16 * l);
            r0.qh = make_int2(v.x, v.y);
            r1.qh = make_int2(v.z, v.w);
            qo = 768;
        }
#pragma unroll
        for (int p = 0; p < 4; p++) {
            const int4 v = ldw4(u + qo + 512 * p + 16 * l);
            r0.q[p] = make_int2(v.x, v.y);
            r1.q[p] = make_int2(v.z, v.w);
        }
    } else {
        WV<TYPE>::load(r0, m, tile, g, sb, tq);
        WV<TYPE>::load(r1, m, tile, g + 8, sb, tq);
    }
}

// Shared-memory activations of a K chunk (wv_fill): the engine's unit layout (16-byte A / B chunks of the 32-element
// units, xa / xb, meta xm) with a token stride of US + 2 units, so the 8-byte B loads of a half-warp (4 tokens x 4 tq)
// hit distinct banks; md[(j * NG + n) * 4 + tq] = {dx_a, -12582912 dx_a, dx_b, -..dx_b} for the two tokens (a, b) of the
// thread's C columns in token group n and natural 32-block j of the chunk (split: of its 16-chunk), mx = (dx qsum)_a, _b.
struct WvXS {
    const int8_t * xa;
    const int8_t * xb;
    int ld; // bytes per token row
    const float4 * md;
    const float2 * mx;
#if MK_WV_Q6F
    const float4 * mx4; // q6 fills: per (j, n, tq) the bits of MAGIC - 32 qsum of (token a, chunk 0), (b, 0), (a, 1), (b, 1)
#endif
};
template <int NG> struct WvAcc { float v[NG][4]; };

template <int US, int T, bool X16>
__host__ __device__ constexpr int wv_smem() {
    using C = WvCfg<T, X16>;
    // (MK_WV_Q6F: + the q32 fills' per-chunk qsum table mx4, 16 B per (j, n, tq))
    return 2 * C::NTOKP * (US + 2) * 16 + T * (US + 2) * 8 + US * C::NG * 4 * (MK_WV_Q6F && !X16 ? 40 : 24);
}

// Fill units [u0, u0 + US) of T activation rows. norm: s_rs[t] = the RMS-norm scale of row t (as fill_xs).
// Also returns the dp4a view of the same units (token stride US + 2: rows_* with XP = 2).
// min2: mx = the MIN 2 accumulator inputs (bits of MAGIC - 32 qsum) instead of dx qsum (the matrix is an offset type)
// q6 (MK_WV_Q6F, q32 fills): also the per-chunk qsum table mx4 of the FM Q6_K tiles' m16n8k16 form
template <int T, bool X16, int US>
__device__ __forceinline__ WvXS wv_fill(const QBuf & q, int u0, bool norm, SX * sxo = nullptr, bool min2 = false, bool q6 = false) {
    using C = WvCfg<T, X16>;
    constexpr int LDU = US + 2;
    static_assert(wv_smem<US, T, X16>() <= SM_MISC, "wv activations");
    uint8_t * const sm = smem;
    int4 * xa = reinterpret_cast<int4 *>(sm);
    int4 * xb = xa + C::NTOKP * LDU;
    uint2 * xm = reinterpret_cast<uint2 *>(xb + C::NTOKP * LDU);
    float4 * md = reinterpret_cast<float4 *>(xm + T * LDU);
    float2 * mx = reinterpret_cast<float2 *>(md + US * C::NG * 4);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = tid; i < T * US; i += NT) {
        const int t = i / US, u = i - t * US;
        const size_t gi = (size_t) t * q.U + u0 + u;
        cp_async16(xa + t * LDU + u, q.xa + gi * 16);
        cp_async16(xb + t * LDU + u, q.xb + gi * 16);
    }
    for (int i = tid; i < T * US / 2; i += NT) {
        const int t = 2 * i / US, u = 2 * i - t * US;
        cp_async16(xm + t * LDU + u, q.meta + (size_t) t * q.U + u0 + u);
    }
    asm volatile("cp.async.commit_group;" ::: "memory");
    float * s_rs = misc(sm);
    if (norm) { // normed consumers always read the K = D residual stream (U_D = 160 partial sums)
        if (warp < T) {
            float pp[U_D / 32];
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) pp[k] = __ldcg(q.ssq + warp * U_D + lane + 32 * k);
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) acc += pp[k];
            acc = wsum(acc);
            if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
        }
    } else if (tid < T) {
        s_rs[tid] = 1.0f;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    // md / mx from the staged meta (no second global round trip)
    for (int i = tid; i < US * C::NG * 4; i += NT) {
        const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
        const int ub = j - (j & 7) + 2 * ((j & 7) >> 1), part = j & 1; // chunk-local unit of the 64-group (hh = 0)
        float v[4];
        float2 x2;
#pragma unroll
        for (int k = 0; k < 2; k++) {
            int t, hh = -1;
            if (X16) {
                t = 4 * n + ((2 * tq) & 3) + k;
                hh = tq >> 1;
            } else {
                t = 8 * n + 2 * tq + k;
            }
            float dx = 0.f;
            int qs = 0;
            if (t < T) {
                const uint2 m0 = xm[t * LDU + ub + (hh < 0 ? 0 : hh)];
                const uint2 m1 = xm[t * LDU + ub + 1];
                dx = __half2float(reinterpret_cast<const __half *>(&m0.x)[part]);
                const int q0 = reinterpret_cast<const short *>(&m0.y)[part], q1 = reinterpret_cast<const short *>(&m1.y)[part];
                qs = hh < 0 ? q0 + q1 : q0;
            }
            v[2 * k] = dx;
            v[2 * k + 1] = -WV_FMAGIC * dx;
            (k ? x2.y : x2.x) = min2 ? __int_as_float(WV_MAGIC - 32 * qs) : dx * (float) qs;
        }
        md[i] = make_float4(v[0], v[1], v[2], v[3]);
        mx[i] = x2;
    }
#if MK_WV_Q6F
    float4 * const mx4 = reinterpret_cast<float4 *>(mx + US * C::NG * 4);
    if (!X16 && q6)
        for (int i = tid; i < US * C::NG * 4; i += NT) { // tokens 8 n + 2 tq + {0, 1} (the non-split md's), chunks 0 / 1
            const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
            const int ub = j - (j & 7) + 2 * ((j & 7) >> 1), part = j & 1;
            int qv[2][2];
#pragma unroll
            for (int k = 0; k < 2; k++) {
                const int t = 8 * n + 2 * tq + k;
#pragma unroll
                for (int c = 0; c < 2; c++) qv[k][c] = t < T ? reinterpret_cast<const short *>(&xm[t * LDU + ub + c].y)[part] : 0;
            }
            mx4[i] = make_float4(__int_as_float(WV_MAGIC - 32 * qv[0][0]), __int_as_float(WV_MAGIC - 32 * qv[1][0]),
                                 __int_as_float(WV_MAGIC - 32 * qv[0][1]), __int_as_float(WV_MAGIC - 32 * qv[1][1]));
        }
#endif
    __syncthreads();
    mark_fill();
    if (sxo) {
        sxo->xa = xa;
        sxo->xb = xb;
        sxo->xm = xm;
        sxo->U = LDU;
    }
#if MK_WV_Q6F
    return WvXS{reinterpret_cast<const int8_t *>(xa), reinterpret_cast<const int8_t *>(xb), LDU * 16, md, mx, mx4};
#else
    return WvXS{reinterpret_cast<const int8_t *>(xa), reinterpret_cast<const int8_t *>(xb), LDU * 16, md, mx};
#endif
}

// one super-block (8 sub-blocks) of the tile from the thread's raw rows r0 (row g) and r1 (row g + 8); sbl = the
// super-block's index within the shared-memory chunk; xra / xrb = this thread's A / B unit row (+ 8 tq)
template <int TYPE, int T, bool X16>
__device__ __forceinline__ void wv_sb(const typename WV<TYPE>::Raw & r0, const typename WV<TYPE>::Raw & r1, int sbl,
                                      const WvXS & xs, const int8_t * xra, const int8_t * xrb, bool bon,
                                      float (&acc)[WvCfg<T, X16>::NG][4]) {
    using W = WV<TYPE>;
    using C = WvCfg<T, X16>;
    const int tq = threadIdx.x & 3;
    typename W::Sc s0, s1;
    W::scales(r0, s0);
    W::scales(r1, s1);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        int a[4];
        float fA0, gA0, fA1, gA1;
        if constexpr (W::MIN == 2) { // offset types (MK_MAXT > 5): the thread's tq / chunk h select its bytes / scales
            W::frag2(r0, k, tq, a[0], a[2]);
            W::frag2(r1, k, tq, a[1], a[3]);
            W::wsc2(s0, k, tq >> 1, fA0);
            W::wsc2(s1, k, tq >> 1, fA1);
            gA0 = gA1 = 0.f;
        } else {
            W::frag(r0, k, a[0], a[2]);
            W::frag(r1, k, a[1], a[3]);
            W::wsc(s0, k, fA0, gA0);
            W::wsc(s1, k, fA1, gA1);
        }
        const int j = 8 * sbl + k;
        const int8_t * xr = ((k & 1) ? xrb : xra) + 128 * sbl + 32 * (k >> 1);
#pragma unroll
        for (int n = 0; n < C::NG; n++) {
            int2 bb = make_int2(0, 0);
            if (bon) bb = *reinterpret_cast<const int2 *>(xr + C::TPG * n * xs.ld);
            const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
            int ci[4] = {WV_MAGIC, WV_MAGIC, WV_MAGIC, WV_MAGIC};
            float2 mx = make_float2(0.f, 0.f);
            if (W::MIN) mx = xs.mx[(j * C::NG + n) * 4 + tq];
            if (W::MIN == 2) { // MAGIC - 32 qsum: the offset's sum (wv_fill with min2)
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

// Tile `tile` (rows 16 tile .. + 15) of m over super-blocks [sb0, sb0 + nsb) (the chunk in shared memory starts at sb0).
// C fragment: v[n][c] = row 16 tile + g + 8 (c >> 1), token TPG n + 2 tq + (c & 1) (split: tokens 4 n + 2 tq & 3 + (c & 1),
// valid in threads tq < 2). Out of line per (type, T, X16): each gets its own register allocation (128 cap).
#if MK_WV_Q6F
// FM Q6_K tile (see "FM Q6_K"): X16 = the split m16n8k32 form, else m16n8k16 per 16-chunk (xs from a q6 fill)
template <int T, bool X16>
__device__ __forceinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile_q6f(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs) {
    using C = WvCfg<T, X16>;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3, h = tq >> 1;
    WvAcc<C::NG> acc;
    float acc1[X16 ? 1 : C::NG][4]; // k16 form: chunk 1's sums (acc: chunk 0's)
#pragma unroll
    for (int n = 0; n < C::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc.v[n][c] = 0.f;
#pragma unroll
    for (int n = 0; n < (X16 ? 1 : C::NG); n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc1[n][c] = 0.f;
    // B: k16 token 8 n + g, elements 16 c + 4 tq of the sub-block; split token 4 n + (g & 3), chunk g >> 2 only
    const int tok = X16 ? (g & 3) : g, co = X16 ? 16 * (g >> 2) : 0;
    const int8_t * xra = xs.xa + tok * xs.ld + co + 4 * tq;
    const int8_t * xrb = xs.xb + tok * xs.ld + co + 4 * tq;
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
        Q6FRaw r;
        q6f_load(r, m, tile, sb0 + sb, g, tq);
        const float d0 = __half2float(__ushort_as_half((unsigned short) (r.dd & 0xffff)));
        const float d1 = __half2float(__ushort_as_half((unsigned short) ((unsigned) r.dd >> 16)));
        const float nd0 = -WV_FMAGIC * d0, nd1 = -WV_FMAGIC * d1;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int j = 8 * sb + k;
            const int8_t * xr = ((k & 1) ? xrb : xra) + 128 * sb + 32 * (k >> 1);
            const int w00 = q6f_frag(r, 0, k, 0), w10 = q6f_frag(r, 1, k, 0);
            const int w01 = q6f_frag(r, 0, k, 1), w11 = q6f_frag(r, 1, k, 1);
            if constexpr (!X16) {
                const float f00 = q6f_fa(r, d0, nd0, 0, 2 * k), f10 = q6f_fa(r, d1, nd1, 1, 2 * k);
                const float f01 = q6f_fa(r, d0, nd0, 0, 2 * k + 1), f11 = q6f_fa(r, d1, nd1, 1, 2 * k + 1);
#pragma unroll
                for (int n = 0; n < C::NG; n++) {
                    const int b0 = *reinterpret_cast<const int *>(xr + 8 * n * xs.ld);
                    const int b1 = *reinterpret_cast<const int *>(xr + 8 * n * xs.ld + 16);
                    const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
                    const float4 mx = xs.mx4[(j * C::NG + n) * 4 + tq];
                    const int ci0[4] = {__float_as_int(mx.x), __float_as_int(mx.y), __float_as_int(mx.x), __float_as_int(mx.y)};
                    const int ci1[4] = {__float_as_int(mx.z), __float_as_int(mx.w), __float_as_int(mx.z), __float_as_int(mx.w)};
                    int c0[4], c1[4];
                    mma16816(c0, w00, w10, b0, ci0);
                    mma16816(c1, w01, w11, b1, ci1);
                    acc.v[n][0] = fmaf(f00, fmaf(md.x, __int_as_float(c0[0]), md.y), acc.v[n][0]);
                    acc.v[n][1] = fmaf(f00, fmaf(md.z, __int_as_float(c0[1]), md.w), acc.v[n][1]);
                    acc.v[n][2] = fmaf(f10, fmaf(md.x, __int_as_float(c0[2]), md.y), acc.v[n][2]);
                    acc.v[n][3] = fmaf(f10, fmaf(md.z, __int_as_float(c0[3]), md.w), acc.v[n][3]);
                    acc1[n][0] = fmaf(f01, fmaf(md.x, __int_as_float(c1[0]), md.y), acc1[n][0]);
                    acc1[n][1] = fmaf(f01, fmaf(md.z, __int_as_float(c1[1]), md.w), acc1[n][1]);
                    acc1[n][2] = fmaf(f11, fmaf(md.x, __int_as_float(c1[2]), md.y), acc1[n][2]);
                    acc1[n][3] = fmaf(f11, fmaf(md.z, __int_as_float(c1[3]), md.w), acc1[n][3]);
                }
            } else {
                const float fA0 = q6f_fa(r, d0, nd0, 0, 2 * k + h), fA1 = q6f_fa(r, d1, nd1, 1, 2 * k + h);
                const int a[4] = {w00, w10, w01, w11};
#pragma unroll
                for (int n = 0; n < C::NG; n++) {
                    const int x = *reinterpret_cast<const int *>(xr + 4 * n * xs.ld);
                    const int b0 = (g >> 2) ? 0 : x, b1 = (g >> 2) ? x : 0;
                    const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
                    const float2 mx = xs.mx[(j * C::NG + n) * 4 + tq];
                    const int ci[4] = {__float_as_int(mx.x), __float_as_int(mx.y), __float_as_int(mx.x), __float_as_int(mx.y)};
                    int cc[4];
                    mma16832(cc, a, b0, b1, ci);
                    acc.v[n][0] = fmaf(fA0, fmaf(md.x, __int_as_float(cc[0]), md.y), acc.v[n][0]);
                    acc.v[n][1] = fmaf(fA0, fmaf(md.z, __int_as_float(cc[1]), md.w), acc.v[n][1]);
                    acc.v[n][2] = fmaf(fA1, fmaf(md.x, __int_as_float(cc[2]), md.y), acc.v[n][2]);
                    acc.v[n][3] = fmaf(fA1, fmaf(md.z, __int_as_float(cc[3]), md.w), acc.v[n][3]);
                }
            }
        }
    }
    if constexpr (!X16) { // chunk 0 + chunk 1 (the split form's tq < 2 sum: own chunk-0 sum + the partner's chunk 1)
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc.v[n][c] += acc1[n][c];
    } else {
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc.v[n][c] += __shfl_xor_sync(0xffffffffu, acc.v[n][c], 2);
    }
    return acc;
}
#endif

template <int TYPE, int T, bool X16, int ROLE = 0> // ROLE: separate instances for MK_WV_OUT's out role (wv_tile_off)
__device__ __noinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile(const Mat m, int tile, int sb0, int nsb, const WvXS xs) {
#if MK_WV_Q6F
    if constexpr (TYPE == T_Q6K) {
        return wv_tile_q6f<T, X16>(m, tile, sb0, nsb, xs);
    } else {
#endif
    static_assert(!wv_s16(TYPE) || X16, "per-16 weight scales need the split MMAs");
    using C = WvCfg<T, X16>;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    WvAcc<C::NG> acc;
#pragma unroll
    for (int n = 0; n < C::NG; n++)
#pragma unroll
        for (int c = 0; c < 4; c++) acc.v[n][c] = 0.f;
    const int tok = X16 ? (g & 3) : g;
    const int8_t * xra = xs.xa + tok * xs.ld + 8 * tq;
    const int8_t * xrb = xs.xb + tok * xs.ld + 8 * tq;
    const bool bon = !X16 || (g >> 2) == (tq >> 1);
#pragma unroll 1
    for (int sb = 0; sb < nsb; sb++) {
        typename WV<TYPE>::Raw r0, r1;
        wv_load2<TYPE>(r0, r1, m, tile, sb0 + sb, g, tq);
        wv_sb<TYPE, T, X16>(r0, r1, sb, xs, xra, xrb, bon, acc.v);
    }
    if (X16) { // chunk 0 (tq 0-1) + chunk 1 (tq 2-3) of the same tokens
#pragma unroll
        for (int n = 0; n < C::NG; n++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc.v[n][c] += __shfl_xor_sync(0xffffffffu, acc.v[n][c], 2);
    }
    return acc;
#if MK_WV_Q6F
    }
#endif
}

template <int T, bool X16>
__device__ __forceinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile_any(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs) {
#if MK_WV_Q6F
    if constexpr (!X16) // FM Q6_K in the q32 phases (m16n8k16 form; the phase's fill was q6)
        if (m.type == T_Q6K) return wv_tile<T_Q6K, T, false>(m, tile, sb0, nsb, xs);
#endif
#if MK_MAXT > 5
    if constexpr (X16) { // ffn_down: the offset types too
        switch (m.type) {
            case T_Q6K: return wv_tile<T_Q6K, T, X16>(m, tile, sb0, nsb, xs);
            case T_Q3K: return wv_tile<T_Q3K, T, X16>(m, tile, sb0, nsb, xs);
            case T_IQ3S: return wv_tile<T_IQ3S, T, X16>(m, tile, sb0, nsb, xs);
            default: break;
        }
    }
#endif
    switch (m.type) {
        case T_Q4K: return wv_tile<T_Q4K, T, X16>(m, tile, sb0, nsb, xs);
        case T_Q5K: return wv_tile<T_Q5K, T, X16>(m, tile, sb0, nsb, xs);
        case T_IQ4NL: return wv_tile<T_IQ4NL, T, X16>(m, tile, sb0, nsb, xs);
        case T_Q8_0: return wv_tile<T_Q8_0, T, X16>(m, tile, sb0, nsb, xs);
        default: return wv_tile<T_IQ4X, T, X16>(m, tile, sb0, nsb, xs);
    }
}

#if MK_MAXT > 5 && MK_WV_OUT
// MK_WV_OUT's out role (offset types only): its own wv_tile instances (ROLE 1). Sharing ffn_down's made ptxas re-split the
// registers around those calls: ffn_down's resid_wv<8, true> spilled 16 / 40 -> 60 / 228 B and ran slower (spill track)
template <int T>
__device__ __forceinline__ WvAcc<WvCfg<T, true>::NG> wv_tile_off(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs) {
    switch (m.type) {
        case T_Q6K: return wv_tile<T_Q6K, T, true, 1>(m, tile, sb0, nsb, xs);
        case T_Q3K: return wv_tile<T_Q3K, T, true, 1>(m, tile, sb0, nsb, xs);
        default: return wv_tile<T_IQ3S, T, true, 1>(m, tile, sb0, nsb, xs);
    }
}
#endif

// ---- phases ----------------------------------------------------------------------------------------------------------
// Work split of a phase (bench k_wv): block b takes the tiles [ntile b / G, ntile (b + 1) / G) of the phase's tile space
// (G blocks per K chunk), its warps round-robin starting from the last warp, so the first warps, which also take the
// phase's dp4a items (alpha / beta, the hybrid's Q6_K / Q3_K / IQ3_S matrices: gemv_pairs over the low global warps),
// get fewer tiles. Every output depends only on its own row and token (T-invariant per tensor).

constexpr int WV_S = 4; // split-K chunks of out / ffn_down (fixed for every T)

// the hybrid's dp4a matrices on the wv fill (rows_* XP = 2)
template <int T, int IT, class Out>
__device__ __forceinline__ void gemv_pairs_dp(const Mat & m, int lo, int hi, Out out) {
    switch (m.type) {
        case T_Q6K: gemv_pairs<T, IT, T_Q6K, 2>(m, lo, hi, out); break;
        case T_Q3K: gemv_pairs<T, IT, T_Q3K, 2>(m, lo, hi, out); break;
        default: gemv_pairs<T, IT, T_IQ3S, 2>(m, lo, hi, out); break;
    }
}
template <int T>
__device__ __forceinline__ LaneT<T> rows16_dp(const Mat & m, int j) {
    switch (m.type) {
        case T_Q6K: return rows16<T, T_Q6K, 2>(m, j);
        case T_Q3K: return rows16<T, T_Q3K, 2>(m, j);
        default: return rows16<T, T_IQ3S, 2>(m, j);
    }
}

// out / ssm_out not in the wv layout: dp4a resid32 (T <= 5: any type, as before; T >= 6: only the hybrid types remain)
template <int T, int IT>
__device__ __forceinline__ void resid_dp(const Mat & W, const QBuf & in, float * x, bool add, const float * wnext, uint8_t * sm) {
    if constexpr (T <= 5) {
        resid_any<T, IT>(W, in, x, add, wnext, sm);
    } else {
        switch (W.type) {
            case T_Q6K: resid32<T, IT, T_Q6K>(W, in, x, add, wnext, sm); break;
            case T_Q3K: resid32<T, IT, T_Q3K>(W, in, x, add, wnext, sm); break;
            default: resid32<T, IT, T_IQ3S>(W, in, x, add, wnext, sm); break;
        }
    }
}

// C fragment (v[0][c]: row g + 8 (c >> 1), token 2 tq + (c & 1)) -> lane r < 16 holds row r (lanes 16-31: copies)
template <int T>
__device__ __forceinline__ LaneT<T> wv_to_lane(const WvAcc<1> & a) {
    const int r = threadIdx.x & 15;
    LaneT<T> o;
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int src = 4 * (r & 7) + (t >> 1);
        const float x0 = __shfl_sync(0xffffffffu, a.v[0][t & 1], src);
        const float x1 = __shfl_sync(0xffffffffu, a.v[0][2 + (t & 1)], src);
        o.v[t] = (r >> 3) ? x1 : x0;
    }
    return o;
}

// non-split C fragment of tile `tile` -> dst[t * ld + 16 tile + row] = v * s_rs[t]
template <int T>
__device__ __forceinline__ void wv_store(float * dst, int ld, int tile, const WvAcc<1> & a, const float * s_rs) {
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int t = 2 * tq + (c & 1);
        if (t < T) dst[t * ld + 16 * tile + g + 8 * (c >> 1)] = a.v[0][c] * s_rs[t];
    }
}

// delta-net input projections: alpha / beta (fp32 path, as dn_proj), qkv -> conv ring and z (wv tiles, or dp4a pairs
// for a hybrid-type matrix)
template <int T>
__device__ __noinline__ void dn_proj_wv(const LayerW & W, int dl, int n, int dn_c) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
#if MK_WV_Q6F // (every main-layer Q6_K is FM: the q32 tiles need the per-chunk qsum table)
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true, nullptr, false, W.qkv.type == T_Q6K || W.gate.type == T_Q6K);
#else
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true);
#endif
    const float * s_rs = misc(nullptr);
    float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int n_ab = AB_SPLIT * 2 * NVH;
    // alpha / beta items. Upstream's dealing (gwarp: warps 0-3 (+ 4) of every block) put them on warps that also take a
    // qkv / z tile below (tiles start at warp NW - 1 going down): those warps finished ~10 us after the others (stockfix K1).
    // K1 (cB.k1, every matrix of the phase in the wv layout): block b takes the items [n_ab b / G, n_ab (b + 1) / G) and
    // deals them round-robin to its warps with one tile fewer than the others (warps 0 .. NW - 1 - nb % NW, nb = the block's
    // tiles: at 82 blocks the 3-4 warps without a tile, <= 2 items each). Each item computes the same row / K-chunk as
    // before: bit-exact.
    int ab0 = gwarp(), ab1 = n_ab, astep = nwarps();
    if (cB.k1 && W.qkv.fm && W.gate.fm) {
        constexpr int NTL = QKV / 16 + VD / 16; // (the tile space below when both are FM)
        const int nb = NTL * (blockIdx.x + 1) / gridDim.x - NTL * blockIdx.x / gridDim.x;
        const int nl = NW - nb % NW; // warps 0 .. nl - 1 carry nb / NW tiles, the others one more
        ab1 = n_ab * (blockIdx.x + 1) / gridDim.x;
        ab0 = warp < nl ? n_ab * blockIdx.x / gridDim.x + warp : ab1;
        astep = nl;
    }
    for (int it = ab0; it < ab1; it += astep) {
        const int row = it % (2 * NVH), ch = it / (2 * NVH);
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        if (W.abq) {
            const int8_t * qr = W.abq + (size_t) row * D;
            const __half * dr = W.abd + (size_t) row * (D / 32);
            for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                const int qw = ldw1(qr + k);
                const float d = __half2float(dr[k >> 5]);
                const float4 fn = __ldg(reinterpret_cast<const float4 *>(W.attn_norm + k));
                float4 w;
                w.x = ((float) (int8_t) (qw) * d) * fn.x;
                w.y = ((float) (int8_t) (qw >> 8) * d) * fn.y;
                w.z = ((float) (int8_t) (qw >> 16) * d) * fn.z;
                w.w = ((float) (int8_t) (qw >> 24) * d) * fn.w;
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                    acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
                }
            }
        } else {
            const float * wr = W.ab + (size_t) row * D;
            for (int k = ch * (D / AB_SPLIT) + lane * 4; k < (ch + 1) * (D / AB_SPLIT); k += 128) {
                const float4 w = ldw4f(wr + k);
#pragma unroll
                for (int t = 0; t < T; t++) {
                    const float4 xv = __ldcg(reinterpret_cast<const float4 *>(cB.x + t * D + k));
                    acc[t] += w.x * xv.x + w.y * xv.y + w.z * xv.z + w.w * xv.w;
                }
            }
        }
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = wsum(acc[t]) * s_rs[t];
        store_lane_t<T>(cB.ab + ch * MAXT * 2 * NVH, 2 * NVH, row, acc);
    }
    int lo = n_ab;
    if (!W.qkv.fm) {
        gemv_pairs_dp<T, 5>(W.qkv, lo, lo + QKV / 2, PairOut{conv, QKV, n});
        lo += QKV / 2;
    }
    if (!W.gate.fm) gemv_pairs_dp<T, 5>(W.gate, lo, lo + VD / 2, PairOut{cB.z, VD, -1});
    const int nt_q = W.qkv.fm ? QKV / 16 : 0, ntile = nt_q + (W.gate.fm ? VD / 16 : 0);
    const int t0 = ntile * blockIdx.x / gridDim.x, t1 = ntile * (blockIdx.x + 1) / gridDim.x;
    for (int tile = t0 + (NW - 1 - warp); tile < t1; tile += NW) {
        const bool isq = tile < nt_q;
        const int lt = isq ? tile : tile - nt_q;
        const WvAcc<1> a = wv_tile_any<T, false>(isq ? W.qkv : W.gate, lt, 0, D / 256, xs);
        if (isq) {
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = 2 * tq + (c & 1), r = 16 * lt + g + 8 * (c >> 1);
                if (t < T) conv[((n + t) % CONV_RING) * QKV + r] = a.v[0][c] * s_rs[t];
            }
        } else {
            wv_store<T>(cB.z, VD, lt, a, s_rs);
        }
    }
    // Warm L2 with what dn_core will read (as dn_proj)
    if (blockIdx.x < NVH) {
        const int h = blockIdx.x, kh = h % NKH, tid = threadIdx.x;
        pf_l2(cB.dstate + ((size_t) dl * NVH + h) * 128 * 128 + tid * 32);
        if (tid == 0) {
            pf_l2(W.dt + (h & ~31));
            pf_l2(W.A + (h & ~31));
            pf_l2(W.snorm);
            pf_l2(W.snorm + 32);
            pf_l2(W.snorm + 64);
            pf_l2(W.snorm + 96);
        }
        if (tid < 384) {
            const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
            if ((tid & 7) == 0) pf_l2(W.conv + ch * 4);
            if ((tid & 31) == 0)
                for (int j = 0; j < 3; j++)
                    if (n - 3 + j >= 0) pf_l2(conv + ((n - 3 + j) % CONV_RING) * QKV + ch);
        }
        const float * ring = cB.kvgb + (size_t) dl * KV_RING * KVGB;
        for (int i = tid; i < (n - dn_c) * 8; i += NT) {
            const float * slot = ring + (size_t) ((dn_c + (i >> 3)) % KV_RING) * KVGB;
            const int c = (i & 7);
            pf_l2(c < 4 ? slot + kh * 128 + c * 32 : slot + KVGB_V + h * 128 + (c - 4) * 32);
        }
    }
}

// the activations of xs from super-block sb on (a K-split part's tile reads its chunk-local super-blocks from 0)
template <int T, bool X16>
__device__ __forceinline__ WvXS wv_xs_at(const WvXS & xs, int sb) {
    constexpr int NG = WvCfg<T, X16>::NG;
    WvXS r = xs;
    r.xa += 128 * sb;
    r.xb += 128 * sb;
    r.md += 8 * sb * NG * 4;
    r.mx += 8 * sb * NG * 4;
#if MK_WV_Q6F
    r.mx4 += 8 * sb * NG * 4;
#endif
    return r;
}

// stock2 K6 (MK_K6 bit 1): part `part` of `ns` of k / v tile lt (w = 1 k, 2 v): super-blocks [NSB part / ns, NSB (part + 1) / ns).
// The part's sums go to cB.wvpart (free in this phase: resid_wv uses it within its own phases) at [part][t][(w - 1) KVD + row];
// the last of the ns arrivals at the tile (counter cB.wvcnt[(w - 1) 64 + lt], role 0's, idle here; reset to 0 by that
// arrival) adds the parts in order 0 .. ns - 1 and stores (sum) * s_rs[t], as wv_store does for an unsplit tile. The split
// depends only on the matrix type, never on T: every output is the same function of its row and token at every T.
template <int T>
__device__ __noinline__ void attn_kv_part(const Mat m, int w, int lt, int part, int ns, const WvXS xs) {
    static_assert(MK_K6_S >= 1 && MK_K6_S <= WV_S && 2 * KVD <= D && 2 * (KVD / 16) <= D / 32, "K6 scratch: wvpart / wvcnt role 0");
    constexpr int NSB = D / 256;
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    const int sb0 = NSB * part / ns, sb1 = NSB * (part + 1) / ns;
    const WvAcc<1> a = wv_tile_any<T, false>(m, lt, sb0, sb1 - sb0, wv_xs_at<T, false>(xs, sb0));
    float * const pb = cB.wvpart + (w - 1) * KVD;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int t = 2 * tq + (c & 1), r = 16 * lt + g + 8 * (c >> 1);
        if (t < T) __stcg(pb + ((size_t) part * MAXT + t) * D + r, a.v[0][c]);
    }
    __threadfence();
    __syncwarp();
    unsigned * const cnt = cB.wvcnt + (w - 1) * (KVD / 16) + lt;
    unsigned old = 0;
    if (lane == 0) old = atomicAdd(cnt, 1u);
    old = __shfl_sync(0xffffffffu, old, 0);
    if (old != (unsigned) ns - 1) return;
    __threadfence();
    const float * s_rs = misc(nullptr);
    float * const dst = w == 1 ? cB.kbuf : cB.vbuf;
#pragma unroll
    for (int c = 0; c < 4; c++) {
        const int t = 2 * tq + (c & 1), r = 16 * lt + g + 8 * (c >> 1);
        if (t < T) {
            float s = __ldcg(pb + (size_t) t * D + r);
            for (int p = 1; p < ns; p++) s += __ldcg(pb + ((size_t) p * MAXT + t) * D + r);
            dst[t * KVD + r] = s * s_rs[t];
        }
    }
    if (lane == 0) *cnt = 0;
}

// stock2 K6 (MK_K6 bit 4, EXACT): rows 16 lt + 8 rh .. + 7 of Q8_0 k / v tile lt (w = 1 k, 2 v), the whole K. The thread
// loads only its row g + 8 rh (half the tile's bytes per warp) and feeds it as both MMA rows (g and g + 8): an MMA row's
// int32 sums and its scale use only that row's data, so the C values of the first row (c < 2) are bit-identical to the
// whole tile's values of that row; the duplicates (c >= 2) are dropped. Same store as wv_store.
template <int T>
__device__ __noinline__ void attn_kv_rh(const Mat m, int w, int lt, int rh, const WvXS xs) {
    using C = WvCfg<T, false>;
    static_assert(C::NG == 1, "K6 row halves: one token group");
    const int lane = threadIdx.x & 31, g = lane >> 2, tq = lane & 3;
    WvAcc<1> acc;
#pragma unroll
    for (int c = 0; c < 4; c++) acc.v[0][c] = 0.f;
    const int8_t * xra = xs.xa + g * xs.ld + 8 * tq;
    const int8_t * xrb = xs.xb + g * xs.ld + 8 * tq;
#pragma unroll 1
    for (int sb = 0; sb < D / 256; sb++) {
        typename WV<T_Q8_0>::Raw r0;
        WV<T_Q8_0>::load(r0, m, lt, g + 8 * rh, sb, tq);
        wv_sb<T_Q8_0, T, false>(r0, r0, sb, xs, xra, xrb, true, acc.v);
    }
    const float * s_rs = misc(nullptr);
    float * const dst = w == 1 ? cB.kbuf : cB.vbuf;
#pragma unroll
    for (int c = 0; c < 2; c++) {
        const int t = 2 * tq + c;
        if (t < T) dst[t * KVD + 16 * lt + g + 8 * rh] = acc.v[0][c] * s_rs[t];
    }
}

// attention q / k / v projections (wv tiles of the wv-layout matrices, dp4a pairs of the hybrid's)
template <int T>
__device__ __noinline__ void attn_proj_wv(const LayerW & W, int slot, int n) {
    const int warp = threadIdx.x >> 5;
#if MK_WV_Q6F // (every main-layer Q6_K is FM: the q32 tiles need the per-chunk qsum table)
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true, nullptr, false, W.q.type == T_Q6K || W.k.type == T_Q6K || W.v.type == T_Q6K);
#else
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true);
#endif
    const float * s_rs = misc(nullptr);
    int lo = 0;
    if (!W.q.fm) {
        gemv_pairs_dp<T, 5>(W.q, lo, lo + QG / 2, PairOut{cB.qbuf, QG, -1});
        lo += QG / 2;
    }
    if (!W.k.fm) {
        gemv_pairs_dp<T, 5>(W.k, lo, lo + KVD / 2, PairOut{cB.kbuf, KVD, -1});
        lo += KVD / 2;
    }
    if (!W.v.fm) gemv_pairs_dp<T, 5>(W.v, lo, lo + KVD / 2, PairOut{cB.vbuf, KVD, -1});
    const int nq = W.q.fm ? QG / 16 : 0, nk = W.k.fm ? KVD / 16 : 0, nv = W.v.fm ? KVD / 16 : 0;
    // stock2 K6: items = the q tiles, then the k tiles, then the v tiles; with bit 1 a Q8_0 k / v tile (87 KB vs 56 KB for a
    // Q5_K q tile: the phase's tail in the 10 Q8_0-v layers) is MK_K6_S items (K parts, attn_kv_part). Dealing: contiguous
    // runs per block (upstream), or with bit 2 item i -> block i % G (the heavy k / v items spread over the blocks)
    // (bit 4, exact, takes precedence over bit 1: a Q8_0 k / v tile is 2 items of 8 rows each, attn_kv_rh)
    const int k6 = cB.k6, ks = (k6 & 4) ? 2 : (k6 & 1) ? MK_K6_S : 1;
    const int sk = W.k.type == T_Q8_0 ? ks : 1, sv = W.v.type == T_Q8_0 ? ks : 1;
    const int nkk = nk * sk, ni = nq + nkk + nv * sv;
    int i0, i1, istep;
    if (k6 & 2) {
        i0 = blockIdx.x + gridDim.x * (NW - 1 - warp);
        i1 = ni;
        istep = gridDim.x * NW;
    } else {
        i0 = ni * blockIdx.x / gridDim.x + (NW - 1 - warp);
        i1 = ni * (blockIdx.x + 1) / gridDim.x;
        istep = NW;
    }
    for (int it = i0; it < i1; it += istep) {
        const int w = it < nq ? 0 : it < nq + nkk ? 1 : 2;
        const int j = it - (w == 0 ? 0 : w == 1 ? nq : nq + nkk), ns = w == 0 ? 1 : w == 1 ? sk : sv;
        const int lt = j / ns;
        if (ns > 1) {
            if (k6 & 4) attn_kv_rh<T>(w == 1 ? W.k : W.v, w, lt, j - lt * ns, xs);
            else attn_kv_part<T>(w == 1 ? W.k : W.v, w, lt, j - lt * ns, ns, xs);
            continue;
        }
        const WvAcc<1> a = wv_tile_any<T, false>(w == 0 ? W.q : w == 1 ? W.k : W.v, lt, 0, D / 256, xs);
        wv_store<T>(w == 0 ? cB.qbuf : w == 1 ? cB.kbuf : cB.vbuf, w == 0 ? QG : KVD, lt, a, s_rs);
    }
    // Warm L2 with what attn_core reads first (as attn_proj)
    pf_block(W.qn, HD * 4);
    pf_block(W.kn, HD * 4);
    pf_block(cB.rope + (size_t) n * 32, T * 32 * 8);
    int nch, CS;
    attn_dims(n + T, nch, CS);
    const size_t ctx = cB.ctx;
    const int rk = cB.kvq8 ? HD * cB.kvb[0][slot] / 8 : HD * 2, rv = cB.kvq8 ? HD * cB.kvb[1][slot] / 8 : HD * 2;
    const int lk = rk / 128, lv = rv / 128, lines = max(lk, lv);
    const uint8_t * Kc = cB.kvq8 ? reinterpret_cast<const uint8_t *>(cB.kvp[0][slot])
                                 : static_cast<const uint8_t *>(cB.kc) + (size_t) slot * NKV * ctx * rk;
    const uint8_t * Vc = cB.kvq8 ? reinterpret_cast<const uint8_t *>(cB.kvp[1][slot])
                                 : static_cast<const uint8_t *>(cB.vc) + (size_t) slot * NKV * ctx * rv;
    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c0 = (item % nch) * CS, c1 = min(n, min(c0 + CS, c0 + 64));
        for (int i = threadIdx.x; i < (c1 - c0) * lines; i += NT) {
            const size_t pos = (size_t) g * ctx + c0 + i / lines;
            const int l = i % lines;
            if (l < lk) pf_l2(Kc + pos * rk + l * 128);
            if (l < lv) pf_l2(Vc + pos * rv + l * 128);
        }
    }
}

// q16_store of chunk j from lane layout (lanes < 16: row r of each token)
template <int T>
__device__ __forceinline__ void glu_store(int j, const float (&act)[T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < T; t += 2) {
        const float other = __shfl_sync(0xffffffffu, act[t + 1 < T ? t + 1 : t], lane & 15);
        const float v = lane < 16 ? act[t] : other;
        q16_store(cB.qa, lane < 16 ? t : t + 1, j, v, lane < 16 || t + 1 < T);
    }
}

// FFN gate / up: warp item = tile j of gate and of up -> silu(g) * u quantised per 16-chunk (chunk j) for ffn_down
template <int T>
__device__ __noinline__ void ffn_gu_wv(const LayerW & W) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tq = lane & 3;
#if MK_WV_Q6F // (every main-layer Q6_K is FM: the q32 tiles need the per-chunk qsum table)
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true, nullptr, false, W.fg.type == T_Q6K || W.fu.type == T_Q6K);
#else
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true);
#endif
    const float * s_rs = misc(nullptr);
    constexpr int NJ = FF / 16;
    const int t0 = NJ * blockIdx.x / gridDim.x, t1 = NJ * (blockIdx.x + 1) / gridDim.x;
    const bool gw = W.fg.fm, uw = W.fu.fm;
    for (int j = t0 + (NW - 1 - warp); j < t1; j += NW) {
        float act[T];
        if (gw && uw) { // GLU on the C fragments, one transpose
            const WvAcc<1> g = wv_tile_any<T, false>(W.fg, j, 0, D / 256, xs);
            const WvAcc<1> u = wv_tile_any<T, false>(W.fu, j, 0, D / 256, xs);
            WvAcc<1> a;
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int t = 2 * tq + (c & 1);
                const float s = t < T ? s_rs[t] : 0.f;
                a.v[0][c] = silu(g.v[0][c] * s) * (u.v[0][c] * s);
            }
            const LaneT<T> l = wv_to_lane<T>(a);
#pragma unroll
            for (int t = 0; t < T; t++) act[t] = lane < 16 ? l.v[t] : 0.f;
        } else { // one of them on dp4a (rows16: lane r = row r, already scaled)
            LaneT<T> g, u;
            if (gw) {
                g = wv_to_lane<T>(wv_tile_any<T, false>(W.fg, j, 0, D / 256, xs));
#pragma unroll
                for (int t = 0; t < T; t++) g.v[t] *= s_rs[t];
            } else {
                g = rows16_dp<T>(W.fg, j);
            }
            if (uw) {
                u = wv_to_lane<T>(wv_tile_any<T, false>(W.fu, j, 0, D / 256, xs));
#pragma unroll
                for (int t = 0; t < T; t++) u.v[t] *= s_rs[t];
            } else {
                u = rows16_dp<T>(W.fu, j);
            }
#pragma unroll
            for (int t = 0; t < T; t++) act[t] = lane < 16 ? silu(g.v[t]) * u.v[t] : 0.f;
        }
        glu_store<T>(j, act);
    }
}

// Output rows with the residual add (out: K = VD, ffn_down: K = FF, X16 = its per-16 activations), split-K WV_S chunks
// (G = gridDim / WV_S blocks per chunk). Each chunk's partial sums go to cB.wvpart; the last of the 2 WV_S (tile, chunk)
// arrivals at a 32-row group adds the chunks in order 0 .. WV_S - 1 (fixed: T-invariant), adds the residual and
// re-quantises it for the next RMS-normed consumer (weight wnext), like resid32's epilogue.
template <int T, bool X16, int K, bool OFF = false> // OFF: MK_WV_OUT's out role (wv_tile_off)
__device__ __noinline__ void resid_wv(const Mat Wm, const QBuf in, const float * wnext, int role) {
    constexpr int NSB = K / 256, CH = NSB / WV_S, US = 8 * CH;
    static_assert(NSB % WV_S == 0, "split-K chunks of whole super-blocks");
    using C = WvCfg<T, X16>;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    if (threadIdx.x < 4 && blockIdx.x + threadIdx.x * gridDim.x < D / 32) pf_l2(wnext + 32 * (blockIdx.x + threadIdx.x * gridDim.x));
    const int G = gridDim.x / WV_S, b = blockIdx.x;
    if (b >= WV_S * G) return;
    const int kc = b / G, bi = b - kc * G;
#if MK_MAXT > 5
#if MK_WV_Q6F
    const WvXS xs = wv_fill<T, X16, US>(in, US * kc, false, nullptr, X16 && wv_min2(Wm.type), !X16 && Wm.type == T_Q6K);
#else
    const WvXS xs = wv_fill<T, X16, US>(in, US * kc, false, nullptr, X16 && wv_min2(Wm.type));
#endif
#else
    const WvXS xs = wv_fill<T, X16, US>(in, US * kc, false);
#endif
    constexpr int NTILE = D / 16;
    const int t0 = NTILE * bi / G, t1 = NTILE * (bi + 1) / G;
    unsigned * cnt = cB.wvcnt + role * (D / 32);
    float * x = cB.x;
    for (int tile = t0 + (NW - 1 - warp); tile < t1; tile += NW) {
#if MK_MAXT > 5 && MK_WV_OUT
        WvAcc<C::NG> a;
        if constexpr (OFF) a = wv_tile_off<T>(Wm, tile, CH * kc, CH, xs);
        else a = wv_tile_any<T, X16>(Wm, tile, CH * kc, CH, xs);
#else
        const WvAcc<C::NG> a = wv_tile_any<T, X16>(Wm, tile, CH * kc, CH, xs);
#endif
        float * part = cB.wvpart + (size_t) kc * MAXT * D;
        if (!X16 || tq < 2) {
#pragma unroll
            for (int n = 0; n < C::NG; n++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int t = X16 ? 4 * n + 2 * tq + (c & 1) : 8 * n + 2 * tq + (c & 1);
                    if (t < T) __stcg(part + (size_t) t * D + 16 * tile + g + 8 * (c >> 1), a.v[n][c]);
                }
        }
        __threadfence();
        __syncwarp();
        unsigned old = 0;
        if (lane == 0) old = atomicAdd(cnt + (tile >> 1), 1u);
        old = __shfl_sync(0xffffffffu, old, 0);
        if (old == 2 * WV_S - 1) { // last arrival at this 32-row group
            __threadfence();
            const int j = tile >> 1, row = 32 * j + lane;
            if (cB.k2) {
                // K2 (stockfix): the loop below made one dependent L2 round trip (+ q32_store's shuffle chains) per token,
                // serially. Here a batch of TB tokens issues all its partial / residual loads first, then runs the tokens'
                // epilogues (independent: their shuffle chains interleave). Same per-element order (((p0 + p1) + p2) + p3)
                // + x: bit-exact.
                constexpr int TB = MK_K2_TB < T ? MK_K2_TB : T;
                const float wn = wnext[row];
#pragma unroll // (T, TB compile-time: no runtime guards; x stores keep the batches apart)
                for (int tb = 0; tb < T; tb += TB) {
                    float p[TB][WV_S], xo[TB];
#pragma unroll
                    for (int u = 0; u < TB; u++)
                        if (tb + u < T) {
#pragma unroll
                            for (int k = 0; k < WV_S; k++) p[u][k] = __ldcg(cB.wvpart + ((size_t) k * MAXT + tb + u) * D + row);
                            xo[u] = __ldcg(x + (tb + u) * D + row);
                        }
#pragma unroll
                    for (int u = 0; u < TB; u++)
                        if (tb + u < T) {
                            float s = p[u][0];
#pragma unroll
                            for (int k = 1; k < WV_S; k++) s += p[u][k];
                            const float xv = s + xo[u];
                            x[(tb + u) * D + row] = xv;
                            q32_store(cB.qn, tb + u, j, xv * wn, xv * xv, true);
                        }
                }
            } else {
#pragma unroll 1
                for (int t = 0; t < T; t++) {
                    float s = __ldcg(cB.wvpart + (size_t) t * D + row);
#pragma unroll
                    for (int k = 1; k < WV_S; k++) s += __ldcg(cB.wvpart + ((size_t) k * MAXT + t) * D + row);
                    const float xv = s + __ldcg(x + t * D + row);
                    x[t * D + row] = xv;
                    q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
                }
            }
            if (lane == 0) cnt[j] = 0;
        }
    }
}

// out / ssm_out (K = VD) in the wv layout, residual add into x, re-quantised for the next normed consumer (role 0).
// MK_WV_OUT (MK_MAXT > 5): the offset types run the split MMAs (X16, min2 fill): their per-16 weight scales need them, and
// the q32 activations fit the split view as they are (a block's two 16-chunks are units with the block's scale each)
template <int T>
__device__ __forceinline__ void resid_out_wv(const Mat & Wm, const float * wnext) {
#if MK_MAXT > 5 && MK_WV_OUT
    if (wv_min2(Wm.type) && !(MK_WV_Q6F && Wm.type == T_Q6K)) { // (MK_WV_Q6F: Q6_K runs the q32 FM tiles below)
        resid_wv<T, true, VD, true>(Wm, cB.qo, wnext, 0);
        return;
    }
#endif
    resid_wv<T, false, VD>(Wm, cB.qo, wnext, 0);
}
