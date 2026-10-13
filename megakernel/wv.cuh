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
__host__ __device__ constexpr bool wv_fm(int t) { return t == T_Q4K || t == T_Q5K || t == T_IQ4X; }

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
};

template <> struct WV<T_Q5K> { // 176 B: d, dmin | scales[12] | qh[32] | qs[128]
    static constexpr int MIN = 1, BS = 176;
    struct Raw { int4 h; int2 qh; int2 q[4]; };
    using Sc = WV<T_Q4K>::Sc;
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
#pragma unroll
        for (int k = 0; k < 8; k++) r.q[k] = ldw2(rp + 32 * (2 * (k >> 1) + (tq >> 1)) + 16 * (k & 1) + 8 * (tq & 1));
    }
    static __device__ __forceinline__ void scales(const Raw & r, Sc & s) { s.d = r.d; }
    static __device__ __forceinline__ void wsc(const Sc & s, int k, float & fA, float & gA) {
        WV<T_IQ4NL>::Sc t;
        t.h = s.d;
        WV<T_IQ4NL>::wsc(t, k, fA, gA);
    }
    static __device__ __forceinline__ void frag(const Raw & r, int k, int & w0, int & w1) { w0 = r.q[k].x; w1 = r.q[k].y; }
};

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
};
template <int T, bool X16> struct WvCfg {
    static constexpr int TPG = X16 ? 4 : 8;
    static constexpr int NG = (T + TPG - 1) / TPG;
    static constexpr int NTOKP = NG * TPG;
};
template <int NG> struct WvAcc { float v[NG][4]; };

template <int US, int T, bool X16>
__host__ __device__ constexpr int wv_smem() {
    using C = WvCfg<T, X16>;
    return 2 * C::NTOKP * (US + 2) * 16 + T * (US + 2) * 8 + US * C::NG * 4 * 24;
}

// Fill units [u0, u0 + US) of T activation rows. norm: s_rs[t] = the RMS-norm scale of row t (as fill_xs).
// Also returns the dp4a view of the same units (token stride US + 2: rows_* with XP = 2).
template <int T, bool X16, int US>
__device__ __forceinline__ WvXS wv_fill(const QBuf & q, int u0, bool norm, SX * sxo = nullptr) {
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
    for (int i = tid; i < US * C::NG * 4; i += NT) {
        const int j = i / (C::NG * 4), n = (i / 4) % C::NG, tq = i & 3;
        const int jg = u0 + j, b = jg >> 3, s = jg & 7, p = s >> 1, part = s & 1;
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
                const uint2 m0 = __ldcg(q.meta + (size_t) t * q.U + 8 * b + 2 * p + (hh < 0 ? 0 : hh));
                const uint2 m1 = __ldcg(q.meta + (size_t) t * q.U + 8 * b + 2 * p + 1);
                dx = __half2float(reinterpret_cast<const __half *>(&m0.x)[part]);
                const int q0 = reinterpret_cast<const short *>(&m0.y)[part], q1 = reinterpret_cast<const short *>(&m1.y)[part];
                qs = hh < 0 ? q0 + q1 : q0;
            }
            v[2 * k] = dx;
            v[2 * k + 1] = -WV_FMAGIC * dx;
            (k ? x2.y : x2.x) = dx * (float) qs;
        }
        md[i] = make_float4(v[0], v[1], v[2], v[3]);
        mx[i] = x2;
    }
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
    mark_fill();
    if (sxo) {
        sxo->xa = xa;
        sxo->xb = xb;
        sxo->xm = xm;
        sxo->U = LDU;
    }
    return WvXS{reinterpret_cast<const int8_t *>(xa), reinterpret_cast<const int8_t *>(xb), LDU * 16, md, mx};
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
        W::frag(r0, k, a[0], a[2]);
        W::frag(r1, k, a[1], a[3]);
        float fA0, gA0, fA1, gA1;
        W::wsc(s0, k, fA0, gA0);
        W::wsc(s1, k, fA1, gA1);
        const int j = 8 * sbl + k;
        const int8_t * xr = ((k & 1) ? xrb : xra) + 128 * sbl + 32 * (k >> 1);
#pragma unroll
        for (int n = 0; n < C::NG; n++) {
            int2 bb = make_int2(0, 0);
            if (bon) bb = *reinterpret_cast<const int2 *>(xr + C::TPG * n * xs.ld);
            const float4 md = xs.md[(j * C::NG + n) * 4 + tq];
            const int ci[4] = {WV_MAGIC, WV_MAGIC, WV_MAGIC, WV_MAGIC};
            float2 mx = make_float2(0.f, 0.f);
            if (W::MIN) mx = xs.mx[(j * C::NG + n) * 4 + tq];
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
template <int TYPE, int T, bool X16>
__device__ __noinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile(const Mat m, int tile, int sb0, int nsb, const WvXS xs) {
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
}

template <int T, bool X16>
__device__ __forceinline__ WvAcc<WvCfg<T, X16>::NG> wv_tile_any(const Mat & m, int tile, int sb0, int nsb, const WvXS & xs) {
    switch (m.type) {
        case T_Q4K: return wv_tile<T_Q4K, T, X16>(m, tile, sb0, nsb, xs);
        case T_Q5K: return wv_tile<T_Q5K, T, X16>(m, tile, sb0, nsb, xs);
        case T_IQ4NL: return wv_tile<T_IQ4NL, T, X16>(m, tile, sb0, nsb, xs);
        case T_Q8_0: return wv_tile<T_Q8_0, T, X16>(m, tile, sb0, nsb, xs);
        default: return wv_tile<T_IQ4X, T, X16>(m, tile, sb0, nsb, xs);
    }
}

// ---- phases ----------------------------------------------------------------------------------------------------------

// delta-net input projections on wv: alpha / beta (fp32 path, as dn_proj), qkv -> conv ring and z tiles.
// Block b takes the tiles [ntile b / G, ntile (b + 1) / G) of the joint qkv | z tile space, its warps round-robin from
// the last warp (the first warps also take the alpha / beta items).
template <int T>
__device__ __noinline__ void dn_proj_wv(const LayerW & W, const Mat & mq, const Mat & mz, int dl, int n, int dn_c) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, tq = lane & 3;
    const WvXS xs = wv_fill<T, false, U_D>(cB.qn, 0, true);
    const float * s_rs = misc(nullptr);
    float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int n_ab = AB_SPLIT * 2 * NVH;
    for (int it = gwarp(); it < n_ab; it += nwarps()) {
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
    constexpr int NTQ = QKV / 16, NTILE = NTQ + VD / 16;
    const int t0 = NTILE * blockIdx.x / gridDim.x, t1 = NTILE * (blockIdx.x + 1) / gridDim.x;
    for (int tile = t0 + (NW - 1 - warp); tile < t1; tile += NW) {
        const bool isq = tile < NTQ;
        const int lt = isq ? tile : tile - NTQ;
        const WvAcc<1> a = wv_tile_any<T, false>(isq ? mq : mz, lt, 0, D / 256, xs);
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const int t = 2 * tq + (c & 1), r = 16 * lt + g + 8 * (c >> 1);
            if (t < T) {
                const float v = a.v[0][c] * s_rs[t];
                if (isq) conv[((n + t) % CONV_RING) * QKV + r] = v;
                else cB.z[t * VD + r] = v;
            }
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

