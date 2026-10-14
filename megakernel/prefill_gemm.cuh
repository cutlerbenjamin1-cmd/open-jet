// Batched (prefill) GEMM on fp16 tensor cores against the k-quant weights in their stored layout.
//
//   Y[tok][row] (+)= sum_k W[row][k] * X[tok][k]
//
// X is fp16 [tok][K] with the elements of every 16-group permuted (hperm) so that the 4-byte weight
// words give m16n8k16 A fragments directly.  W stays Q4_K / Q5_K (GGUF layout), Q6_K (224-byte
// repack, see qwen38_mk.cu), Q8_0 (unit repack), IQ4_XS (272-byte block pairs, indices in Q4_K order) or
// IQ4_NL (Q4_K-shaped 144-byte repack) and is dequantised to fp16 in registers.
// Tile: 128 weight rows x 128 tokens per CTA, 8 warps of 16 rows x 128 tokens.  K is staged 64 at a
// time (one 64-group of a super-block: two 32-element sub-blocks) through a cp.async pipeline with
// one block barrier per stage; per-row sub-block scales are decoded one stage ahead.  Products
// accumulate in fp16 over two stages (k = 128) and are then added to fp32 accumulators.
#pragma once

#include <cuda_fp16.h>
#include <cstdint>
#include <type_traits>

#ifndef MK_WV_Q6F
#define MK_WV_Q6F 0 // (qwen38_mk.cu sets it; FM Q6_K units, see mmh's FM6)
#endif

namespace pf {

constexpr int BM = 128, BN = 128, BK = 64;
constexpr int TQ4K = 12, TQ5K = 13, TQ6K = 14, TQ80 = 8, TIQ4 = 23, TIQ4NL = 20, TQ3K = 11, TIQ3S = 21;
// kvalues_iq4nl + 128 as bytes (the Q8_0 path's offset)
constexpr unsigned IQ4U_T0 = 0x3F2D1801u, IQ4U_T1 = 0x766A5D4Fu, IQ4U_T2 = 0xA6998D81u, IQ4U_T3 = 0xF1D9C5B5u;

struct WMat { // device view of one weight matrix (mirrors Mat in qwen38_mk.cu)
    const uint8_t * w;
    const __half * d;
    int type, K, N, rb, nd;
};

// Raw bytes of one stage (64-group p of super-block sb) per weight row, copied as 16 B chunks:
//   Q4_K: [hdr 16 | qs 32]                    src: blk + {0, 16 + 32p}
//   Q5_K: [hdr 16 | qh 32 | qs 32]            src: blk + {0, 16, 32, 48 + 32p, 64 + 32p}
//   Q6_K: [qlo 32 | qhi 16 | meta 8]          src: blk + {32p, 16 + 32p, 128 + 16p}, 8 B at 192 + 8p
//   Q8_0: [units 64 | d 4]                    src: blk + {64p + 16c}, 4 B of d at 2 * (2 * st)
//   IQ4_XS, IQ4_NL: as Q4_K
template <int TYPE> struct WT;
template <> struct WT<TQ4K> { static constexpr int BLK = 144, NCH = 3; };
template <> struct WT<TIQ4> { static constexpr int BLK = 136, NCH = 3; }; // in 272 B pairs, see mmh's load
template <> struct WT<TIQ4NL> { static constexpr int BLK = 144, NCH = 3; };
template <> struct WT<TQ5K> { static constexpr int BLK = 176, NCH = 5; };
template <> struct WT<TQ6K> { static constexpr int BLK = 192, NCH = 3; }; // + the meta stream (W.d), see load
template <> struct WT<TQ80> { static constexpr int BLK = 256, NCH = 4; };
// native Q3_K / IQ3_S (rows_q3k / rows_iq3s): the 16 B of index / low bits of the stage's two units; the rest (Q3_K: high
// bits 8 B + Q6_K-style meta 4 + 4 B; IQ3_S: signs 8 B + the block's 16 B meta) is copied by the first BM threads, see load
template <> struct WT<TQ3K> { static constexpr int BLK = 96, NCH = 1; };
template <> struct WT<TIQ3S> { static constexpr int BLK = 96, NCH = 1; };

template <int TYPE> __device__ __forceinline__ void chunk_src(int c, int & off, int & inc) {
    if constexpr (TYPE == TQ3K || TYPE == TIQ3S) { off = 0; inc = 16; }
    else if constexpr (TYPE == TQ4K || TYPE == TIQ4 || TYPE == TIQ4NL) { off = c == 0 ? 0 : 16 * c; inc = c == 0 ? 0 : 32; }
    else if constexpr (TYPE == TQ5K) { off = c < 3 ? 16 * c : 48 + 16 * (c - 3); inc = c < 3 ? 0 : 32; }
    else if constexpr (TYPE == TQ6K) { off = c < 2 ? 16 * c : 128; inc = c < 2 ? 32 : 16; }
    else { off = 16 * c; inc = 64; }
}

__device__ __forceinline__ void cpa16(unsigned s, const void * g) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa8(unsigned s, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa4(unsigned s, const void * g) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }

__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

constexpr unsigned M4 = 0x0F0F0F0Fu;

enum { OUT_STORE = 0, OUT_ADD = 1 };

struct Out {
    float * y; // y[tok * ldy + row]
    int ldy;
    int mode;
    __half * yh; // fused gate / up (mmh<.., true>): yh[tok * ldy + row] = silu(gate) * up, fp16 in hperm order
};

// Element e of a 16-group sits at slot hperm(e): 4 t + {0, 1} -> 2 t + {0, 1}, 4 t + {2, 3} ->
// 8 + 2 t + {0, 1}.  With that order a weight word (elements 4 t4 .. 4 t4 + 3) gives the A fragment
// registers a0 / a1 = (e0, e1) and a2 / a3 = (e2, e3) of an m16n8k16 MMA.
__host__ __device__ constexpr int hperm(int e) {
    return (e & ~15) | ((((e >> 1) & 1) << 2 | ((e >> 2) & 3)) << 1) | (e & 1);
}

template <int TYPE> struct HT;
// SR: smem bytes per weight row; SEP: fp16 scales in a separate double buffer (else in the row:
// Q6_K rows are 64 B, chunks XOR-swizzled by (row / 2) % 4, scales in the meta chunk's spare 8 B)
template <> struct HT<TQ4K> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
template <> struct HT<TIQ4> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
template <> struct HT<TIQ4NL> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
template <> struct HT<TQ5K> { static constexpr int SR = 80, NS = 3; static constexpr bool SEP = true; };
template <> struct HT<TQ6K> { static constexpr int SR = 64, NS = 4; static constexpr bool SEP = false; };
template <> struct HT<TQ80> { static constexpr int SR = 80, NS = 3; static constexpr bool SEP = true; };
// Q3_K row: [lows 16 | highs 8 | sc 4 | d pair 4 | pad 16]; IQ3_S row: [index lows 16 | meta 16 | signs 8 | pad 8]
// (48 B rows: the 8 row groups g of a fragment load hit distinct banks, as Q4_K's)
template <> struct HT<TQ3K> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
template <> struct HT<TIQ3S> { static constexpr int SR = 48, NS = 4; static constexpr bool SEP = true; };
constexpr int HCH = 2; // stages accumulated in fp16 before adding to the fp32 accumulators
constexpr int HMI = 1, HWR = 16 * HMI, HNWM = BM / HWR; // warp tiles: 8 warps of 16 rows, each over all NB tokens
// NB: tokens per CTA (BN = 128 or, for short prompts, 64 / 32; picked per GEMM by the host, see Impl::pf_nb).
// Each output element is the same sequence of fp16 products and sums in any NB (the K order per row and token does
// not depend on which other tokens share the CTA), so every NB gives bit-identical results.
template <int NB> __host__ __device__ constexpr int hm_x() { return NB * 128; } // activation tile per stage: NB tokens x 64 halves, 16 B chunks XOR-swizzled
template <int TYPE, int NB = BN> __host__ __device__ constexpr int hm_stage() { return HT<TYPE>::SR * BM + hm_x<NB>(); }
// (A deeper cp.async pipeline for the narrow tiles, up to 8 stages, measured no faster: notes/agent_ttft.md.)
// At NB = 32 every type fits two CTAs per SM (<= 48 KB smem, <= 96 registers), at NB = 64 / 128 one.
template <int TYPE, int NB = BN> __host__ __device__ constexpr int hm_smem() {
    return HT<TYPE>::NS * hm_stage<TYPE, NB>() + (HT<TYPE>::SEP ? 2 * 2 * BM * 8 : 0);
}

__device__ __forceinline__ void hmma16(unsigned (&d)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                 : "+r"(d[0]), "+r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void hmma16z(unsigned (&d)[2], const unsigned (&a)[4], unsigned b0, unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%8};"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}
__device__ __forceinline__ __half2 u2h(unsigned u) { return *reinterpret_cast<__half2 *>(&u); }
__device__ __forceinline__ unsigned h2u(__half2 h) { return *reinterpret_cast<unsigned *>(&h); }

// GLU: W2 holds the up matrix (same type and shape as W = gate); the CTA computes 64 gate rows
// (warps 0-3) and the same 64 up rows (warps 4-7) and writes silu(gate) * up as fp16. Only for
// types whose rows carry their own scales (Q6_K / Q8_0 read them through W alone).
// LAY 1 (MK_WV): W (and W2) are stored in the wv layout of wv.cuh: units (16-row tile, super-block) of 16 x UB bytes.
// Q4_K / Q5_K / IQ4_XS are fragment-ordered there (FMT: a 16-B chunk of qh / qs holds 8 B of rows g and g + 8), IQ4_NL /
// Q8_0 (and Q8_0's scales) are tile-major whole blocks (shared memory receives the same bytes as from the row-major layout).
// FMT stage layout: the FM chunks land whole (one cp.async 16 each): per 16-row tile a 512-B region of qs (Q5_K: and qh)
// chunks, chunk (g, tq) at fmt_off = 128 (2 (g & 1) + (tq >> 1)) + 16 (2 (g >> 1) + (tq & 1)), then the headers (16 B per
// row; IQ4_XS: 16 B per row pair g / g + 8). rawq's 32-bit reads are bank-conflict-free because odd-g lanes read their
// chunks' second half (row g + 8) as their "row A": the lane's MMA rows are g + 8 (g & 1) and the other one, which the
// scales and the epilogue follow (the same products and sums per row: exact).
template <int TYPE, bool GLU = false, int NB = BN, int LAY = 0>
__global__ void __launch_bounds__(256, 1) mmh(WMat W, const __half * X, int ntok, Out O, const uint8_t * W2 = nullptr) {
    static_assert(!GLU || TYPE == TQ4K || TYPE == TQ5K || TYPE == TIQ4, "fused gate / up needs in-row scales");
    static_assert(!LAY || TYPE == TQ4K || TYPE == TQ5K || TYPE == TIQ4 || TYPE == TIQ4NL || TYPE == TQ80 || TYPE == TQ6K ||
                      TYPE == TQ3K || TYPE == TIQ3S, "wv layout types");
    constexpr bool FMT = LAY && (TYPE == TIQ4 || (MK_WV_FMK && (TYPE == TQ4K || TYPE == TQ5K))); // fragment-ordered
    // FM6 (MK_WV_Q6F): Q6_K in wv.cuh's FM Q6_K units (3360 B per tile and super-block, meta inline). Stage layout (the
    // weight part, 64 B x BM): qlo of tile t at F6L + 512 t (the unit's group-p block, whole), qhi at F6H + 256 t (likewise),
    // per row 16 B at F6M + 16 r: sc[4p .. 4p + 3] | d of rows g / g + 8 | the fp16 scale pairs (scales). The unit's qlo / qhi
    // order is the thread order of rawq (lane l = 4 g + t4: rows g / g + 8, elements 4 t4 .. + 3 of each 16-chunk), so rawq
    // reads one 16-B and one 8-B word and builds the same wd words as from the row-major stage: exact.
    constexpr bool FM6 = LAY && TYPE == TQ6K && MK_WV_Q6F;
    constexpr int F6L = 0, F6H = 4096, F6M = 6144, F6U = 3360;
    static_assert(NB % 32 == 0 && NB <= BN, "token tile: a multiple of 32 (X-tile copy loop, n8-tile pairs)");
    static_assert(NB * (BM + 4) * 4 <= hm_smem<TYPE, NB>(), "epilogue tile");
    constexpr int HWT = NB / (8 / HNWM), HNI = HWT / 8, XIT = NB * 8 / 256;
    extern __shared__ __align__(16) uint8_t sm[];
    constexpr int BLK = WT<TYPE>::BLK, NCH = WT<TYPE>::NCH, SR = HT<TYPE>::SR, NSH = HT<TYPE>::NS;
    constexpr int STG = hm_stage<TYPE, NB>();
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t4 = lane & 3;
    const int wm = warp % HNWM, wn = warp / HNWM; // warp tile: rows HWR wm .., tokens HWT wn ..
    const int ntt = gridDim.y;
    const int lin = blockIdx.y * gridDim.x + blockIdx.x;
    const int row0 = (lin / ntt) * (GLU ? BM / 2 : BM), tok0 = (lin % ntt) * NB;
    const unsigned sbase = (unsigned) __cvta_generic_to_shared(sm);
    uint2 * sws = reinterpret_cast<uint2 *>(sm + NSH * STG); // [2][2 j][BM] fp16 scale pairs

    // FMT stage layout (see above): Q5_K qh regions at FQH, qs regions at FQS, headers at FHO; FCPT chunks per tile and stage
    constexpr int FQH = 0, FQS = TYPE == TQ5K ? BM * 32 : 0, FHO = FQS + BM * 32;
    constexpr int FNQ = TYPE == TQ5K ? 64 : 32, FCPT = FNQ + (TYPE == TIQ4 ? 8 : 16);
    static_assert(!FMT || FHO + BM * 16 <= SR * BM, "FMT stage layout fits the stage's weight part");
    constexpr int WCH = FMT ? BM / 16 * FCPT : BM * NCH, WIT = (WCH + 255) / 256;
    static_assert(!FM6 || (WCH == BM / 16 * 48 && F6M + 16 * BM == SR * BM), "FM6: 48 chunks per tile and stage");
    constexpr int UB = FM6 ? F6U : 16 * (TYPE == TIQ4 ? 136 : BLK); // LAY: bytes per (tile, super-block) unit
    const uint8_t * wsrc[WIT];
    int winc[WIT];
    unsigned wdst[WIT];
    bool wq[WIT]; // IQ4_XS: an index chunk (not the pair's header), at + 128 in the odd block of a pair (LAY: not a header)
#pragma unroll
    for (int k = 0; k < WIT; k++) {
        const int i = tid + k * 256, r = i / NCH, c = i % NCH;
        int off, inc;
        chunk_src<TYPE>(c, off, inc);
        const int rr = i < WCH ? r : 0;
        if constexpr (LAY) {
            const uint8_t * mb = GLU ? (rr < BM / 2 ? W.w : W2) : W.w;
            const int rowm = row0 + (GLU ? (rr & (BM / 2 - 1)) : rr), q16 = rowm & 15, gg = q16 & 7, hf = q16 >> 3;
            const uint8_t * ub = mb + (size_t) (rowm >> 4) * (W.K >> 8) * UB;
            if constexpr (FMT) { // whole FM chunks: thread i -> (tile t, chunk cc); 8 consecutive threads -> 8 bank quads
                const int ii = i < WCH ? i : 0, t = ii / FCPT, cc = ii % FCPT;
                const uint8_t * mbt = GLU ? (t < BM / 32 ? W.w : W2) : W.w;
                const int trow = row0 + 16 * (GLU ? (t & (BM / 32 - 1)) : t);
                const uint8_t * ubt = mbt + (size_t) (trow >> 4) * (W.K >> 8) * UB;
                int o2, i2 = 0, d2;
                if (cc < FNQ) { // qs chunk (Q5_K: cc < 32 qh, then qs) of lane (gq, tq), stored at fmt_off(gq, tq)
                    const int qq = cc & 31, q = qq & 7, rs = qq >> 3, gq = 2 * (q >> 1) + (rs & 1), tq = 2 * (rs >> 1) + (q & 1);
                    const bool qh = TYPE == TQ5K && cc < 32;
                    o2 = (qh ? 256 : TYPE == TQ4K ? 256 : TYPE == TQ5K ? 768 : 128) + 16 * (4 * gq + tq);
                    i2 = qh ? 0 : 512; // qs of 64-group p: + 512 p
                    d2 = (qh ? FQH : FQS) + 512 * t + 128 * (2 * (rs & 1) + (rs >> 1)) + 16 * q;
                } else { // header: Q4_K / Q5_K row h of the tile (FM chunk 2 (h & 7) + (h >> 3)); IQ4_XS rows h, h + 8 (chunk h)
                    const int h = cc - FNQ;
                    o2 = TYPE == TIQ4 ? 16 * h : 16 * (2 * (h & 7) + (h >> 3));
                    d2 = FHO + 16 * ((TYPE == TIQ4 ? 8 : 16) * t + h);
                }
                wsrc[k] = ubt + o2;
                winc[k] = i2;
                wdst[k] = d2;
            } else if constexpr (FM6) { // thread i -> (tile t, chunk cc): qlo chunks 0..31, qhi chunks 32..47 of group p
                const int ii = i < WCH ? i : 0, t = ii / 48, cc = ii % 48;
                const uint8_t * ubt = W.w + (size_t) (row0 / 16 + t) * (W.K >> 8) * UB;
                wsrc[k] = ubt + (cc < 32 ? 288 + 16 * cc : 2336 + 16 * (cc - 32));
                winc[k] = cc < 32 ? 512 : 256;
                wdst[k] = cc < 32 ? F6L + 512 * t + 16 * cc : F6H + 256 * t + 16 * (cc - 32);
            } else {
                wsrc[k] = ub + q16 * BLK + off;
                winc[k] = inc;
            }
        } else {
            wsrc[k] = GLU ? (rr < BM / 2 ? W.w : W2) + (size_t) (row0 + (rr & (BM / 2 - 1))) * W.rb + off
                          : W.w + (size_t) (row0 + rr) * W.rb + off;
            winc[k] = inc;
        }
        wq[k] = c != 0;
        if constexpr (!FMT && !FM6) wdst[k] = r * SR + 16 * (TYPE == TQ6K ? c ^ ((r >> 1) & 3) : c);
    }
    // FM6: the thread's row (tid < BM) in its unit: sc at 32 g + 16 half (+ 4 p per stage), the d pair at 256 + 4 g
    const uint8_t * f6src = nullptr, * f6d = nullptr;
    if constexpr (FM6) {
        const int r = tid & (BM - 1), q16 = r & 15;
        const uint8_t * ur = W.w + (size_t) ((row0 + r) >> 4) * (W.K >> 8) * UB;
        f6src = ur + 32 * (q16 & 7) + 16 * (q16 >> 3);
        f6d = ur + 256 + 4 * (q16 & 7);
    }
    // LAY Q8_0: the thread's row of the tile-major scale stream (16 B per row and super-block)
    const uint8_t * dsrc8 = nullptr;
    if constexpr (LAY && TYPE == TQ80) {
        const int rowm = row0 + (tid & (BM - 1));
        dsrc8 = reinterpret_cast<const uint8_t *>(W.d) + ((size_t) (rowm >> 4) * (W.K >> 8) * 16 + (rowm & 15)) * 16;
    }
    // Q6_K: the row's meta stream (pairs of blocks: sc even | sc odd | d even | d odd, 36 B)
    const uint8_t * esrc = reinterpret_cast<const uint8_t *>(W.d + (size_t) (row0 + (tid & (BM - 1))) * W.nd);
    const __half * dsrc = W.d + (size_t) (row0 + (tid & (BM - 1))) * W.nd;
    const unsigned edst = (tid & (BM - 1)) * SR + (TYPE == TQ6K ? 16 * (3 ^ (((tid & (BM - 1)) >> 1) & 3)) : 64);
    // Q3_K / IQ3_S: the row's value stream (high bits / signs at 64 + 8 p of each block) and its smem row
    const uint8_t * vsrc = W.w + (size_t) (row0 + (tid & (BM - 1))) * W.rb;
    // LAY 1 offset types (MK_MAXT > 5 ffn_down): value blocks and meta entries are tile-major, so the row's stream advances by
    // 16 entries per block (VST: value bytes per super-block, EST: meta bytes per Q6_K / Q3_K block pair or IQ3_S block)
    constexpr int VST = LAY ? 16 * BLK : BLK, EST = LAY ? 16 * (TYPE == TIQ3S ? 16 : 36) : (TYPE == TIQ3S ? 16 : 36);
    if constexpr (LAY && !FM6 && (TYPE == TQ6K || TYPE == TQ3K || TYPE == TIQ3S)) {
        const int rowm = row0 + (tid & (BM - 1)), nb = W.K >> 8;
        esrc = reinterpret_cast<const uint8_t *>(W.d) + ((size_t) (rowm >> 4) * (TYPE == TIQ3S ? nb : nb / 2) * 16 + (rowm & 15)) *
                                                            (TYPE == TIQ3S ? 16 : 36);
        vsrc = W.w + ((size_t) (rowm >> 4) * nb * 16 + (rowm & 15)) * BLK;
    }
    const unsigned rdst = (tid & (BM - 1)) * SR;
    const __half * xsrc[XIT];
    unsigned xdst[XIT];
#pragma unroll
    for (int k = 0; k < XIT; k++) {
        const int i = tid + k * 256, t = i >> 3, c = i & 7;
        xsrc[k] = X + (size_t) (tok0 + t) * W.K + 8 * c;
        xdst[k] = SR * BM + t * 128 + ((c ^ (t & 7)) << 4);
    }

    auto load = [&](int st) {
        const unsigned s = sbase + (st % NSH) * STG;
        const int sb = st >> 2, p = st & 3;
#pragma unroll
        for (int k = 0; k < WIT; k++)
            if (WCH % 256 == 0 || tid + k * 256 < WCH) {
                if constexpr (LAY) { // (FMT: whole FM chunks to the FMT stage layout)
                    cpa16(s + wdst[k], wsrc[k] + (size_t) sb * UB + p * winc[k]);
                } else if constexpr (TYPE == TIQ4) // block pairs of 272 B: headers, then the even and the odd block's indices
                    cpa16(s + wdst[k], wsrc[k] + (size_t) (sb >> 1) * 272 + (wq[k] && (sb & 1) ? 128 : 0) + p * winc[k]);
                else
                    cpa16(s + wdst[k], wsrc[k] + sb * BLK + p * winc[k]);
            }
        if constexpr (FM6) { // the row's meta slot: sc[4p..4p+3] | d of the tile's rows g / g + 8 (scales picks one)
            if (tid < BM) {
                cpa4(s + F6M + 16 * tid, f6src + (size_t) sb * UB + 4 * p);
                cpa4(s + F6M + 16 * tid + 4, f6d + (size_t) sb * UB);
            }
        } else if constexpr (TYPE == TQ6K) { // meta slot: sc[4p..4p+3] | the pair's two d (scales picks one)
            if (tid < BM) {
                cpa4(s + edst, esrc + (sb >> 1) * EST + 16 * (sb & 1) + 4 * p);
                cpa4(s + edst + 4, esrc + (sb >> 1) * EST + 32);
            }
        } else if constexpr (TYPE == TQ80) {
            if (tid < BM) {
                if constexpr (LAY) cpa4(s + edst, dsrc8 + (size_t) sb * 256 + 4 * p);
                else cpa4(s + edst, dsrc + 2 * st);
            }
        } else if constexpr (TYPE == TQ3K) { // high bits of the stage's two units | sc[4p..4p+3] | the pair's two d
            if (tid < BM) {
                cpa8(s + rdst + 16, vsrc + sb * VST + 64 + 8 * p);
                cpa4(s + rdst + 24, esrc + (sb >> 1) * EST + 16 * (sb & 1) + 4 * p);
                cpa4(s + rdst + 28, esrc + (sb >> 1) * EST + 32);
            }
        } else if constexpr (TYPE == TIQ3S) { // the block's meta (qh[8] | scales[4] | d) | signs of the stage's two units
            if (tid < BM) {
                cpa16(s + rdst + 16, esrc + sb * EST);
                cpa8(s + rdst + 32, vsrc + sb * VST + 64 + 8 * p);
            }
        }
#pragma unroll
        for (int k = 0; k < XIT; k++) cpa16(s + xdst[k], xsrc[k] + BK * st);
    };

    // fp16 scales of stage st -> sws[st & 1][j][row] (thread = (row, j)):
    //   Q4_K / Q5_K: (S, S, -M, -M);  Q6_K: (S_lo, S_lo, S_hi, S_hi);  Q8_0: (d, d, d, d)
    auto scales = [&](int st) {
        const int hr = tid >> 1; // FMT: the headers at FHO (IQ4_XS: row pair chunk (tile, hr & 7), half hr >> 3 & 1)
        const uint8_t * row = sm + (st % NSH) * STG +
                              (!FMT ? hr * SR : TYPE == TIQ4 ? FHO + 16 * (8 * (hr >> 4) + (hr & 7)) + 8 * ((hr >> 3) & 1) : FHO + 16 * hr);
        const int j = tid & 1, p = st & 3;
        __half2 a, b;
        if constexpr (TYPE == TQ4K || TYPE == TQ5K) {
            const int4 hd = *reinterpret_cast<const int4 *>(row);
            const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&hd.x));
            const int sh = (p & 1) * 16 + 8 * j;
            const unsigned y = (unsigned) hd.y >> sh, z = (unsigned) hd.z >> sh, w = (unsigned) hd.w >> sh;
            unsigned sc, m;
            if (p < 2) {
                sc = y & 63;
                m = z & 63;
            } else {
                sc = (w & 0xF) | ((y >> 2) & 0x30);
                m = ((w >> 4) & 0xF) | ((z >> 2) & 0x30);
            }
            a = __float2half2_rn(dm.x * (float) sc);
            b = __float2half2_rn(-dm.y * (float) m);
        } else if constexpr (TYPE == TIQ4) { // the pair's two GGUF headers: fp16 d | u16 scales_h | u8 scales_l[4]
            const int2 hd = *reinterpret_cast<const int2 *>(row + (FMT ? 0 : 8 * ((st >> 2) & 1)));
            const float d = __half2float(__ushort_as_half((unsigned short) (hd.x & 0xffff)));
            const unsigned sl = ((unsigned) hd.y >> (8 * p + 4 * j)) & 0xfu, sh = ((unsigned) hd.x >> (16 + 4 * p + 2 * j)) & 3u;
            a = b = __float2half2_rn(d * (float) ((int) (sl | (sh << 4)) - 32));
        } else if constexpr (TYPE == TIQ4NL) { // header: the fp16 d of the 8 sub-blocks
            const __half dd = *reinterpret_cast<const __half *>(row + 4 * p + 2 * j);
            a = b = __half2half2(dd);
        } else if constexpr (TYPE == TQ3K) { // as Q6_K: (S_lo, S_lo), (S_hi, S_hi) of sub-block j, the same roundings
            const unsigned sc4 = *reinterpret_cast<const unsigned *>(row + 24), dd = *reinterpret_cast<const unsigned *>(row + 28);
            const float d = __half2float(__ushort_as_half((unsigned short) (dd >> (16 * ((st >> 2) & 1)))));
            const unsigned sp = sc4 >> (16 * j);
            a = __float2half2_rn(d * (float) (int8_t) (sp & 0xff));
            b = __float2half2_rn(d * (float) (int8_t) ((sp >> 8) & 0xff));
        } else if constexpr (TYPE == TIQ3S) { // d (1 + 2 s) of sub-block 2p + j, for both 16-halves
            const unsigned sc = *reinterpret_cast<const unsigned *>(row + 24);
            const float d = __half2float(*reinterpret_cast<const __half *>(row + 28));
            a = b = __float2half2_rn(d * (float) (int) (1 + 2 * ((sc >> (8 * p + 4 * j)) & 0xfu)));
        } else if constexpr (TYPE == TQ6K) {
            uint8_t * mrow = FM6 ? const_cast<uint8_t *>(sm + (st % NSH) * STG + F6M + 16 * hr)
                                 : const_cast<uint8_t *>(row) + 16 * (3 ^ (((tid >> 1) >> 1) & 3));
            const int2 meta = *reinterpret_cast<const int2 *>(mrow);
            const int dh = FM6 ? (hr >> 3) & 1 : (st >> 2) & 1; // FM6: rows g / g + 8 of the tile, else the pair's block
            const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) meta.y >> (16 * dh))));
            const unsigned sp = (unsigned) meta.x >> (16 * j);
            *reinterpret_cast<__half2 *>(mrow + 8 + 4 * j) =
                __floats2half2_rn(d * (float) (int8_t) (sp & 0xff), d * (float) (int8_t) ((sp >> 8) & 0xff));
            return;
        } else {
            const __half dd = *reinterpret_cast<const __half *>(row + 64 + 2 * j);
            a = b = __half2half2(dd);
        }
        if constexpr (HT<TYPE>::SEP) sws[(st & 1) * 2 * BM + j * BM + (tid >> 1)] = make_uint2(h2u(a), h2u(b));
    };

    float acc[HMI][HNI][4];
#pragma unroll
    for (int m = 0; m < HMI; m++)
#pragma unroll
        for (int i = 0; i < HNI; i++) acc[m][i][0] = acc[m][i][1] = acc[m][i][2] = acc[m][i][3] = 0.f;

    const int nst = W.K / BK;
#pragma unroll
    for (int s = 0; s < NSH - 1; s++) {
        if (s < nst) load(s);
        cpa_commit();
    }
    cpa_wait<NSH - 2>();
    __syncthreads();
    scales(0);

    // ldmatrix rows: lanes 0-7 / 8-15 / 16-23 / 24-31 -> tokens +0..7 (slots 0-7) / +0..7 (slots 8-15)
    // / +8..15 (slots 0-7) / +8..15 (slots 8-15) of a 16-token pair of n8 tiles
    const unsigned xl = SR * BM + (HWT * wn + 8 * (lane >> 4) + (lane & 7)) * 128;
    unsigned xo[4];
#pragma unroll
    for (int G = 0; G < 4; G++) xo[G] = xl + (((2 * G + ((lane >> 3) & 1)) ^ (lane & 7)) << 4);

    // integer-valued fp16 weights (offset removed) of a stage: vq[mi][j][kk][0..3], A-fragment order
    unsigned vq[HMI][2][2][4];
    auto rawq = [&](int st) {
        const uint8_t * sw = sm + (st % NSH) * STG;
        const int p = st & 3;
        constexpr float OFF = TYPE == TQ6K || TYPE == TIQ3S ? -1056.f
                            : TYPE == TQ3K                  ? -1028.f
                            : (TYPE == TQ80 || TYPE == TIQ4 || TYPE == TIQ4NL) ? -1152.f
                                                                               : -1024.f;
        const __half2 off2 = __float2half2_rn(OFF);
#pragma unroll
        for (int mi = 0; mi < HMI; mi++) {
        const int rw = HWR * wm + 16 * mi + g;
        const uint8_t * ra = sw + rw * SR;
        const uint8_t * rb = ra + 8 * SR;
        // FMT: word (half, sec) = bytes 16 half + 4 t4 .. of the 32-B group of the lane's row A (sec 0: g + 8 (g & 1)) / B
        const int od = g & 1;
        auto fw = [&](int fr, int half, int sec) {
            return *reinterpret_cast<const unsigned *>(sw + fr + 512 * (rw >> 4) + 128 * (2 * od + half) +
                                                       16 * (2 * (g >> 1) + (t4 >> 1)) + 8 * (od ^ sec) + 4 * (t4 & 1));
        };
        // weight words: wd[j][kk][r] = 4 values of elements 16 kk + 4 t4 .. of sub-block j, row g + 8 r
        unsigned wd[2][2][2];
        if constexpr (TYPE == TQ4K || TYPE == TQ5K) {
            constexpr int QO = TYPE == TQ4K ? 16 : 48;
            const unsigned w0 = FMT ? fw(FQS, 0, 0) : *reinterpret_cast<const unsigned *>(ra + QO + 4 * t4);
            const unsigned w1 = FMT ? fw(FQS, 1, 0) : *reinterpret_cast<const unsigned *>(ra + QO + 16 + 4 * t4);
            const unsigned w2 = FMT ? fw(FQS, 0, 1) : *reinterpret_cast<const unsigned *>(rb + QO + 4 * t4);
            const unsigned w3 = FMT ? fw(FQS, 1, 1) : *reinterpret_cast<const unsigned *>(rb + QO + 16 + 4 * t4);
            wd[0][0][0] = w0 & M4; wd[0][0][1] = w2 & M4; wd[0][1][0] = w1 & M4; wd[0][1][1] = w3 & M4;
            wd[1][0][0] = (w0 >> 4) & M4; wd[1][0][1] = (w2 >> 4) & M4;
            wd[1][1][0] = (w1 >> 4) & M4; wd[1][1][1] = (w3 >> 4) & M4;
            if constexpr (TYPE == TQ5K) {
                const unsigned h0 = (FMT ? fw(FQH, 0, 0) : *reinterpret_cast<const unsigned *>(ra + 16 + 4 * t4)) >> (2 * p);
                const unsigned h1 = (FMT ? fw(FQH, 1, 0) : *reinterpret_cast<const unsigned *>(ra + 32 + 4 * t4)) >> (2 * p);
                const unsigned h2 = (FMT ? fw(FQH, 0, 1) : *reinterpret_cast<const unsigned *>(rb + 16 + 4 * t4)) >> (2 * p);
                const unsigned h3 = (FMT ? fw(FQH, 1, 1) : *reinterpret_cast<const unsigned *>(rb + 32 + 4 * t4)) >> (2 * p);
                wd[0][0][0] |= (h0 & 0x01010101u) << 4; wd[0][0][1] |= (h2 & 0x01010101u) << 4;
                wd[0][1][0] |= (h1 & 0x01010101u) << 4; wd[0][1][1] |= (h3 & 0x01010101u) << 4;
                wd[1][0][0] |= (h0 & 0x02020202u) << 3; wd[1][0][1] |= (h2 & 0x02020202u) << 3;
                wd[1][1][0] |= (h1 & 0x02020202u) << 3; wd[1][1][1] |= (h3 & 0x02020202u) << 3;
            }
        } else if constexpr (TYPE == TIQ4 || TYPE == TIQ4NL) { // indices -> kvalues + 128 (low nibbles: sub-block 0, high: 1)
            const unsigned w0 = FMT ? fw(FQS, 0, 0) : *reinterpret_cast<const unsigned *>(ra + 16 + 4 * t4);
            const unsigned w1 = FMT ? fw(FQS, 1, 0) : *reinterpret_cast<const unsigned *>(ra + 32 + 4 * t4);
            const unsigned w2 = FMT ? fw(FQS, 0, 1) : *reinterpret_cast<const unsigned *>(rb + 16 + 4 * t4);
            const unsigned w3 = FMT ? fw(FQS, 1, 1) : *reinterpret_cast<const unsigned *>(rb + 32 + 4 * t4);
            int lo, hi;
            iq4_lut(w0, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][0][0] = lo; wd[1][0][0] = hi;
            iq4_lut(w2, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][0][1] = lo; wd[1][0][1] = hi;
            iq4_lut(w1, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][1][0] = lo; wd[1][1][0] = hi;
            iq4_lut(w3, IQ4U_T0, IQ4U_T1, IQ4U_T2, IQ4U_T3, lo, hi); wd[0][1][1] = lo; wd[1][1][1] = hi;
        } else if constexpr (TYPE == TQ3K) { // u3 = lows | high << 2 (the value is u3 - 4: OFF)
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint8_t * rr = r ? rb : ra;
                    const unsigned x = *reinterpret_cast<const unsigned *>(rr + 8 * kk);
                    const unsigned y = *reinterpret_cast<const unsigned *>(rr + 8 * kk + 4);
                    const unsigned h = *reinterpret_cast<const unsigned *>(rr + 16 + 4 * kk);
                    wd[0][kk][r] = ((x >> (2 * t4)) & 0x03030303u) | (((h >> t4) & 0x01010101u) << 2);
                    wd[1][kk][r] = ((y >> (2 * t4)) & 0x03030303u) | (((h >> (4 + t4)) & 0x01010101u) << 2);
                }
        } else if constexpr (TYPE == TIQ3S) { // 32 +- grid (the value: OFF)
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint8_t * rr = r ? rb : ra;
                    const unsigned qh = rr[16 + 2 * p + kk];
                    const unsigned sg = *reinterpret_cast<const unsigned *>(rr + 32 + 4 * kk);
#pragma unroll
                    for (int j = 0; j < 2; j++) {
                        const unsigned qs = rr[8 * kk + 4 * j + t4];
                        const unsigned gv = iq3s_grid(qs | (((qh >> (4 * j + t4)) & 1u) << 8));
                        const unsigned m = ((sg >> (4 * j + t4)) & 0x01010101u) * 0xffu;
                        wd[j][kk][r] = 0x20202020u + (gv ^ m) - m;
                    }
                }
        } else if constexpr (FM6) { // the lane's 16 B of qlo (rows g, g + 8: words A, B) and 8 B of qhi (rows g, g + 8)
            const uint4 q = *reinterpret_cast<const uint4 *>(sw + F6L + 512 * (rw >> 4) + 16 * (4 * g + t4));
            const uint2 hq = *reinterpret_cast<const uint2 *>(sw + F6H + 256 * (rw >> 4) + 8 * (4 * g + t4));
            constexpr unsigned M30 = 0x30303030u;
            wd[0][0][0] = (q.x & M4) | (hq.x & M30);
            wd[0][0][1] = (q.z & M4) | (hq.y & M30);
            wd[0][1][0] = (q.y & M4) | ((hq.x << 4) & M30);
            wd[0][1][1] = (q.w & M4) | ((hq.y << 4) & M30);
            wd[1][0][0] = ((q.x >> 4) & M4) | ((hq.x >> 2) & M30);
            wd[1][0][1] = ((q.z >> 4) & M4) | ((hq.y >> 2) & M30);
            wd[1][1][0] = ((q.y >> 4) & M4) | ((hq.x << 2) & M30);
            wd[1][1][1] = ((q.w >> 4) & M4) | ((hq.y << 2) & M30);
        } else if constexpr (TYPE == TQ6K) {
            const int z = (g >> 1) & 3; // rows g and g + 8 share the swizzle
            const unsigned w0 = *reinterpret_cast<const unsigned *>(ra + 16 * z + 4 * t4);
            const unsigned w1 = *reinterpret_cast<const unsigned *>(ra + 16 * (1 ^ z) + 4 * t4);
            const unsigned w2 = *reinterpret_cast<const unsigned *>(rb + 16 * z + 4 * t4);
            const unsigned w3 = *reinterpret_cast<const unsigned *>(rb + 16 * (1 ^ z) + 4 * t4);
            const uint4 ha = *reinterpret_cast<const uint4 *>(ra + 16 * (2 ^ z));
            const uint4 hb = *reinterpret_cast<const uint4 *>(rb + 16 * (2 ^ z));
            const int s = 2 * t4;
            auto mk = [](unsigned lo4, unsigned hi2) { return lo4 | ((hi2 & 0x03030303u) << 4); }; // 0..63
            wd[0][0][0] = mk(w0 & M4, ha.x >> s);
            wd[0][0][1] = mk(w2 & M4, hb.x >> s);
            wd[0][1][0] = mk(w1 & M4, ha.z >> s);
            wd[0][1][1] = mk(w3 & M4, hb.z >> s);
            wd[1][0][0] = mk((w0 >> 4) & M4, ha.y >> s);
            wd[1][0][1] = mk((w2 >> 4) & M4, hb.y >> s);
            wd[1][1][0] = mk((w1 >> 4) & M4, ha.w >> s);
            wd[1][1][1] = mk((w3 >> 4) & M4, hb.w >> s);
        } else {
#pragma unroll
            for (int j = 0; j < 2; j++)
#pragma unroll
                for (int kk = 0; kk < 2; kk++) {
                    wd[j][kk][0] = *reinterpret_cast<const unsigned *>(ra + 16 * j + 32 * kk + 4 * t4) ^ 0x80808080u;
                    wd[j][kk][1] = *reinterpret_cast<const unsigned *>(rb + 16 * j + 32 * kk + 4 * t4) ^ 0x80808080u;
                }
        }
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++)
#pragma unroll
                    for (int hh = 0; hh < 2; hh++)
                        vq[mi][j][kk][2 * hh + r] =
                            h2u(__hadd2(u2h(__byte_perm(wd[j][kk][r], 0x64646464u, hh ? 0x4342 : 0x4140)), off2));
        }
    };
    unsigned ch[HMI][HNI][2];
    auto step = [&](int st, auto ph) {
        constexpr int PH = decltype(ph)::value;
        cpa_wait<NSH - 3>();
        __syncthreads();

        const uint8_t * sw = sm + (st % NSH) * STG;
        const unsigned sx = sbase + (st % NSH) * STG;
        const uint2 * ws = sws + (st & 1) * 2 * BM;
        const int p = st & 3;
        rawq(st);
        unsigned a[HMI][2][2][4]; // fp16 A fragments [mi][j][kk][0..3]
#pragma unroll
        for (int mi = 0; mi < HMI; mi++) {
        const int rw = HWR * wm + 16 * mi + g;
        const uint8_t * ra = sw + rw * SR;
        const uint8_t * rb = ra + 8 * SR;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            uint2 sA, sB;
            if constexpr (HT<TYPE>::SEP) {
                sA = ws[j * BM + rw + (FMT ? 8 * (g & 1) : 0)]; // FMT: the lane's rows A / B (rawq)
                sB = ws[j * BM + rw + (FMT ? 8 - 8 * (g & 1) : 8)];
            } else { // Q6_K: (S_j0, S_j1) in the row -> broadcast pairs (FM6: in the row's meta slot)
                const int z = (g >> 1) & 3;
                const __half2 qa = *reinterpret_cast<const __half2 *>(FM6 ? sw + F6M + 16 * rw + 8 + 4 * j : ra + 16 * (3 ^ z) + 8 + 4 * j);
                const __half2 qb = *reinterpret_cast<const __half2 *>(FM6 ? sw + F6M + 16 * (rw + 8) + 8 + 4 * j : rb + 16 * (3 ^ z) + 8 + 4 * j);
                sA = make_uint2(h2u(__low2half2(qa)), h2u(__high2half2(qa)));
                sB = make_uint2(h2u(__low2half2(qb)), h2u(__high2half2(qb)));
            }
#pragma unroll
            for (int kk = 0; kk < 2; kk++)
#pragma unroll
                for (int r = 0; r < 2; r++) {
                    const uint2 sc = r ? sB : sA;
#pragma unroll
                    for (int hh = 0; hh < 2; hh++) {
                        const __half2 v = u2h(vq[mi][j][kk][2 * hh + r]);
                        __half2 o;
                        if constexpr (TYPE == TQ4K || TYPE == TQ5K) o = __hfma2(v, u2h(sc.x), u2h(sc.y));
                        else if constexpr (TYPE == TQ6K || TYPE == TQ3K || TYPE == TIQ3S) o = __hmul2(v, u2h(kk ? sc.y : sc.x));
                        else o = __hmul2(v, u2h(sc.x));
                        a[mi][j][kk][2 * hh + r] = h2u(o);
                    }
                }
        }
        }
#pragma unroll
        for (int G = 0; G < 4; G++) {
            const int j = G >> 1, kk = G & 1;
#pragma unroll
            for (int np = 0; np < HNI / 2; np++) {
                unsigned r[4];
                ldsm_x4(r, sx + xo[G] + np * 16 * 128);
#pragma unroll
                for (int mi = 0; mi < HMI; mi++) {
                    if (G == 0 && (PH == 0 || HCH == 1)) {
                        hmma16z(ch[mi][2 * np], a[mi][j][kk], r[0], r[1]);
                        hmma16z(ch[mi][2 * np + 1], a[mi][j][kk], r[2], r[3]);
                    } else {
                        hmma16(ch[mi][2 * np], a[mi][j][kk], r[0], r[1]);
                        hmma16(ch[mi][2 * np + 1], a[mi][j][kk], r[2], r[3]);
                    }
                }
            }
            // next stages' copies and scales are issued once the tensor pipe has work queued
            if (G == 0) {
                if (st + NSH - 1 < nst) load(st + NSH - 1);
                cpa_commit();
            }
            if (G == 1 && st + 1 < nst) scales(st + 1);
        }
        if (PH == HCH - 1 || HCH == 1)
#pragma unroll
        for (int mi = 0; mi < HMI; mi++)
#pragma unroll
        for (int ni = 0; ni < HNI; ni++) {
            const float2 lo = __half22float2(u2h(ch[mi][ni][0])), hi = __half22float2(u2h(ch[mi][ni][1]));
            acc[mi][ni][0] += lo.x; acc[mi][ni][1] += lo.y; acc[mi][ni][2] += hi.x; acc[mi][ni][3] += hi.y;
        }
    };
    for (int st = 0; st < nst; st += 2) { // nst is even for every K of the model
        step(st, std::integral_constant<int, 0>());
        step(st + 1, std::integral_constant<int, 1>());
    }
    cpa_wait<0>();
    __syncthreads();
    constexpr int ELD = BM + 4;
    float * se = reinterpret_cast<float *>(sm);
#pragma unroll
    for (int mi = 0; mi < HMI; mi++)
#pragma unroll
    for (int ni = 0; ni < HNI; ni++) {
        const int r = HWR * wm + 16 * mi + g, tk = HWT * wn + 8 * ni + 2 * t4;
        const int rA = r + (FMT ? 8 * (g & 1) : 0), rB = r + (FMT ? 8 - 8 * (g & 1) : 8); // FMT: rows A / B (rawq)
        se[tk * ELD + rA] = acc[mi][ni][0];
        se[(tk + 1) * ELD + rA] = acc[mi][ni][1];
        se[tk * ELD + rB] = acc[mi][ni][2];
        se[(tk + 1) * ELD + rB] = acc[mi][ni][3];
    }
    __syncthreads();
    if constexpr (GLU) {
        for (int i = tid; i < NB * 4; i += 256) { // (token, 16-group of the 64 outputs)
            const int tk = i >> 2, q = i & 3;
            if (tok0 + tk >= ntok) continue;
            const float * gp = se + tk * ELD + 16 * q;
            __align__(16) __half hv[16];
#pragma unroll
            for (int e = 0; e < 16; e++) {
                const float gv = gp[e], v = gv / (1.0f + expf(-gv)) * gp[e + BM / 2];
                hv[hperm(e)] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
            }
            uint4 * dst = reinterpret_cast<uint4 *>(O.yh + (size_t) (tok0 + tk) * O.ldy + row0 + 16 * q);
            dst[0] = reinterpret_cast<const uint4 *>(hv)[0];
            dst[1] = reinterpret_cast<const uint4 *>(hv)[1];
        }
        return;
    }
    for (int i = tid; i < NB * BM / 4; i += 256) {
        const int tk = i / (BM / 4), c = (i % (BM / 4)) * 4;
        if (tok0 + tk >= ntok) continue;
        float4 v = *reinterpret_cast<const float4 *>(se + tk * ELD + c);
        float4 * dst = reinterpret_cast<float4 *>(O.y + (size_t) (tok0 + tk) * O.ldy + row0 + c);
        if (O.mode == OUT_ADD) {
            const float4 o = __ldcg(dst);
            v.x += o.x; v.y += o.y; v.z += o.z; v.w += o.w;
        }
        *dst = v;
    }
}

} // namespace pf
