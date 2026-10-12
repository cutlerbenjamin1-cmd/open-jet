// Persistent cooperative "megakernel" for Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6;
// 64 layers: 48 Gated-DeltaNet + 16 gated full-attention, plus one MTP block), tuned for
// RTX 3090 (sm_86, 82 SMs).
//
// One kernel launch executes an entire speculative decoding cycle:
//   main forward over T = 1 + n_draft tokens  ->  lm_head + greedy verify on device
//   -> MTP catch-up rows  ->  n_draft MTP draft steps.
// Phases inside the launch are separated by a software grid barrier.  Weights stay in their
// GGUF k-quant encodings (Q6_K / Q8_0 are re-laid-out for aligned loads, values unchanged)
// and are streamed once per cycle with int8 dp4a dot products against q8 activations.
//
// Gated-DeltaNet rollback after rejected drafts is done by lazy replay: the committed state
// only ever includes surely-accepted tokens; accepted drafts are replayed from a small ring
// of (k, v, g, beta) at the start of the next cycle, so no state snapshots are written.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <map>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

#include <unistd.h>

#include "ggml.h"
#include "gguf.h"
#include "qwen38_mk.h"

// MK_SPEC_LOG=1: the draft head keeps its top logits, and with MK_SPEC_LOG_FILE=<path> every decode
// cycle appends a JSON line (drafts + their MTP top logits, verify rows, outcome). Analysis only.
#ifndef MK_SPEC_LOG
#define MK_SPEC_LOG 0
#endif
// MK_KV_EMU=1 (accuracy experiments): MK_KV_EMU_SPEC rounds chosen K / V cache slots to a lower-bit grid (kv_store_emu)
// while keeping the q8 layout, so lower-bit KV formats can be scored per layer before the kernels support them.
#ifndef MK_KV_EMU
#define MK_KV_EMU 0
#endif

// A failed CUDA call prints its message and throws MKCudaError (the C API returns -1 with the message and resets
// the engine; the server answers with an HTTP error and keeps serving). A sticky error (the context is unusable:
// a faulting kernel, illegal address, trap, ECC, ...) cannot be recovered in-process: every later CUDA call fails
// too, so the process exits with code 4 after the message (the launcher / user restarts it). The library used to
// exit(1) on every error, which took the server down without an answer.
[[noreturn]] static void mk_cuda_fail(cudaError_t e, const char * expr, const char * file, int line) {
    char msg[768];
    snprintf(msg, sizeof msg, "CUDA error %s at %s:%d: %s", expr, file, line, cudaGetErrorString(e));
    fprintf(stderr, "%s\n", msg);
    bool sticky = false;
    switch (e) {
        case cudaErrorIllegalAddress: case cudaErrorLaunchFailure: case cudaErrorHardwareStackError:
        case cudaErrorIllegalInstruction: case cudaErrorMisalignedAddress: case cudaErrorInvalidAddressSpace:
        case cudaErrorInvalidPc: case cudaErrorLaunchTimeout: case cudaErrorECCUncorrectable: case cudaErrorAssert:
        case cudaErrorContextIsDestroyed:
            sticky = true;
            break;
        default: // anything else is sticky if it survives clearing the last error
            cudaGetLastError();
            sticky = cudaGetLastError() != cudaSuccess;
    }
    if (sticky) {
        fprintf(stderr, "[mk] the CUDA error is sticky (the CUDA context is unusable): exiting with code 4\n");
        fflush(stdout);
        fflush(stderr);
        _exit(4);
    }
    throw MKCudaError(msg);
}
#define CK(x)                                                                                   \
    do {                                                                                        \
        const cudaError_t e_ = (x);                                                             \
        if (e_ != cudaSuccess) mk_cuda_fail(e_, #x, __FILE__, __LINE__);                        \
    } while (0)

namespace {

constexpr int D = 5120, FF = 17408, VOCAB = 248320, NL = 64, NLA = 65;
constexpr int NH = 24, NKV = 4, HD = 256;
constexpr int NKH = 16, NVH = 48, HK = 128;
constexpr int QKV = 10240, VD = 6144, QG = 12288, KVD = 1024;
constexpr int MAXT = 5;
constexpr int NT = 512, NW = NT / 32;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 10000000.0f;
constexpr int CONV_RING = 8, KV_RING = 16, H_RING = 16;
constexpr int KVGB = NKH * HK + NVH * HK + 2 * NVH; // k | v | g | beta
constexpr int KVGB_V = NKH * HK, KVGB_G = KVGB_V + NVH * HK, KVGB_B = KVGB_G + NVH;
constexpr int N_ATT_SLOTS = 17; // 16 trunk attention layers + MTP block
constexpr int MAX_CH = 32;      // attention split-K chunks per kv head
constexpr int DV_EXTRA = 4096;  // draft-vocabulary rows reserved for tokens of the current sequence
constexpr int AB_SPLIT = 4;     // K-chunks per delta-net alpha/beta row
constexpr int U_D = D / 32, U_FF = FF / 32, U_VD = VD / 32, U_EH = 2 * D / 32;
constexpr int UMAX = U_FF;

constexpr int AR = 6 * MAXT;   // attention query rows per kv head (6 q heads x tokens)
constexpr int SM_XU = 4 * UMAX; // activation units (x tokens) shared memory holds; T = 5 ffn_down fills in two halves
constexpr int SM_XA = 0;
constexpr int SM_XB = SM_XA + SM_XU * 16;
constexpr int SM_XM = SM_XB + SM_XU * 16;
constexpr int SM_MISC = SM_XM + SM_XU * 8;
constexpr int ATP = 32; // attention: positions per K/V tile (two tiles in flight)
constexpr int ATT_NCH = 20;                // max split-K chunks per kv head (4 x 20 items fit one wave)
constexpr int ATT_BUCKET = ATT_NCH * ATP;  // the chunk size steps by one tile every ATT_BUCKET positions
static_assert(ATT_NCH <= MAX_CH, "attention partials buffer");
// largest T <= T_req whose rows n .. n+T-1 all see the same attention partition (see attn_dims)
inline int att_cap_T(int n, int T_req) {
    const int lim = ATT_BUCKET * ((n + ATT_BUCKET) / ATT_BUCKET) - n; // rows with n + 1 + t <= next multiple
    return T_req < lim ? T_req : lim;
}
constexpr int KVQ_ROW = 256 + 16; // staged q8 K or V row: 256 int8 values + 8 fp16 scales
// K/V tile region: fp16 cache: 2 stages of fp16 K/V tiles; q8 cache: one fp16 K/V tile + 2 stages of q8 tiles
constexpr int SM_ATTN_KV = (4 * ATP * (256 + 8) * 2 > 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW)
                               ? 4 * ATP * (256 + 8) * 2 : 2 * ATP * (256 + 8) * 2 + 4 * ATP * KVQ_ROW;
constexpr int SM_ATTN = 32 * (256 + 8) * 2 + SM_ATTN_KV + 32 * ATP * 4 + 32 * (ATP + 8) * 2 + 96 * 4; // Q, K/V, S, P, stats
constexpr int SM_TOTAL = (SM_MISC + 2048 > SM_ATTN ? SM_MISC + 2048 : SM_ATTN);
static_assert(SM_TOTAL <= 99 * 1024, "shared memory budget");

enum { T_F32 = 0, T_Q8_0 = 8, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14, T_IQ4NL = 20, T_IQ4X = 23 };
enum { MODE_PREFILL = 0, MODE_DECODE = 1, MODE_BENCH = 2 };

struct Mat {
    const uint8_t * w;
    const __half * d; // per-row block scales for repacked Q6_K / Q8_0
    int type, K, N, rb, nd;
};

struct LayerW {
    int attn;
    const float *attn_norm, *post_norm;
    // gated delta net
    Mat qkv, gate, sout;
    const float *ab, *conv, *dt, *A, *snorm; // ab: alpha | beta rows, fp32 with attn_norm folded in, or (UD, Q8_0):
    const int8_t * abq;                      // int8 rows [2 NVH][D] + fp16 scales per 32 (abd): w = (q * d) * attn_norm
    const __half * abd;
    // gated attention
    Mat q, k, v, o;
    const float *qn, *kn;
    // ffn
    Mat fg, fu, fd;
};

struct ModelW {
    LayerW L[NLA];
    Mat eh, head;
    Mat head_d;            // MTP draft head: subset of lm_head rows (== head when no subset is used)
    const int * draft_ids; // row -> token id for head_d (nullptr: identity)
    const float *enorm, *hnorm, *shnorm, *onorm;
    const uint8_t * embd;
};

// Quantised activation vector(s) in the "unit" layout consumed by the GEMV:
// unit u = (super-block b, 64-group p, half h) covers elements A=[256b+64p+16h,+16) and
// B=[256b+64p+32+16h,+16).  xa/xb hold the int8 values, d the per-32 scales of the natural
// blocks the A/B halves live in, qs the int sums of each 16-chunk, ssq per natural block x^2.
struct QBuf {
    int8_t *xa, *xb;
    uint2 * meta; // per unit: half2 (scale of A / B chunk), short2 (int sums of A / B chunk)
    float * ssq;
    int U;
};

struct Ctl {
    int n;         // position of tok[0]
    int tok[8];    // tokens fed to the main model this cycle
    int dn_c;      // delta-net committed position (state covers positions < dn_c)
    int mtp_valid; // MTP kv valid for positions < mtp_valid
    int n_out, out[8];
    int n_acc;
    int mtp_R, mtp_logits, mtp_P;
    int mtp_pos[8], mtp_tok[8];
    int tgt[8];
    // draft confidence: dconf[k] = the MTP's own softmax probability of draft k + 1 (tok[k + 1]) over the
    // draft vocabulary, dchain = their running product; nd_prod = drafts this launch produced (the
    // early-stop gate may end drafting before Launch::ndraft), nd_stop = the gate fired
    float dconf[8];
    float dchain;
    int nd_prod, nd_stop;
    int dsmp; // the drafts in tok[1..] were sampled from q' (cB.dq; speculative sampling with sampled drafts)
    int pad[4];
};

struct Bufs {
    float *x, *xm;
    QBuf qn, qo, qa, qe;
    float *z, *ab, *qbuf, *kbuf, *vbuf;
    float *conv, *kvgb, *dstate;
    void *kc, *vc;     // KV cache [slot][kv head][ctx][HD]: fp16, or int8 (q8_0) when kvq8 ...
    __half *ksc, *vsc; // ... with one fp16 scale per 32 values: [slot][kv head][ctx][HD / 32]
    int kvq8;
    // kvq8: per attention slot (16 layers + MTP) and K / V (MK_KV): kvb bits per value, 8 (q8_0) or 4 (q4: q4_0
    // rounding, 16 levels, low nibbles of a lane's word = values 0..3), kvp its values [kv head][ctx][HD * bits / 8]
    // (slots packed one after the other in kc / vc), kvs its scales [kv head][ctx][HD / 32] (= ksc / vsc + slot offset)
    int8_t * kvp[2][N_ATT_SLOTS];
    __half * kvs[2][N_ATT_SLOTS];
    int kvb[2][N_ATT_SLOTS];
    int kvr[N_ATT_SLOTS]; // K stored Hadamard-rotated (MK_KV "4r"): the queries get the same orthonormal 256-point rotation
    float * apart;
    float2 * aml;
    float *h_hist, *hm;
    const float2 * rope; // [ctx][32] (cos, sin) of pos * base^(-2i/64)
    float2 * blk_best; // per block: best (logit, id) per row, or the TOPK_MAX best when sampling
    float2 * blk_lse;  // per block: (max, sum of exp) of the draft logits (draft confidence)
    float2 * dlog;     // MK_SPEC_LOG: per draft [MAXT][TOPK_MAX + 1]: its top logits (logit, token id), then (lse, best)
    float2 * dq;       // sampled drafts: per draft [MAXT][TOPK_MAX] its proposal q' as (prob, token id), then (0, -1)
    float2 * cand;     // [MAXT][TOPK_MAX] best (logit, id) of the main lm_head rows, descending (sampling)
    float * logits;    // [MAXT][VOCAB] main lm_head logits of a scoring launch (Launch::score == 2)
    unsigned * bar;
    unsigned * cnt;
    Ctl * ctl;
    unsigned long long * prof; // optional per-barrier timestamps (block 0)
    int ctx;
#if MK_KV_EMU
    int kvemu[2][N_ATT_SLOTS]; // per slot, K / V: 0 plain q8, else the kv_store_emu mode (MK_KV_EMU_SPEC)
#endif
};

__constant__ ModelW cW;
__constant__ Bufs cB;

struct Launch {
    int mode, T, last, ndraft;
    int score; // prefill only: greedy argmax for every row (teacher-forced evaluation), no MTP; 2: also their logits
    // sampling (temp > 0; greedy otherwise), applied in llama.cpp's order: top_k, top_p, min_p, temperature
    float temp, top_p, min_p;
    int top_k; // <= TOPK_MAX
    unsigned long long seed;
    // draft early stop on the MTP's confidence f (its softmax top-1 probability, or with gate_chain the
    // running product over this launch's drafts), per draft index k: f < gate_drop[k] drops the draft and
    // stops drafting, f < gate_stop[k] keeps it and stops. All 0: off (always Launch::ndraft drafts).
    float gate_drop[4], gate_stop[4];
    int gate_chain;
    // sampling: draft from the MTP's own sampling distribution q' and verify with min(1, p / q') plus the
    // residual max(0, p - q') (distribution-exact; higher acceptance than argmax drafts at temp 1)
    int spec_smp;
};
constexpr int TOPK_MAX = 32; // candidates kept per main lm_head row when sampling

// ------------------------------------------------------------------------------------------
// small device helpers
// ------------------------------------------------------------------------------------------

// Weights are streamed once per cycle: bypass L1 and mark them evict-first in L2 so that data
// prefetched with evict-last (recurrent state, KV cache) survives the stream.
__device__ __forceinline__ unsigned long long l2_evict_first() {
    unsigned long long pol;
    asm("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;" : "=l"(pol));
    return pol;
}
__device__ __forceinline__ int4 ldw4(const void * p) {
    int4 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v4.s32 {%0,%1,%2,%3}, [%4], %5;"
        : "=r"(r.x), "=r"(r.y), "=r"(r.z), "=r"(r.w)
        : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int2 ldw2(const void * p) {
    int2 r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.v2.s32 {%0,%1}, [%2], %3;" : "=r"(r.x), "=r"(r.y) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ int ldw1(const void * p) {
    int r;
    asm("ld.global.nc.L1::no_allocate.L2::cache_hint.s32 %0, [%1], %2;" : "=r"(r) : "l"(p), "l"(l2_evict_first()));
    return r;
}
__device__ __forceinline__ void pf_l2(const void * p) { asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(p)); }
// prefetch `bytes` starting at p (128-byte lines) with the threads of the calling block
__device__ __forceinline__ void pf_block(const void * p, int bytes) {
    for (int off = threadIdx.x * 128; off < bytes; off += NT * 128) pf_l2(reinterpret_cast<const uint8_t *>(p) + off);
}
__device__ __forceinline__ float4 ldw4f(const float * p) {
    int4 r = ldw4(p);
    return make_float4(__int_as_float(r.x), __int_as_float(r.y), __int_as_float(r.z), __int_as_float(r.w));
}
__device__ __forceinline__ unsigned short ldwu16(const void * p) {
    unsigned short r;
    asm("ld.global.nc.L1::no_allocate.u16 %0, [%1];" : "=h"(r) : "l"(p));
    return r;
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }

enum {
    PH_EMBED, PH_ATTN_PROJ, PH_ATTN_CORE, PH_ATTN_COMB, PH_RESID_O, PH_DN_PROJ, PH_DN_CORE, PH_RESID_SOUT, PH_FFN_GU,
    PH_FFN_DOWN, PH_LMHEAD, PH_MTP_M0, PH_MTP_EH, PH_MTP_ATTN_PROJ, PH_MTP_ATTN_CORE, PH_MTP_ATTN_COMB, PH_MTP_RESID_O,
    PH_MTP_FFN_GU, PH_MTP_FFN_DOWN, PH_MTP_LMHEAD, PH_COUNT
};

__device__ __forceinline__ unsigned long long gtimer() {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    return t;
}

__shared__ unsigned s_bar_k; // index of the last grid barrier passed (profiling only)
extern __shared__ __align__(16) uint8_t smem[]; // dynamic shared memory of the persistent block

// Grid barrier: release-add on arrival, acquire-poll until every block of this generation arrived.
__device__ __forceinline__ void gsync(unsigned & epoch, int tag) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (cB.prof) {
            const unsigned k = epoch / gridDim.x + 1;
            if (k < 1024) cB.prof[4096 + k * 128 + blockIdx.x] = gtimer();
        }
        epoch += gridDim.x;
        asm volatile("red.release.gpu.global.add.u32 [%0], 1;" ::"l"(cB.bar) : "memory");
        unsigned v;
        do {
            asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(cB.bar) : "memory");
        } while (v < epoch);
        if (cB.prof && blockIdx.x == 0) {
            const unsigned k = epoch / gridDim.x;
            s_bar_k = k;
            if (k < 2048) cB.prof[k] = (gtimer() << 8) | (unsigned long long) tag;
        }
    }
    __syncthreads();
}

// profiling: timestamp the end of a phase's activation fill (block 0)
__device__ __forceinline__ void mark_fill() {
    if (cB.prof && blockIdx.x == 0 && threadIdx.x == 0 && s_bar_k + 1 < 2048) cB.prof[2048 + s_bar_k + 1] = gtimer();
}

// block-wide sum, result broadcast. red must hold >= NW floats.
__device__ __forceinline__ float bsum(float v, float * red) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = wsum(v);
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    float r = lane < NW ? red[lane] : 0.f;
    r = wsum(r);
    return r;
}

// lane i holds element 32*j+i of row t; quantise the natural 32-block j into the unit layout.
__device__ __forceinline__ void q32_store(const QBuf & q, int t, int j, float v, float ss, bool wss) {
    const int lane = threadIdx.x & 31;
    const float amax = wmax(fabsf(v));
    const float d = amax / 127.0f;
    const int qi = amax == 0.0f ? 0 : (int) roundf(v / d);
    const int b = j >> 3, p = (j >> 1) & 3, part = j & 1, hh = lane >> 4, byte = lane & 15;
    const int gi = t * q.U + 8 * b + 2 * p + hh;
    (part ? q.xb : q.xa)[gi * 16 + byte] = (int8_t) qi;
    int qs = qi;
    qs += __shfl_xor_sync(0xffffffffu, qs, 8);
    qs += __shfl_xor_sync(0xffffffffu, qs, 4);
    qs += __shfl_xor_sync(0xffffffffu, qs, 2);
    qs += __shfl_xor_sync(0xffffffffu, qs, 1);
    if (byte == 0) {
        reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(d);
        reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs;
    }
    if (wss) {
        const float s = wsum(ss);
        if (lane == 0) q.ssq[t * q.U + j] = s;
    }
}

struct SX {
    int4 * xa;
    int4 * xb;
    uint2 * xm; // per unit: half2 scales of the A / B 16-chunks, short2 int sums of the chunks
    int U;
};

__device__ __forceinline__ SX make_sx(uint8_t * /*sm*/, int U) {
    uint8_t * const sm = smem;
    SX s;
    s.xa = reinterpret_cast<int4 *>(sm + SM_XA);
    s.xb = reinterpret_cast<int4 *>(sm + SM_XB);
    s.xm = reinterpret_cast<uint2 *>(sm + SM_XM);
    s.U = U;
    return s;
}

__device__ __forceinline__ float * misc(uint8_t * /*sm*/) {
    uint8_t * const sm = smem; return reinterpret_cast<float *>(sm + SM_MISC); }

__device__ __forceinline__ void cp_async16(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
}

// Load T rows (starting at row t0) of a quantised activation buffer into shared memory with
// async copies.  norm: s_rs[t] receives the RMS-norm scale of row t (from the producer's
// per-block sums of squares); consumers apply it to their dot products, otherwise s_rs = 1.
template <int T>
__device__ void fill_xs(const QBuf & q, int t0, bool norm, const SX & s, float * s_rs) {
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int U = q.U;
    const int8_t * ga = q.xa + (size_t) t0 * U * 16;
    const int8_t * gb = q.xb + (size_t) t0 * U * 16;
    const uint2 * gm = q.meta + (size_t) t0 * U;
    for (int i = tid; i < T * U; i += NT) {
        cp_async16(s.xa + i, ga + (size_t) i * 16);
        cp_async16(s.xb + i, gb + (size_t) i * 16);
    }
    for (int i = tid; i < T * U / 2; i += NT) cp_async16(s.xm + 2 * i, gm + 2 * i);
    asm volatile("cp.async.commit_group;" ::: "memory");
    if (norm) { // normed consumers always read the K = D residual stream (U_D = 160 partial sums)
        if (warp < T) {
            float p[U_D / 32];
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) p[k] = __ldcg(q.ssq + (t0 + warp) * U_D + lane + 32 * k);
            float acc = 0.f;
#pragma unroll
            for (int k = 0; k < U_D / 32; k++) acc += p[k];
            acc = wsum(acc);
            if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
        }
    } else if (tid < T) {
        s_rs[tid] = 1.0f;
    }
    asm volatile("cp.async.wait_all;" ::: "memory");
    __syncthreads();
    mark_fill();
}

// Async copy of units [u0, u0 + US) of T activation rows (one K-slice) into shared memory.
template <int T, int US>
__device__ void fill_xs_part(const QBuf & q, int u0, const SX & s) {
    const int tid = threadIdx.x, U = q.U;
    __syncthreads(); // the previous slice may still be in use
    for (int i = tid; i < T * US; i += NT) {
        const int t = i / US, u = i - t * US;
        const size_t gi = (size_t) t * U + u0 + u;
        cp_async16(s.xa + i, q.xa + gi * 16);
        cp_async16(s.xb + i, q.xb + gi * 16);
    }
    for (int i = tid; i < T * US / 2; i += NT) {
        const int t = 2 * i / US, u = 2 * i - t * US;
        cp_async16(s.xm + 2 * i, q.meta + (size_t) t * U + u0 + u);
    }
    asm volatile("cp.async.commit_group;\ncp.async.wait_all;" ::: "memory");
    __syncthreads();
}

// ------------------------------------------------------------------------------------------
// quantised GEMV: R rows per warp share every activation load (per-lane partial sums;
// the caller reduces across the warp).  Lane handles units u = lane + 32 i.
// ------------------------------------------------------------------------------------------

constexpr int M4 = 0x0F0F0F0F;

__device__ __forceinline__ int gwarp() { return (threadIdx.x >> 5) * gridDim.x + blockIdx.x; }
__device__ __forceinline__ int nwarps() { return NW * gridDim.x; }

template <int T, int R>
__device__ __forceinline__ void zero_acc(float (&a)[R][T]) {
#pragma unroll
    for (int k = 0; k < R; k++)
#pragma unroll
        for (int t = 0; t < T; t++) a[k][t] = 0.f;
}

template <int T, int R>
__device__ __forceinline__ void wsum_acc(float (&a)[R][T]) {
#pragma unroll
    for (int k = 0; k < R; k++)
#pragma unroll
        for (int t = 0; t < T; t++) a[k][t] = wsum(a[k][t]);
}

template <int T, int R, int UU>
__device__ __forceinline__ void accR(const SX & s, int u, const int (&A)[R][4], const int (&B)[R][4],
                                     const float (&fA)[R], const float (&gA)[R], const float (&fB)[R],
                                     const float (&gB)[R], float (&acc)[R][T]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int ix = t * UU + u;
        const int4 va = s.xa[ix], vb = s.xb[ix];
        const uint2 mm = s.xm[ix];
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
        const int qq = (int) mm.y;
        const float sqA = dx.x * (float) (short) (qq & 0xffff), sqB = dx.y * (float) (short) (qq >> 16);
#pragma unroll
        for (int k = 0; k < R; k++) {
            int sa = __dp4a(A[k][0], va.x, 0);
            sa = __dp4a(A[k][1], va.y, sa);
            sa = __dp4a(A[k][2], va.z, sa);
            sa = __dp4a(A[k][3], va.w, sa);
            int sb = __dp4a(B[k][0], vb.x, 0);
            sb = __dp4a(B[k][1], vb.y, sb);
            sb = __dp4a(B[k][2], vb.z, sb);
            sb = __dp4a(B[k][3], vb.w, sb);
            acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb) - gA[k] * sqA - gB[k] * sqB;
        }
    }
}

__device__ __forceinline__ void q4k_scales(const int4 & h, int p, float & fA, float & gA, float & fB, float & gB) {
    const float2 dm = __half22float2(*reinterpret_cast<const __half2 *>(&h.x));
    const int sh = (p & 1) * 16;
    const unsigned y = (unsigned) h.y >> sh, z = (unsigned) h.z >> sh, w = (unsigned) h.w >> sh;
    unsigned sc0, sc1, m0, m1;
    if (p < 2) {
        sc0 = y & 63;
        sc1 = (y >> 8) & 63;
        m0 = z & 63;
        m1 = (z >> 8) & 63;
    } else {
        sc0 = (w & 0xF) | ((y >> 2) & 0x30);
        sc1 = ((w >> 8) & 0xF) | ((y >> 10) & 0x30);
        m0 = ((w >> 4) & 0xF) | ((z >> 2) & 0x30);
        m1 = ((w >> 12) & 0xF) | ((z >> 10) & 0x30);
    }
    fA = dm.x * (float) sc0;
    fB = dm.x * (float) sc1;
    gA = dm.y * (float) m0;
    gB = dm.y * (float) m1;
}

template <int T, int IT, int R>
__device__ __forceinline__ void rows_q4k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * 144;
            h[k] = ldw4(blk);
            q[k] = ldw4(blk + 16 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
            A[k][0] = q[k].x & M4; A[k][1] = q[k].y & M4; A[k][2] = q[k].z & M4; A[k][3] = q[k].w & M4;
            B[k][0] = (q[k].x >> 4) & M4; B[k][1] = (q[k].y >> 4) & M4;
            B[k][2] = (q[k].z >> 4) & M4; B[k][3] = (q[k].w >> 4) & M4;
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

template <int T, int IT, int R>
__device__ __forceinline__ void rows_q5k(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 h[R], qh[R], q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * 176;
            h[k] = ldw4(blk);
            qh[k] = ldw4(blk + 16 + 16 * hh);
            q[k] = ldw4(blk + 48 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
        const int s0 = 2 * p, s1 = 2 * p + 1;
#pragma unroll
        for (int k = 0; k < R; k++) {
            q4k_scales(h[k], p, fA[k], gA[k], fB[k], gB[k]);
            A[k][0] = (q[k].x & M4) | (((qh[k].x >> s0) & 0x01010101) << 4);
            A[k][1] = (q[k].y & M4) | (((qh[k].y >> s0) & 0x01010101) << 4);
            A[k][2] = (q[k].z & M4) | (((qh[k].z >> s0) & 0x01010101) << 4);
            A[k][3] = (q[k].w & M4) | (((qh[k].w >> s0) & 0x01010101) << 4);
            B[k][0] = ((q[k].x >> 4) & M4) | (((qh[k].x >> s1) & 0x01010101) << 4);
            B[k][1] = ((q[k].y >> 4) & M4) | (((qh[k].y >> s1) & 0x01010101) << 4);
            B[k][2] = ((q[k].z >> 4) & M4) | (((qh[k].z >> s1) & 0x01010101) << 4);
            B[k][3] = ((q[k].w >> 4) & M4) | (((qh[k].w >> s1) & 0x01010101) << 4);
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q6_K, 210 B per 256 weights like the GGUF block, in two streams per row:
//   rows (Mat::w, Q6_BLK = 192 B per block): qlo[128] (q4k-style nibbles) | qhi[64] (8 B per unit);
//   meta (Mat::d, 18 B per block, in pairs of blocks: sc[16] even | sc[16] odd | d even | d odd, 36 B).
// A unit reads its 4 scales sc[4p..4p+3] and the pair's two d (one 4-byte load each).
constexpr int Q6_BLK = 192, Q6_META = 18;
template <int T, int IT, int R>
__device__ __forceinline__ void rows_q6k(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 q[R];
        int2 hb[R];
        int sc4[R], dd[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * Q6_BLK;
            const uint8_t * mp = reinterpret_cast<const uint8_t *>(dp[k]) + (b >> 1) * (2 * Q6_META);
            q[k] = ldw4(blk + 32 * p + 16 * hh);
            hb[k] = ldw2(blk + 128 + 8 * (u & 7));
            sc4[k] = ldw1(mp + 16 * (b & 1) + 4 * p);
            dd[k] = ldw1(mp + 32);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) ((unsigned) dd[k] >> (16 * (b & 1)))));
            fA[k] = d * (float) (int8_t) (sc4[k] >> (8 * hh));
            fB[k] = d * (float) (int8_t) (sc4[k] >> (16 + 8 * hh));
            gA[k] = 32.0f * fA[k];
            gB[k] = 32.0f * fB[k];
            const unsigned hx = (unsigned) hb[k].x, hy = (unsigned) hb[k].y;
            A[k][0] = (q[k].x & M4) | (int) (((hx >> 0) & 0x03030303u) << 4);
            A[k][1] = (q[k].y & M4) | (int) (((hx >> 2) & 0x03030303u) << 4);
            A[k][2] = (q[k].z & M4) | (int) (((hx >> 4) & 0x03030303u) << 4);
            A[k][3] = (q[k].w & M4) | (int) (((hx >> 6) & 0x03030303u) << 4);
            B[k][0] = ((q[k].x >> 4) & M4) | (int) (((hy >> 0) & 0x03030303u) << 4);
            B[k][1] = ((q[k].y >> 4) & M4) | (int) (((hy >> 2) & 0x03030303u) << 4);
            B[k][2] = ((q[k].z >> 4) & M4) | (int) (((hy >> 4) & 0x03030303u) << 4);
            B[k][3] = ((q[k].w >> 4) & M4) | (int) (((hy >> 6) & 0x03030303u) << 4);
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q8_0: per unit 32 B (A 16 int8 | B 16 int8), scales per natural 32-block
template <int T, int IT, int R>
__device__ __forceinline__ void rows_q80(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i;
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const int4 a = ldw4(rp[k] + 32 * u), bq = ldw4(rp[k] + 32 * u + 16);
            const int dd = ldw1(dp[k] + 2 * (u >> 1));
            const float2 df = __half22float2(*reinterpret_cast<const __half2 *>(&dd));
            A[k][0] = a.x; A[k][1] = a.y; A[k][2] = a.z; A[k][3] = a.w;
            B[k][0] = bq.x; B[k][1] = bq.y; B[k][2] = bq.z; B[k][3] = bq.w;
            fA[k] = df.x; fB[k] = df.y; gA[k] = 0.f; gB[k] = 0.f;
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// kvalues_iq4nl as int8, four per word
constexpr unsigned IQ4_T0 = 0xBFAD9881u, IQ4_T1 = 0xF6EADDCFu, IQ4_T2 = 0x26190D01u, IQ4_T3 = 0x71594535u;
// The 8 4-bit indices of word q -> table bytes of its low nibbles (lo) and high nibbles (hi), four
// per int (llama.cpp's get_int_from_table_16). __byte_perm reads only bits 0-2 of each selector nibble.
__device__ __forceinline__ void iq4_lut(unsigned q, unsigned t0, unsigned t1, unsigned t2, unsigned t3, int & lo, int & hi) {
    const unsigned sel = 0x32103210u | ((q & 0x88888888u) >> 1);
    const unsigned a = __byte_perm(__byte_perm(t0, t1, q), __byte_perm(t2, t3, q), sel);
    const unsigned b = __byte_perm(__byte_perm(t0, t1, q >> 16), __byte_perm(t2, t3, q >> 16), sel >> 16);
    lo = (int) __byte_perm(a, b, 0x6420);
    hi = (int) __byte_perm(a, b, 0x7531);
}

// accR for signed weights (no min term)
template <int T, int R, int UU>
__device__ __forceinline__ void accR0(const SX & s, int u, const int (&A)[R][4], const int (&B)[R][4], const float (&fA)[R],
                                      const float (&fB)[R], float (&acc)[R][T]) {
#pragma unroll
    for (int t = 0; t < T; t++) {
        const int ix = t * UU + u;
        const int4 va = s.xa[ix], vb = s.xb[ix];
        const uint2 mm = s.xm[ix];
        const float2 dx = __half22float2(*reinterpret_cast<const __half2 *>(&mm.x));
#pragma unroll
        for (int k = 0; k < R; k++) {
            int sa = __dp4a(A[k][0], va.x, 0);
            sa = __dp4a(A[k][1], va.y, sa);
            sa = __dp4a(A[k][2], va.z, sa);
            sa = __dp4a(A[k][3], va.w, sa);
            int sb = __dp4a(B[k][0], vb.x, 0);
            sb = __dp4a(B[k][1], vb.y, sb);
            sb = __dp4a(B[k][2], vb.z, sb);
            sb = __dp4a(B[k][3], vb.w, sb);
            acc[k][t] += fA[k] * (dx.x * (float) sa) + fB[k] * (dx.y * (float) sb);
        }
    }
}

// IQ4_XS re-laid-out at load in pairs of 256-weight blocks, 272 B per pair (repack_iq4xs): the two GGUF block
// headers (fp16 d | u16 scales_h | u8 scales_l[4]), then each block's 128 index bytes in Q4_K order.
// A weight is d * (ls - 32) * kvalues_iq4nl[index], ls the 6-bit scale of its 32-weight sub-block.
template <int T, int IT, int R>
__device__ __forceinline__ void rows_iq4x(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int2 h[R];
        int4 q[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * pr = rp[k] + (b >> 1) * 272;
            h[k] = ldw2(pr + 8 * (b & 1));
            q[k] = ldw4(pr + 16 + 128 * (b & 1) + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], fB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) (h[k].x & 0xffff)));
            const unsigned sl = ((unsigned) h[k].y >> (8 * p)) & 0xffu, sh = ((unsigned) h[k].x >> (16 + 4 * p)) & 0xfu;
            fA[k] = d * (float) ((int) ((sl & 0xfu) | ((sh & 3u) << 4)) - 32);
            fB[k] = d * (float) ((int) ((sl >> 4) | ((sh >> 2) << 4)) - 32);
            iq4_lut(q[k].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][0], B[k][0]);
            iq4_lut(q[k].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][1], B[k][1]);
            iq4_lut(q[k].z, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][2], B[k][2]);
            iq4_lut(q[k].w, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][3], B[k][3]);
        }
        accR0<T, R, 32 * IT>(s, u, A, B, fA, fB, acc);
    }
}

// IQ4_NL re-laid-out at load into 144-byte blocks of 256 weights (repack_iq4nl): the fp16 scales d of the eight
// 32-weight blocks, then the 128 index bytes in Q4_K order (as IQ4_XS).  A weight is d * kvalues_iq4nl[index].
// Same integer products and the same float arithmetic (accR with zero mins) as the Q8_0 path these tensors were
// re-encoded for before, so the results are bit-identical to it at 0.53x the bytes.
template <int T, int IT, int R>
__device__ __forceinline__ void rows_iq4nl(const uint8_t * const (&rp)[R], const SX & s, float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 4) : IT)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
        int4 q[R];
        int dd[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * 144;
            dd[k] = ldw1(blk + 4 * p);
            q[k] = ldw4(blk + 16 + 32 * p + 16 * hh);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float2 df = __half22float2(*reinterpret_cast<const __half2 *>(&dd[k]));
            fA[k] = df.x; fB[k] = df.y; gA[k] = 0.f; gB[k] = 0.f;
            iq4_lut(q[k].x, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][0], B[k][0]);
            iq4_lut(q[k].y, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][1], B[k][1]);
            iq4_lut(q[k].z, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][2], B[k][2]);
            iq4_lut(q[k].w, IQ4_T0, IQ4_T1, IQ4_T2, IQ4_T3, A[k][3], B[k][3]);
        }
        accR<T, R, 32 * IT>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// rows row0 .. row0+R-1 of m, weight type fixed at compile time
// u0: first unit of a K-slice (multiple of 32) when the activations in s hold only that slice
template <int TYPE>
__device__ __forceinline__ int unit_bytes(int u0) {
    return TYPE == T_Q4K || TYPE == T_IQ4NL ? (u0 >> 3) * 144
         : TYPE == T_IQ4X                 ? (u0 >> 3) * 136 // u0 is a multiple of 32 units: whole block pairs
         : TYPE == T_Q5K                  ? (u0 >> 3) * 176
         : TYPE == T_Q6K                  ? (u0 >> 3) * Q6_BLK
                                          : 32 * u0;
}
// R rows of one weight type given their row pointers (and, for Q8_0, their scale rows)
template <int T, int IT, int R, int TYPE>
__device__ __forceinline__ void rows_t(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                       float (&acc)[R][T]) {
    if constexpr (TYPE == T_Q4K) rows_q4k<T, IT, R>(rp, s, acc);
    else if constexpr (TYPE == T_Q5K) rows_q5k<T, IT, R>(rp, s, acc);
    else if constexpr (TYPE == T_Q6K) rows_q6k<T, IT, R>(rp, dp, s, acc);
    else if constexpr (TYPE == T_IQ4X) rows_iq4x<T, IT, R>(rp, s, acc);
    else if constexpr (TYPE == T_IQ4NL) rows_iq4nl<T, IT, R>(rp, s, acc);
    else rows_q80<T, IT, R>(rp, dp, s, acc);
}
template <int T, int IT, int R, int TYPE>
__device__ __forceinline__ void mrows(const Mat & m, int row0, const SX & s, float (&acc)[R][T], int u0 = 0) {
    const uint8_t * rp[R];
    const __half * dp[R];
#pragma unroll
    for (int k = 0; k < R; k++) {
        rp[k] = m.w + (size_t) (row0 + k) * m.rb + unit_bytes<TYPE>(u0);
        dp[k] = m.d + (size_t) (row0 + k) * m.nd + (TYPE == T_Q6K ? (u0 >> 3) * (Q6_META / 2) : u0); // Q6_K: meta, Q8_0: d
    }
    rows_t<T, IT, R, TYPE>(rp, dp, s, acc);
}

// Row pairs [lo, hi) of a single-matrix item range, distributed over all warps of the grid.
// out(row0, acc) receives the reduced sums for rows row0, row0 + 1.
template <int T, int IT, int TYPE, class Out>
__device__ __noinline__ void gemv_pairs(const Mat m, int lo, int hi, Out out) {
    const SX sx = make_sx(nullptr, 32 * IT); // activations already in shared memory
    const int W = nwarps(), gw = gwarp();
    const int first = lo + (((gw - lo) % W) + W) % W;
    for (int it = first; it < hi; it += W) {
        const int r = 2 * (it - lo);
        float a[2][T];
        zero_acc<T, 2>(a);
        mrows<T, IT, 2, TYPE>(m, r, sx, a);
        wsum_acc<T, 2>(a);
        out(r, a);
    }
}

// Runtime dispatch on a matrix's type (uniform across the grid). Plain switches: a capturing
// lambda would take the address of the caller's locals and push them to local memory.
template <int T, int IT, class Out>
__device__ __forceinline__ void gemv_pairs_any(const Mat & m, int lo, int hi, Out out) {
    switch (m.type) {
        case T_Q4K: gemv_pairs<T, IT, T_Q4K>(m, lo, hi, out); break;
        case T_Q5K: gemv_pairs<T, IT, T_Q5K>(m, lo, hi, out); break;
        case T_Q6K: gemv_pairs<T, IT, T_Q6K>(m, lo, hi, out); break;
        case T_Q8_0: gemv_pairs<T, IT, T_Q8_0>(m, lo, hi, out); break;
        case T_IQ4NL: gemv_pairs<T, IT, T_IQ4NL>(m, lo, hi, out); break;
        default: gemv_pairs<T, IT, T_IQ4X>(m, lo, hi, out); break;
    }
}


// Prefetch the first `bytes` of rows row0..row0+n-1 into L2 so the DRAM stream of a phase starts
// while its activations are still being loaded into shared memory.
__device__ __forceinline__ void pf_rows(const Mat & m, int row0, int n, int bytes) {
    const int lane = threadIdx.x & 31;
    for (int k = 0; k < n; k++) {
        const uint8_t * p = m.w + (size_t) (row0 + k) * m.rb;
        for (int off = lane * 128; off < bytes; off += 32 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + off));
    }
}

// lanes 0-15 quantise 16-chunk c of row t, lanes 16-31 the same chunk of row t + 1 (if valid)
__device__ __forceinline__ void q16_store(const QBuf & q, int t, int c, float v, bool valid) {
    const int lane = threadIdx.x & 31;
    float amax = fabsf(v);
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 8));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    const float d = amax / 127.0f;
    const int qi = amax == 0.0f ? 0 : (int) roundf(v / d);
    int qs = qi;
    qs += __shfl_xor_sync(0xffffffffu, qs, 8);
    qs += __shfl_xor_sync(0xffffffffu, qs, 4);
    qs += __shfl_xor_sync(0xffffffffu, qs, 2);
    qs += __shfl_xor_sync(0xffffffffu, qs, 1);
    if (!valid) return;
    const int b = c >> 4, p = (c >> 2) & 3, part = (c >> 1) & 1, hh = c & 1, byte = lane & 15;
    const int gi = t * q.U + 8 * b + 2 * p + hh;
    (part ? q.xb : q.xa)[gi * 16 + byte] = (int8_t) qi;
    if (byte == 0) {
        reinterpret_cast<__half *>(q.meta + gi)[part] = __float2half_rn(d);
        reinterpret_cast<short *>(q.meta + gi)[2 + part] = (short) qs;
    }
}

template <int T>
__device__ __forceinline__ void wsum_all(float (&a)[T]) {
#pragma unroll
    for (int t = 0; t < T; t++) a[t] = wsum(a[t]);
}

// ------------------------------------------------------------------------------------------
// phases
// ------------------------------------------------------------------------------------------

__device__ __forceinline__ float dequant_q4k(const uint8_t * row, int e) {
    const int b = e >> 8, w = e & 255, sb = w >> 5, l = w & 31;
    const uint8_t * blk = row + b * 144;
    const float d = __half2float(*reinterpret_cast<const __half *>(blk));
    const float dmin = __half2float(*reinterpret_cast<const __half *>(blk + 2));
    const uint8_t * sc = blk + 4;
    int s, m;
    if (sb < 4) {
        s = sc[sb] & 63;
        m = sc[sb + 4] & 63;
    } else {
        s = (sc[sb + 4] & 0xF) | ((sc[sb - 4] >> 6) << 4);
        m = (sc[sb + 4] >> 4) | ((sc[sb] >> 6) << 4);
    }
    const uint8_t qb = blk[16 + 32 * (sb >> 1) + l];
    const int qv = (sb & 1) ? (qb >> 4) : (qb & 0xF);
    return d * (float) s * (float) qv - dmin * (float) m;
}

constexpr int EMBD_RB = D / 256 * 144; // bytes of one Q4_K token_embd row

// token_embd row -> shared memory in one coalesced read: the table is in mapped host memory, where
// every dependent access pays a PCIe round trip
__device__ __forceinline__ const uint8_t * stage_embd_row(int tok, uint8_t * dst) {
    if (threadIdx.x < EMBD_RB / 16)
        reinterpret_cast<uint4 *>(dst)[threadIdx.x] = reinterpret_cast<const uint4 *>(cW.embd + (size_t) tok * EMBD_RB)[threadIdx.x];
    __syncthreads();
    return dst;
}

// token embedding rows -> x, plus quantised attn_norm input for layer 0
template <int T>
__device__ __noinline__ void embed_prologue() {
    const int t = blockIdx.x;
    if (t >= T) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tok = ((volatile Ctl *) cB.ctl)->tok[t];
    const uint8_t * row = stage_embd_row(tok, smem);
    const float * w0 = cW.L[0].attn_norm;
    for (int j = warp; j < U_D; j += NW) {
        const int e = 32 * j + lane;
        const float v = dequant_q4k(row, e);
        cB.x[t * D + e] = v;
        q32_store(cB.qn, t, j, v * w0[e], v * v, true);
    }
}

// T = 5 ffn_down K-half, out of line: inlined, resid32<5,17,*> spilled 136-188 B into this loop.
template <int T, int IT, int TYPE>
__device__ __noinline__ void half_rows(const Mat W, int row0, const SX s, float (&a)[2][T], int u0) {
    float b[2][T];
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
        for (int t = 0; t < T; t++) b[k][t] = a[k][t];
    mrows<T, IT, 2, TYPE>(W, row0, s, b, u0);
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
        for (int t = 0; t < T; t++) a[k][t] = b[k][t];
}

// Output rows (32 consecutive rows per block item) with optional residual add; the new
// residual value is re-quantised for the next RMS-normed consumer (weight wnext).
template <int T, int IT, int TYPE>
__device__ __noinline__ void resid32(const Mat W, const QBuf in, float * x, bool add, const float * wnext, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(sm);
    float * s_res = misc(sm) + 8;
    if (blockIdx.x < D / 32) pf_rows(W, 32 * blockIdx.x + 2 * warp, 2, W.rb / IT);
    if (threadIdx.x < 4 && blockIdx.x + threadIdx.x * gridDim.x < D / 32) pf_l2(wnext + 32 * (blockIdx.x + threadIdx.x * gridDim.x));
    auto epilogue = [&](int j, float (&a)[2][T]) {
        wsum_acc<T, 2>(a);
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < T; t++) {
                s_res[t * 32 + 2 * warp] = a[0][t];
                s_res[t * 32 + 2 * warp + 1] = a[1][t];
            }
        }
        __syncthreads();
        if (warp < T) {
            const int t = warp, row = 32 * j + lane;
            float xv = s_res[t * 32 + lane];
            if (add) xv += __ldcg(x + t * D + row);
            x[t * D + row] = xv;
            q32_store(cB.qn, t, j, xv * wnext[row], xv * xv, true);
        }
        __syncthreads();
    };
    if constexpr (T * 32 * IT > SM_XU) {
        // The activations do not fit (T = 5 ffn_down): fill one K-half at a time.  A block's two row
        // groups are processed one after the other; the first group's per-lane partial sums of the
        // first half are parked in shared memory while the second group is computed.
        constexpr int U0 = 32 * ((IT + 1) / 2), U1 = 32 * IT - U0;
        static_assert(T * U0 * 40 + 2 * T * NT * 4 <= SM_MISC, "T = 5 ffn_down shared memory");
        auto sxh = [&](int US) {
            SX r;
            r.xa = reinterpret_cast<int4 *>(sm);
            r.xb = reinterpret_cast<int4 *>(sm + T * US * 16);
            r.xm = reinterpret_cast<uint2 *>(sm + 2 * T * US * 16);
            r.U = US;
            return r;
        };
        float * park = reinterpret_cast<float *>(sm + T * U0 * 40); // [2 * T][NT]
        const SX s0 = sxh(U0), s1 = sxh(U1);
        for (int j0 = blockIdx.x; j0 < D / 32; j0 += 2 * gridDim.x) {
            const int j1 = j0 + gridDim.x;
            fill_xs_part<T, U0>(in, 0, s0);
            if (j0 == blockIdx.x) mark_fill();
            float a[2][T];
            zero_acc<T, 2>(a);
            half_rows<T, U0 / 32, TYPE>(W, 32 * j0 + 2 * warp, s0, a, 0);
#pragma unroll
            for (int t = 0; t < T; t++) {
                park[t * NT + threadIdx.x] = a[0][t];
                park[(T + t) * NT + threadIdx.x] = a[1][t];
            }
            if (j1 < D / 32) {
                zero_acc<T, 2>(a);
                half_rows<T, U0 / 32, TYPE>(W, 32 * j1 + 2 * warp, s0, a, 0);
            }
            fill_xs_part<T, U1>(in, U0, s1);
            if (j1 < D / 32) {
                half_rows<T, U1 / 32, TYPE>(W, 32 * j1 + 2 * warp, s1, a, U0);
                epilogue(j1, a);
            }
#pragma unroll
            for (int t = 0; t < T; t++) {
                a[0][t] = park[t * NT + threadIdx.x];
                a[1][t] = park[(T + t) * NT + threadIdx.x];
            }
            half_rows<T, U1 / 32, TYPE>(W, 32 * j0 + 2 * warp, s1, a, U0);
            epilogue(j0, a);
        }
    } else {
        const SX sx = make_sx(sm, in.U);
        fill_xs<T>(in, 0, false, sx, s_rs);
        for (int j = blockIdx.x; j < D / 32; j += gridDim.x) {
            float a[2][T];
            zero_acc<T, 2>(a);
            mrows<T, IT, 2, TYPE>(W, 32 * j + 2 * warp, sx, a);
            epilogue(j, a);
        }
    }
}

template <int T, int IT>
__device__ __forceinline__ void resid_any(const Mat & W, const QBuf & in, float * x, bool add, const float * wnext, uint8_t * sm) {
    switch (W.type) {
        case T_Q4K: resid32<T, IT, T_Q4K>(W, in, x, add, wnext, sm); break;
        case T_Q5K: resid32<T, IT, T_Q5K>(W, in, x, add, wnext, sm); break;
        case T_Q6K: resid32<T, IT, T_Q6K>(W, in, x, add, wnext, sm); break;
        case T_Q8_0: resid32<T, IT, T_Q8_0>(W, in, x, add, wnext, sm); break;
        case T_IQ4NL: resid32<T, IT, T_IQ4NL>(W, in, x, add, wnext, sm); break;
        default: resid32<T, IT, T_IQ4X>(W, in, x, add, wnext, sm); break;
    }
}

// FFN gate/up: warp item = 16 consecutive rows of both matrices (4 rows in flight sharing the
// activations) -> silu(g)*u quantised per 16-chunk for the down projection.
template <int T>
__device__ __noinline__ void ffn_gu_q4k(const LayerW & W, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    const float * s_rs = misc(sm);
    if (gwarp() < FF / 16) {
        pf_rows(W.fg, 16 * gwarp(), 2, 576);
        pf_rows(W.fu, 16 * gwarp(), 2, 576);
    }
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const uint8_t * gw = W.fg.w;
    const uint8_t * uw = W.fu.w;
    const int rb = W.fg.rb;
    for (int j = gwarp(); j < FF / 16; j += nwarps()) {
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = 0.f;
        constexpr int RG = T >= 3 ? 1 : 2; // rows per matrix per step (fewer at large T: register budget)
#pragma unroll 1
        for (int r = 0; r < 16; r += RG) {
            const int row = 16 * j + r;
            const uint8_t * rp[2 * RG];
#pragma unroll
            for (int k = 0; k < RG; k++) {
                rp[k] = gw + (size_t) (row + k) * rb;
                rp[RG + k] = uw + (size_t) (row + k) * rb;
            }
            float a[2 * RG][T];
            zero_acc<T, 2 * RG>(a);
            rows_q4k<T, 5, 2 * RG>(rp, sx, a);
            wsum_acc<T, 2 * RG>(a);
#pragma unroll
            for (int k = 0; k < RG; k++)
                if (lane == r + k) {
#pragma unroll
                    for (int t = 0; t < T; t++) act[t] = silu(a[k][t] * s_rs[t]) * (a[RG + k][t] * s_rs[t]);
                }
        }
#pragma unroll
        for (int t = 0; t < T; t += 2) {
            const float other = __shfl_sync(0xffffffffu, act[t + 1 < T ? t + 1 : t], lane & 15);
            const float v = lane < 16 ? act[t] : other;
            q16_store(cB.qa, lane < 16 ? t : t + 1, j, v, lane < 16 || t + 1 < T);
        }
    }
}

template <int T> struct LaneT { float v[T]; };

// Rows 16 j .. 16 j + 15 of m against the normed activations in shared memory: lane r gets row
// 16 j + r (lanes 16-31: 0). As many rows per step as ffn_gu_q4k shares activations across.
template <int T, int TYPE>
__device__ __noinline__ LaneT<T> rows16(const Mat m, int j) {
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(nullptr, U_D);
    const float * s_rs = misc(nullptr);
    constexpr int R = T >= 3 ? 2 : 4;
    LaneT<T> res;
#pragma unroll
    for (int t = 0; t < T; t++) res.v[t] = 0.f;
#pragma unroll 1
    for (int r = 0; r < 16; r += R) {
        float a[R][T];
        zero_acc<T, R>(a);
        mrows<T, 5, R, TYPE>(m, 16 * j + r, sx, a);
        wsum_acc<T, R>(a);
#pragma unroll
        for (int k = 0; k < R; k++)
            if (lane == r + k) {
#pragma unroll
                for (int t = 0; t < T; t++) res.v[t] = a[k][t] * s_rs[t];
            }
    }
    return res;
}

template <int T>
__device__ __forceinline__ LaneT<T> rows16_any(const Mat & m, int j) {
    switch (m.type) {
        case T_Q4K: return rows16<T, T_Q4K>(m, j);
        case T_Q5K: return rows16<T, T_Q5K>(m, j);
        case T_Q6K: return rows16<T, T_Q6K>(m, j);
        case T_Q8_0: return rows16<T, T_Q8_0>(m, j);
        case T_IQ4NL: return rows16<T, T_IQ4NL>(m, j);
        default: return rows16<T, T_IQ4X>(m, j);
    }
}

// ffn_gu_q4k for any other gate / up types (each may be any type): gate and up rows of a warp item
// are computed by one call per matrix, so each weight type gets its own register allocation.
template <int T>
__device__ __noinline__ void ffn_gu_any(const LayerW & W, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    if (gwarp() < FF / 16) {
        pf_rows(W.fg, 16 * gwarp(), 2, 576);
        pf_rows(W.fu, 16 * gwarp(), 2, 576);
    }
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    for (int j = gwarp(); j < FF / 16; j += nwarps()) {
        const LaneT<T> g = rows16_any<T>(W.fg, j);
        const LaneT<T> u = rows16_any<T>(W.fu, j);
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = lane < 16 ? silu(g.v[t]) * u.v[t] : 0.f;
#pragma unroll
        for (int t = 0; t < T; t += 2) {
            const float other = __shfl_sync(0xffffffffu, act[t + 1 < T ? t + 1 : t], lane & 15);
            const float v = lane < 16 ? act[t] : other;
            q16_store(cB.qa, lane < 16 ? t : t + 1, j, v, lane < 16 || t + 1 < T);
        }
    }
}

template <int T>
__device__ __forceinline__ void ffn_gu(const LayerW & W, uint8_t * sm) {
    if (W.fg.type == T_Q4K && W.fu.type == T_Q4K) ffn_gu_q4k<T>(W, sm);
    else ffn_gu_any<T>(W, sm);
}

template <int T>
__device__ __forceinline__ void store_lane_t(float * base, int stride, int idx, const float (&a)[T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int t = 0; t < T; t++)
        if (lane == t) base[t * stride + idx] = a[t];
}

// a[k][t] -> base[t * stride + idx + k]
template <int T>
__device__ __forceinline__ void store_pair_t(float * base, int stride, int idx, const float (&a)[2][T]) {
    const int lane = threadIdx.x & 31;
    const float * s_rs = misc(nullptr);
#pragma unroll
    for (int t = 0; t < T; t++)
        if ((lane >> 1) == t) base[t * stride + idx + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * s_rs[t];
}

// delta-net input projections: alpha/beta (fp32), qkv -> conv ring, z
template <int T>
__device__ __noinline__ void dn_proj(const LayerW & W, int dl, int n, int dn_c, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    float * s_rs = misc(sm);
    {
        const int gw = gwarp(), W_ = nwarps(), lo = AB_SPLIT * 2 * NVH;
        const int first = lo + (((gw - lo) % W_) + W_) % W_;
        if (first < lo + QKV / 2) pf_rows(W.qkv, 2 * (first - lo), 2, 1024);
    }
    fill_xs<T>(cB.qn, 0, true, sx, s_rs);
    float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;
    const int n_ab = AB_SPLIT * 2 * NVH, n_qkv = QKV / 2, n_z = VD / 2;
    // alpha / beta: fp32 rows with the norm weight folded in (dot with the raw residual), each
    // row split into AB_SPLIT K-chunks so these heavier items do not form a tail
    for (int it = gwarp(); it < n_ab; it += nwarps()) {
        const int row = it % (2 * NVH), ch = it / (2 * NVH);
        float acc[T];
#pragma unroll
        for (int t = 0; t < T; t++) acc[t] = 0.f;
        if (W.abq) { // Q8_0 rows: the folded fp32 weights (q * d) * attn_norm, formed on the fly (same values)
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
    gemv_pairs_any<T, 5>(W.qkv, n_ab, n_ab + n_qkv, [=](int r, const float (&a)[2][T]) {
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int t = 0; t < T; t++)
            if ((lane >> 1) == t)
                conv[((n + t) % CONV_RING) * QKV + r + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * misc(nullptr)[t];
    });
    gemv_pairs_any<T, 5>(W.gate, n_ab + n_qkv, n_ab + n_qkv + n_z,
                         [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.z, VD, r, a); });
    // Warm L2 with what dn_core will read, while the other blocks are still projecting:
    // the head's recurrent state (64 KB), its conv weights, older conv rows and replay entries.
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

// delta-net core: one block per value head (48 blocks).  conv1d+silu, l2-norm, gated delta
// rule (with replay of pending accepted tokens), gated rms-norm, quantise for ssm_out.
template <int T>
__device__ __noinline__ void dn_core(const LayerW & W, int dl, int n, int dn_c, int n_commit, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int h = blockIdx.x;
    if (h >= NVH) return;
    const int kh = h % NKH;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // thread owns state column `col`, rows [32 r, 32 r + 32)  (S stored as [col][row])
    const int col = tid >> 2, r = tid & 3;
    constexpr int NE = T + MAXT - 1;     // max events: up to MAXT - 1 replayed accepted drafts + T new tokens
    constexpr int QS = 4 * 36;           // q / k with a 4-float pad per 32-row quarter (bank spread)
    float * s_q = reinterpret_cast<float *>(sm); // [NE][QS]
    float * s_k = s_q + NE * QS;                 // [NE][QS]
    float * s_v = s_k + NE * QS;                 // [NE][128]
    float * s_o = s_v + NE * 128;                // [T][128]
    float * s_z = s_o + T * 128;                 // [T][128]
    float * s_gb = s_z + T * 128;                // [NE][2]
    float * s_raw = s_gb + 32;                   // [T][384] conv outputs (silu), 16-byte aligned

    float * Sg = cB.dstate + ((size_t) dl * NVH + h) * 128 * 128 + col * 128 + 32 * r;
    float4 S[8];
#pragma unroll
    for (int i = 0; i < 8; i++) S[i] = __ldcg(reinterpret_cast<const float4 *>(Sg) + i);

    const int np = n - dn_c; // replayed (accepted, not yet committed) tokens
    float * ring = cB.kvgb + (size_t) dl * KV_RING * KVGB;
    const float * conv = cB.conv + (size_t) dl * CONV_RING * QKV;

    // ---- prologue: every input of every event, one round of loads ----
    if (tid < 384) {
        const int ch = tid < 128 ? kh * 128 + tid : (tid < 256 ? 2048 + kh * 128 + (tid - 128) : 4096 + h * 128 + (tid - 256));
        const float4 w = *reinterpret_cast<const float4 *>(W.conv + ch * 4);
        float raw[T + 3];
#pragma unroll
        for (int j = 0; j < T + 3; j++) {
            const int p = n - 3 + j;
            raw[j] = p >= 0 ? __ldcg(conv + (p % CONV_RING) * QKV + ch) : 0.f;
        }
#pragma unroll
        for (int t = 0; t < T; t++)
            s_raw[t * 384 + tid] = silu(w.x * raw[t] + w.y * raw[t + 1] + w.z * raw[t + 2] + w.w * raw[t + 3]);
    } else if (tid < 384 + T) {
        const int t = tid - 384;
        float al = 0.f, be = 0.f;
#pragma unroll
        for (int c = 0; c < AB_SPLIT; c++) {
            al += __ldcg(cB.ab + (c * MAXT + t) * 2 * NVH + h);
            be += __ldcg(cB.ab + (c * MAXT + t) * 2 * NVH + NVH + h);
        }
        float sp = al + W.dt[h];
        sp = sp > 20.0f ? sp : logf(1.0f + expf(sp));
        s_gb[2 * (np + t)] = expf(sp * W.A[h]);
        s_gb[2 * (np + t) + 1] = 1.0f / (1.0f + expf(-be));
    } else if (tid >= 448 && tid < 448 + np) {
        const int e = tid - 448;
        const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB;
        s_gb[2 * e] = __ldcg(slot + KVGB_G + h);
        s_gb[2 * e + 1] = __ldcg(slot + KVGB_B + h);
    }
    for (int i = tid; i < np * 128; i += NT) { // replayed k / v
        const int e = i >> 7, c = i & 127;
        const float * slot = ring + (size_t) ((dn_c + e) % KV_RING) * KVGB;
        s_k[e * QS + (c >> 5) * 36 + (c & 31)] = __ldcg(slot + kh * 128 + c);
        s_v[e * 128 + c] = __ldcg(slot + KVGB_V + h * 128 + c);
    }
    for (int i = tid; i < T * 128; i += NT) s_z[i] = __ldcg(cB.z + (i >> 7) * VD + h * 128 + (i & 127));
    __syncthreads();

    // l2-normalise q and k of the new tokens (warp 2t: q of token t, warp 2t+1: k), copy v
    if (warp < 2 * T) {
        const int t = warp >> 1, isk = warp & 1;
        const float4 v = *reinterpret_cast<const float4 *>(s_raw + t * 384 + isk * 128 + lane * 4);
        const float ss = wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w);
        float sc = rsqrtf(fmaxf(ss, EPS * EPS));
        if (!isk) sc *= 0.08838834764831845f; // 1/sqrt(128)
        float * dst = (isk ? s_k : s_q) + (np + t) * QS + (lane >> 3) * 36 + (lane & 7) * 4;
        *reinterpret_cast<float4 *>(dst) = make_float4(v.x * sc, v.y * sc, v.z * sc, v.w * sc);
    }
    for (int i = tid; i < T * 128; i += NT) s_v[(np + (i >> 7)) * 128 + (i & 127)] = s_raw[(i >> 7) * 384 + 256 + (i & 127)];
    __syncthreads();

    // save the new tokens' (k, v, g, beta) for a possible replay next cycle
    for (int i = tid; i < T * 128; i += NT) {
        const int t = i >> 7, c = i & 127;
        float * slot = ring + (size_t) ((n + t) % KV_RING) * KVGB;
        if (h < NKH) slot[h * 128 + c] = s_k[(np + t) * QS + (c >> 5) * 36 + (c & 31)];
        slot[KVGB_V + h * 128 + c] = s_v[(np + t) * 128 + c];
        if (c == 0) {
            slot[KVGB_G + h] = s_gb[2 * (np + t)];
            slot[KVGB_B + h] = s_gb[2 * (np + t) + 1];
        }
    }

    // ---- gated delta rule over all events; inputs are in shared memory, no barriers needed ----
    for (int e = 0; e < np + T; e++) {
        const float g = s_gb[2 * e], beta = s_gb[2 * e + 1];
        const float4 * kq = reinterpret_cast<const float4 *>(s_k + e * QS + r * 36);
        float kv = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            kv += S[i].x * k4.x + S[i].y * k4.y + S[i].z * k4.z + S[i].w * k4.w;
        }
        kv += __shfl_xor_sync(0xffffffffu, kv, 1);
        kv += __shfl_xor_sync(0xffffffffu, kv, 2);
        const float dlt = (s_v[e * 128 + col] - g * kv) * beta;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 k4 = kq[i];
            S[i].x = g * S[i].x + k4.x * dlt;
            S[i].y = g * S[i].y + k4.y * dlt;
            S[i].z = g * S[i].z + k4.z * dlt;
            S[i].w = g * S[i].w + k4.w * dlt;
        }
        if (e >= np) {
            const float4 * qq = reinterpret_cast<const float4 *>(s_q + e * QS + r * 36);
            float o = 0.f;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const float4 q4 = qq[i];
                o += S[i].x * q4.x + S[i].y * q4.y + S[i].z * q4.z + S[i].w * q4.w;
            }
            o += __shfl_xor_sync(0xffffffffu, o, 1);
            o += __shfl_xor_sync(0xffffffffu, o, 2);
            if (r == 0) s_o[(e - np) * 128 + col] = o;
        }
        if (e == np + n_commit - 1) {
#pragma unroll
            for (int i = 0; i < 8; i++) reinterpret_cast<float4 *>(Sg)[i] = S[i];
        }
    }
    __syncthreads();

    // gated rms-norm + quantise: warp w -> token w / 4, 32-element block w % 4 of this head
    for (int w = warp; w < 4 * T; w += NW) {
        const int t = w >> 2, jb = w & 3;
        const float4 v = *reinterpret_cast<const float4 *>(s_o + t * 128 + lane * 4);
        const float rs = rsqrtf(wsum(v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w) / 128.0f + EPS);
        const int c = jb * 32 + lane;
        const float y = s_o[t * 128 + c] * rs * W.snorm[c] * silu(s_z[t * 128 + c]);
        q32_store(cB.qo, t, 4 * h + jb, y, 0.f, false);
    }
}

// q8_0 store of one KV row: lane holds values 8 lane .. 8 lane + 7 (4 lanes per 32-value block)
__device__ __forceinline__ void kv_store_q8(int8_t * q, __half * sc, const float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    float am = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) am = fmaxf(am, fabsf(v[k]));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 1));
    am = fmaxf(am, __shfl_xor_sync(0xffffffffu, am, 2));
    const float id = am > 0.f ? 127.0f / am : 0.f;
    unsigned w[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        unsigned x = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) x |= ((unsigned) __float2int_rn(v[4 * h + k] * id) & 0xffu) << (8 * k);
        w[h] = x;
    }
    *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(am / 127.0f);
}

// q4 store of one KV row (row = HD / 2 bytes), q4_0 rounding: per 32-value block d = (the value of largest
// magnitude) / -8, nibble = min(15, (int) (x / d + 8.5)), value = (nibble - 8) d. Lane holds values 8 lane .. + 7 and
// writes one word: low nibbles = values 0..3, high nibbles = values 4..7. Ties of |x| go to the larger x (the same
// in all 4 lanes of a block). Bit-identical to MK_KV_EMU mode "4a".
__device__ __forceinline__ void kv_store_q4(int8_t * q, __half * sc, const float (&v)[8]) {
    const int lane = threadIdx.x & 31;
    float am = 0.f, sv = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (fabsf(v[k]) > am || (fabsf(v[k]) == am && v[k] > sv)) { am = fabsf(v[k]); sv = v[k]; }
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, m), os = __shfl_xor_sync(0xffffffffu, sv, m);
        if (oa > am || (oa == am && os > sv)) { am = oa; sv = os; }
    }
    const float d = sv / -8.0f, id = d != 0.f ? 1.0f / d : 0.f;
    unsigned w = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) w |= (unsigned) min(15, (int) (v[k] * id + 8.5f)) << (k < 4 ? 8 * k : 8 * (k - 4) + 4);
    *reinterpret_cast<unsigned *>(q + lane * 4) = w;
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(d);
}

// 256-point fast Walsh-Hadamard transform of one row, 8 values per lane (scaled by 1/16: orthonormal, its own inverse)
__device__ __forceinline__ void fwht256(float (&v)[8]) {
#pragma unroll
    for (int m = 1; m < 8; m <<= 1)
#pragma unroll
        for (int k = 0; k < 8; k++)
            if (!(k & m)) {
                const float a = v[k], b = v[k | m];
                v[k] = a + b;
                v[k | m] = a - b;
            }
    const int lane = threadIdx.x & 31;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1)
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float o = __shfl_xor_sync(0xffffffffu, v[k], m);
            v[k] = (lane & m) ? o - v[k] : v[k] + o;
        }
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] *= 0.0625f;
}

// one K (w = 0) or V (w = 1) row of attention slot `slot` in its MK_KV format; row = kv head * ctx + position
__device__ __forceinline__ void kv_store(int w, int slot, size_t row, const float (&vin)[8]) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (w == 0 && cB.kvr[slot]) fwht256(v);
    if (cB.kvb[w][slot] == 4) kv_store_q4(cB.kvp[w][slot] + row * (HD / 2), cB.kvs[w][slot] + row * (HD / 32), v);
    else kv_store_q8(cB.kvp[w][slot] + row * HD, cB.kvs[w][slot] + row * (HD / 32), v);
}

#if MK_KV_EMU

// A K / V row rounded to a lower-bit grid with one fp16 scale per 32 values (the format a real low-bit cache would
// store), written in the q8 layout. mode: bits (2..8) | 16: Hadamard-rotated first (the dequantized row is rotated
// back and stored as ordinary q8, which adds q8 rounding) | 32: q4_0-style scale d = (signed max) / -2^(b-1) with
// 2^b levels (else symmetric: d = max |x| / (2^(b-1) - 1)).
__device__ __forceinline__ void kv_store_emu(int8_t * q, __half * sc, const float (&vin)[8], int mode) {
    const int lane = threadIdx.x & 31, bits = mode & 15;
    const bool rot = mode & 16, q40 = mode & 32;
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (rot) fwht256(v);
    float am = 0.f, sv = 0.f; // largest |x| of the 32-value block and that x (ties: the larger x, same in all 4 lanes)
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (fabsf(v[k]) > am || (fabsf(v[k]) == am && v[k] > sv)) { am = fabsf(v[k]); sv = v[k]; }
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, m), os = __shfl_xor_sync(0xffffffffu, sv, m);
        if (oa > am || (oa == am && os > sv)) { am = oa; sv = os; }
    }
    const int h = 1 << (bits - 1);
    const float d = q40 ? sv / (float) -h : am / (float) (h - 1);
    const float id = d != 0.f ? 1.0f / d : 0.f;
    const __half dh = __float2half_rn(d);
    int qi[8];
#pragma unroll
    for (int k = 0; k < 8; k++)
        qi[k] = q40 ? min(2 * h - 1, (int) (v[k] * id + ((float) h + 0.5f))) - h : max(1 - h, min(h - 1, __float2int_rn(v[k] * id)));
    if (!rot) {
        unsigned w[2] = {0u, 0u};
#pragma unroll
        for (int k = 0; k < 8; k++) w[k >> 2] |= ((unsigned) qi[k] & 0xffu) << (8 * (k & 3));
        *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
        if ((lane & 3) == 0) sc[lane >> 2] = dh;
        return;
    }
    const float df = __half2float(dh);
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = df * (float) qi[k];
    fwht256(v);
    kv_store_q8(q, sc, v);
}
#endif

// fp16 store of one KV row (same lane layout)
__device__ __forceinline__ void kv_store_f16(__half * dst, const float (&v)[8]) {
    __half2 h[4];
#pragma unroll
    for (int k = 0; k < 4; k++) h[k] = __floats2half2_rn(v[2 * k], v[2 * k + 1]);
    *reinterpret_cast<int4 *>(dst + (threadIdx.x & 31) * 8) = *reinterpret_cast<int4 *>(h);
}

// 8 q8 values + scale -> 4 fp16 pairs
__device__ __forceinline__ void q8_to_f16(uint2 q, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned x = k < 2 ? q.x : q.y;
        const int sh = 16 * (k & 1);
        h[k] = __floats2half2_rn(d * (float) (int8_t) (x >> sh), d * (float) (int8_t) (x >> (sh + 8)));
    }
}

// 8 q4 values (one word in kv_store_q4's nibble order) + scale -> 4 fp16 pairs, the q8_to_f16 arithmetic
__device__ __forceinline__ void q4_to_f16(unsigned w, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int j0 = 2 * k, j1 = 2 * k + 1;
        const int a = (int) ((w >> (j0 < 4 ? 8 * j0 : 8 * (j0 - 4) + 4)) & 0xfu) - 8;
        const int b = (int) ((w >> (j1 < 4 ? 8 * j1 : 8 * (j1 - 4) + 4)) & 0xfu) - 8;
        h[k] = __floats2half2_rn(d * (float) a, d * (float) b);
    }
}

// staged q8 / q4 K/V rows (KVQ_ROW bytes: values, then scales at 256) -> fp16 rows of stride ld; rows x 2 tensors
__device__ __forceinline__ void kv_tile_to_f16(const uint8_t * src, __half * dK, __half * dV, int rows, int ld, bool kq4, bool vq4) {
    for (int i = threadIdx.x; i < 2 * rows * 32; i += blockDim.x) {
        const int which = i / (rows * 32), r = (i >> 5) % rows, c = i & 31;
        const uint8_t * row = src + (size_t) (which * rows + r) * KVQ_ROW;
        const float d = __half2float(reinterpret_cast<const __half *>(row + 256)[c >> 2]);
        __half2 h[4];
        if (which ? vq4 : kq4) q4_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), d, h);
        else q8_to_f16(*reinterpret_cast<const uint2 *>(row + 8 * c), d, h);
        *reinterpret_cast<int4 *>((which ? dV : dK) + (size_t) r * ld + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// async copy of q8 / q4 K/V rows r < np (slot rows row0 + r) into the staging layout (rows >= np: zero scales)
__device__ __forceinline__ void kv_stage(uint8_t * dst, const int8_t * Kq, const int8_t * Vq, const __half * Ks, const __half * Vs,
                                         size_t row0, int rows, int np, bool kq4, bool vq4) {
    for (int i = threadIdx.x; i < 2 * rows * 17; i += blockDim.x) {
        const int which = i / (rows * 17), r = (i / 17) % rows, c = i % 17;
        uint8_t * d = dst + (size_t) (which * rows + r) * KVQ_ROW + 16 * c;
        if (r < np) {
            const size_t row = row0 + r;
            const bool q4 = which ? vq4 : kq4;
            if (c < 16) {
                if (!q4 || c < 8) cp_async16(d, (which ? Vq : Kq) + row * (q4 ? HD / 2 : HD) + 16 * c);
            } else {
                cp_async16(d, (which ? Vs : Ks) + row * (HD / 32));
            }
        } else if (c == 16) {
            *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
        }
    }
}

// split-K over the context: at most ATT_NCH chunks per kv head so all 4 x nch items run in one wave.
// The chunk size is a multiple of the K/V tile that only changes when L_ctx crosses a multiple of
// ATT_BUCKET, and the host never lets one launch's rows straddle such a boundary (att_cap_T). So the
// chunk boundaries below any query position p (and the float summation order of its attention) depend
// on p alone, not on how many tokens the cycle verifies: greedy output is independent of the draft depth.
// Same tiles per item as the previous ceil(L / 20) split for L > 1280; one tile per item up to 640.
__device__ __forceinline__ void attn_dims(int L_ctx, int & nch, int & cs) {
    cs = ATP * max(1, (L_ctx + ATT_BUCKET - 1) / ATT_BUCKET);
    nch = max(1, (L_ctx + cs - 1) / cs);
}

// attention q/k/v projections
template <int T>
__device__ __noinline__ void attn_proj(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    (void) lane;
    const SX sx = make_sx(sm, U_D);
    if (gwarp() < QG / 2) pf_rows(W.q, 2 * gwarp(), 2, 576);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const int nq = QG / 2, nk = KVD / 2;
    gemv_pairs_any<T, 5>(W.q, 0, nq, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.qbuf, QG, r, a); });
    gemv_pairs_any<T, 5>(W.k, nq, nq + nk, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.kbuf, KVD, r, a); });
    gemv_pairs_any<T, 5>(W.v, nq + nk, nq + 2 * nk, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.vbuf, KVD, r, a); });
    // Warm L2 with what attn_core reads first: q/k norm weights, RoPE rows of the new positions,
    // and the first K / V tile of this block's attention item (positions < n).
    pf_block(W.qn, HD * 4);
    pf_block(W.kn, HD * 4);
    pf_block(cB.rope + (size_t) n * 32, T * 32 * 8);
    int nch, CS;
    attn_dims(n + T, nch, CS);
    const size_t ctx = cB.ctx;
    // bytes and 128-byte lines per K / V row (fp16 256 x 2, q8 256, q4 128)
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

__device__ __forceinline__ void rope8(float (&v)[8], int lane, int pos) {
    float o[8];
#pragma unroll
    for (int k = 0; k < 8; k++) o[k] = __shfl_xor_sync(0xffffffffu, v[k], 4);
    if (lane < 8) {
        const bool first = lane < 4;
        const float2 * tab = cB.rope + (size_t) pos * 32 + (lane & 3) * 8;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float2 cs = __ldg(tab + k);
            v[k] = first ? v[k] * cs.x - o[k] * cs.y : o[k] * cs.y + v[k] * cs.x;
        }
    }
}

__device__ __forceinline__ void load8(const float * p, float (&v)[8]) {
    const float4 a = __ldcg(reinterpret_cast<const float4 *>(p));
    const float4 b = __ldcg(reinterpret_cast<const float4 *>(p + 4));
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}


// split-K flash decoding over (kv head, chunk) items; also writes this cycle's K/V.
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// Split-K flash decoding over (kv head, chunk) items.  The 6*T query rows of a kv head form the
// M dimension of tensor-core MMAs: S = Q K^T and O += P V per 64-position tile of K / V staged
// in shared memory (fp16 operands, fp32 accumulation, online softmax).  Also writes this cycle's K/V.
template <int T, bool Q8>
__device__ __forceinline__ void attn_core_t(const LayerW & W, int slot, int n) {
    uint8_t * const sm = smem;
    constexpr int R = 6 * T;
    constexpr int MT = (R + 15) / 16; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;           // positions per K/V tile; tiles are double-buffered
    constexpr int LDH = HD + 8;       // smem row stride in halves (conflict-free fragment loads)
    constexpr int LDP = TP + 8;
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims(L_ctx, nch, CS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
    __half * sQ = reinterpret_cast<__half *>(sm); // [32][LDH]
    // fp16 cache: 2 stages x (K [TP][LDH], V [TP][LDH]); q8 cache: one fp16 K/V tile + 2 q8 stages
    __half * sKV = sQ + 32 * LDH;
    uint8_t * sKVq = reinterpret_cast<uint8_t *>(sKV + 2 * TP * LDH);
    float * sS = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(sKV) + SM_ATTN_KV); // [32][TP]
    __half * sP = reinterpret_cast<__half *>(sS + 32 * TP); // [32][LDP]
    float * sM = reinterpret_cast<float *>(sP + 32 * LDP);
    float * sL = sM + 32;
    float * sAl = sL + 32;
    const size_t ctx = cB.ctx;
    const size_t slot0 = (size_t) slot * NKV * ctx; // first row of this slot
    __half * Kc = static_cast<__half *>(cB.kc) + slot0 * HD;
    __half * Vc = static_cast<__half *>(cB.vc) + slot0 * HD;
    const bool kq4 = Q8 && cB.kvb[0][slot] == 4, vq4 = Q8 && cB.kvb[1][slot] == 4; // MK_KV per-slot formats
    const int mt = warp >> 3, wn = warp & 7; // MMA work split: warp -> (row tile, column group)

    auto issue_tile = [&](int g, int tb, int np_, int st) {
        if constexpr (Q8) {
            kv_stage(sKVq + st * 2 * TP * KVQ_ROW, cB.kvp[0][slot], cB.kvp[1][slot], cB.kvs[0][slot], cB.kvs[1][slot],
                     (size_t) g * ctx + tb, TP, np_, kq4, vq4);
            asm volatile("cp.async.commit_group;" ::: "memory");
            return;
        }
        __half * sK = sKV + st * 2 * TP * LDH, * sV = sK + TP * LDH;
        const __half * gK = Kc + ((size_t) g * ctx + tb) * HD;
        const __half * gV = Vc + ((size_t) g * ctx + tb) * HD;
        for (int i = tid; i < TP * (HD / 8); i += NT) {
            const int p = i >> 5, cc = (i & 31) * 8;
            if (p < np_) {
                cp_async16(sK + p * LDH + cc, gK + (size_t) p * HD + cc);
                cp_async16(sV + p * LDH + cc, gV + (size_t) p * HD + cc);
            } else {
                *reinterpret_cast<int4 *>(sK + p * LDH + cc) = make_int4(0, 0, 0, 0);
                *reinterpret_cast<int4 *>(sV + p * LDH + cc) = make_int4(0, 0, 0, 0);
            }
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
    };
    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c = item % nch;
        const int c0 = c * CS, c1 = min(L_ctx, c0 + CS);
        // one round of small loads (q rows, norm weights, RoPE rows) before any bulk K/V traffic:
        // per-SM requests are served roughly in order
        float qv[2][8], qw[8];
        float2 rc[2][8];
        load8(W.qn + lane * 8, qw);
        // Unconditional loads (rows >= R and lanes >= 8 load valid, unused data): conditionally assigned arrays
        // were carried across the item loop in local memory (~1.5 KB of mk_cycle's stack frame, which the driver
        // reserves for all 1,536 thread slots of every SM).
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = min(warp + 16 * k2, R - 1);
            const int hq = r / T, t = r % T, head = g * 6 + hq;
            load8(cB.qbuf + t * QG + head * 512 + lane * 8, qv[k2]);
            const float2 * tab = cB.rope + (size_t) (n + t) * 32 + (lane & 3) * 8;
#pragma unroll
            for (int k = 0; k < 8; k++) rc[k2][k] = __ldg(tab + k);
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = warp + 16 * k2;
            if (r < R) {
                // a copy, not a pointer into qv: a pointer made ptxas keep qv / qw / rc in local memory (~2 KB of
                // mk_cycle's 2.8 KB stack frame over the 10 attn_core_t instances, reserved for 1,536 threads per SM)
                float v[8];
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] = qv[k2][k];
                float ss = 0.f;
#pragma unroll
                for (int k = 0; k < 8; k++) ss += v[k] * v[k];
                const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] *= rs * qw[k];
                float o[8];
#pragma unroll
                for (int k = 0; k < 8; k++) o[k] = __shfl_xor_sync(0xffffffffu, v[k], 4);
                if (lane < 8) {
#pragma unroll
                    for (int k = 0; k < 8; k++)
                        v[k] = lane < 4 ? v[k] * rc[k2][k].x - o[k] * rc[k2][k].y : o[k] * rc[k2][k].y + v[k] * rc[k2][k].x;
                }
                if (Q8 && cB.kvr[slot]) fwht256(v); // K of this slot is stored rotated (MK_KV "4r")
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
            } else {
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u; // padded rows of P stay 0
            }
        }
        // this cycle's K / V rows that fall into the chunk (positions >= n)
        const int pn0 = max(c0, n), pn1 = min(c1, n + T);
        for (int pp = pn0 + warp; pp < pn1; pp += NW) {
            const int t = pp - n;
            float v[8];
            load8(cB.kbuf + t * KVD + g * 256 + lane * 8, v);
            float ss = 0.f;
#pragma unroll
            for (int k = 0; k < 8; k++) ss += v[k] * v[k];
            const float rs = rsqrtf(wsum(ss) / 256.0f + EPS);
#pragma unroll
            for (int k = 0; k < 8; k++) v[k] *= rs * W.kn[lane * 8 + k];
            rope8(v, lane, pp);
            const size_t row = slot0 + (size_t) g * ctx + pp, srow = (size_t) g * ctx + pp;
#if MK_KV_EMU
            if (Q8 && cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot]);
            else
#endif
            if constexpr (Q8) kv_store(0, slot, srow, v);
            else kv_store_f16(static_cast<__half *>(cB.kc) + row * HD, v);
            load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
            if (Q8 && cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot]);
            else
#endif
            if constexpr (Q8) kv_store(1, slot, srow, v);
            else kv_store_f16(static_cast<__half *>(cB.vc) + row * HD, v);
        }
        if (tid < 32) {
            sM[tid] = -INFINITY;
            sL[tid] = 0.f;
        }
        __threadfence_block();
        __syncthreads();

        float acc[4][4]; // O tiles: rows 16*mt + {g8, g8+8}, dims 8*(wn + 8j) + {2tig, 2tig+1}
#pragma unroll
        for (int j = 0; j < 4; j++)
#pragma unroll
            for (int k = 0; k < 4; k++) acc[j][k] = 0.f;

        issue_tile(g, c0, min(TP, c1 - c0), 0);
        int st = 0;
        for (int tb = c0; tb < c1; tb += TP, st ^= 1) {
            const int np_ = min(TP, c1 - tb);
            const __half * sK = Q8 ? sKV : sKV + st * 2 * TP * LDH, * sV = sK + TP * LDH;
            // the next tile streams into the other stage while this one is processed
            if (tb + TP < c1) {
                issue_tile(g, tb + TP, min(TP, c1 - tb - TP), st ^ 1);
                asm volatile("cp.async.wait_group 1;" ::: "memory");
            } else {
                asm volatile("cp.async.wait_group 0;" ::: "memory");
            }
            __syncthreads();
            if constexpr (Q8) { // q8 stage -> the fp16 tile
                kv_tile_to_f16(sKVq + st * 2 * TP * KVQ_ROW, sKV, sKV + TP * LDH, TP, LDH, kq4, vq4);
                __syncthreads();
            }
            // S = Q K^T : warp -> row tile mt, positions 8*wn .. 8*wn+7
            if (mt < MT && wn < TP / 8) {
                float s[4] = {0.f, 0.f, 0.f, 0.f};
                const __half * qa = sQ + (16 * mt + g8) * LDH + 2 * tig;
                const __half * kb = sK + (8 * wn + g8) * LDH + 2 * tig;
#pragma unroll
                for (int ks = 0; ks < HD / 16; ks++) {
                    const unsigned a0 = *reinterpret_cast<const unsigned *>(qa + 16 * ks);
                    const unsigned a1 = *reinterpret_cast<const unsigned *>(qa + 8 * LDH + 16 * ks);
                    const unsigned a2 = *reinterpret_cast<const unsigned *>(qa + 16 * ks + 8);
                    const unsigned a3 = *reinterpret_cast<const unsigned *>(qa + 8 * LDH + 16 * ks + 8);
                    const unsigned b0 = *reinterpret_cast<const unsigned *>(kb + 16 * ks);
                    const unsigned b1 = *reinterpret_cast<const unsigned *>(kb + 16 * ks + 8);
                    mma16816(s, a0, a1, a2, a3, b0, b1);
                }
                float * so = sS + (16 * mt + g8) * TP + 8 * wn + 2 * tig;
                so[0] = s[0];
                so[1] = s[1];
                so[8 * TP] = s[2];
                so[8 * TP + 1] = s[3];
            }
            __syncthreads();
            // online softmax per query row (causal mask: row token t sees positions <= n + t)
            for (int r = warp; r < R; r += NW) {
                const int lim = n + (r % T);
                float s0 = sS[r * TP + lane], s1 = TP > 32 ? sS[r * TP + (lane + 32) % TP] : -INFINITY;
                if (lane >= np_ || tb + lane > lim) s0 = -INFINITY;
                if (lane + 32 >= np_ || tb + lane + 32 > lim) s1 = -INFINITY;
                const float mt_ = wmax(fmaxf(s0, s1));
                const float mold = sM[r];
                const float mnew = fmaxf(mold, mt_);
                float p0 = 0.f, p1 = 0.f, al = 1.f;
                if (mnew != -INFINITY) {
                    p0 = expf(s0 - mnew);
                    p1 = expf(s1 - mnew);
                    al = expf(mold - mnew);
                }
                const float ps = wsum(p0 + p1);
                sP[r * LDP + lane] = __float2half_rn(p0);
                if (TP > 32) sP[r * LDP + (lane + 32) % TP] = __float2half_rn(p1);
                if (lane == 0) {
                    sL[r] = sL[r] * al + ps;
                    sM[r] = mnew;
                    sAl[r] = al;
                }
            }
            __syncthreads();
            // O = O * alpha + P V : warp -> row tile mt, dims 8 * (wn + 8 j)
            if (mt < MT) {
                const float al0 = sAl[16 * mt + g8], al1 = sAl[16 * mt + g8 + 8];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    acc[j][0] *= al0;
                    acc[j][1] *= al0;
                    acc[j][2] *= al1;
                    acc[j][3] *= al1;
                }
                const __half * pa = sP + (16 * mt + g8) * LDP + 2 * tig;
#pragma unroll
                for (int ks = 0; ks < TP / 16; ks++) {
                    const unsigned a0 = *reinterpret_cast<const unsigned *>(pa + 16 * ks);
                    const unsigned a1 = *reinterpret_cast<const unsigned *>(pa + 8 * LDP + 16 * ks);
                    const unsigned a2 = *reinterpret_cast<const unsigned *>(pa + 16 * ks + 8);
                    const unsigned a3 = *reinterpret_cast<const unsigned *>(pa + 8 * LDP + 16 * ks + 8);
#pragma unroll
                    for (int j = 0; j < 4; j++) {
                        const __half * vb = sV + (16 * ks + (lane & 15)) * LDH + 8 * (wn + 8 * j);
                        unsigned b0, b1;
                        asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
                                     : "=r"(b0), "=r"(b1)
                                     : "r"((unsigned) __cvta_generic_to_shared(vb)));
                        mma16816(acc[j], a0, a1, a2, a3, b0, b1);
                    }
                }
            }
            __syncthreads();
        }
        if (mt < MT) {
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int r0 = 16 * mt + g8, d0 = 8 * (wn + 8 * j) + 2 * tig;
                if (r0 < R) {
                    float * o = cB.apart + ((size_t) item * AR + r0) * 256 + d0;
                    o[0] = acc[j][0];
                    o[1] = acc[j][1];
                }
                if (r0 + 8 < R) {
                    float * o = cB.apart + ((size_t) item * AR + r0 + 8) * 256 + d0;
                    o[0] = acc[j][2];
                    o[1] = acc[j][3];
                }
            }
        }
        if (tid < R) cB.aml[item * AR + tid] = make_float2(sM[tid], sL[tid]);
        __syncthreads();
    }
}

template <int T>
__device__ __noinline__ void attn_core(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    if (cB.kvq8) attn_core_t<T, true>(W, slot, n);
    else attn_core_t<T, false>(W, slot, n);
}

// combine split-K partials, apply sigmoid output gate, quantise for o_proj
template <int T>
__device__ __noinline__ void attn_combine(int n) {
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims(L_ctx, nch, CS);
    const int tid = threadIdx.x, d = tid & 255, wh = (tid & 255) >> 5;
    for (int it = blockIdx.x * 2 + (tid >> 8); it < NH * T; it += 2 * gridDim.x) {
        const int h = it / T, t = it % T, g = h / 6, r = (h % 6) * T + t;
        float M = -INFINITY;
        for (int c = 0; c < nch; c++) M = fmaxf(M, __ldcg(&cB.aml[(g * nch + c) * AR + r].x));
        float num = 0.f, den = 0.f;
        for (int c = 0; c < nch; c++) {
            const float2 ml = __ldcg(&cB.aml[(g * nch + c) * AR + r]);
            const float w = ml.x == -INFINITY ? 0.f : expf(ml.x - M);
            num += w * __ldcg(cB.apart + ((size_t) (g * nch + c) * AR + r) * 256 + d);
            den += w * ml.y;
        }
        const float gate = __ldcg(cB.qbuf + t * QG + h * 512 + 256 + d);
        const float o = num / den / (1.0f + expf(-gate));
        q32_store(cB.qo, t, 8 * h + wh, o, 0.f, false);
    }
}

// final RMS norm rows -> h (fp32) for the MTP head.
template <int T>
__device__ __noinline__ void write_h_rows(const float * x, const QBuf & q, const float * w, float * dst_base, int pos0, bool ring,
                             uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    if (blockIdx.x != 0) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float * s_rs = misc(sm);
    if (warp < T) {
        float acc = 0.f;
        for (int j = lane; j < U_D; j += 32) acc += __ldcg(q.ssq + warp * U_D + j);
        acc = wsum(acc);
        if (lane == 0) s_rs[warp] = rsqrtf(acc / (float) D + EPS);
    }
    __syncthreads();
    for (int i = threadIdx.x; i < T * D; i += NT) {
        const int t = i / D, e = i - t * D;
        float * dst = ring ? dst_base + (size_t) ((pos0 + t) % H_RING) * D : dst_base;
        if (!ring && t != T - 1) continue;
        dst[e] = __ldcg(x + i) * s_rs[t] * w[e];
    }
    __syncthreads();
}

// warp-wide sorted (descending) list of the 32 best (value, id), lane i holding rank i; v is warp-uniform.
// Returns false (and inserts nothing) when v does not beat the current 32nd value.
__device__ __forceinline__ bool topk_insert(float & tv, int & ti, float v, int id) {
    const float mn = __shfl_sync(0xffffffffu, tv, 31);
    if (!(v > mn)) return false;
    const int lane = threadIdx.x & 31;
    const int pos = __popc(__ballot_sync(0xffffffffu, tv >= v));
    const float uv = __shfl_up_sync(0xffffffffu, tv, 1);
    const int ui = __shfl_up_sync(0xffffffffu, ti, 1);
    if (lane == pos) { tv = v; ti = id; }
    else if (lane > pos) { tv = uv; ti = ui; }
    return true;
}

// lm_head over TL rows (qn rows t0..t0+TL-1) with greedy argmax; returns true in the single
// block that finished last, which holds the reduced argmax in s_tok.  TOPK: additionally keeps the
// TOPK_MAX best logits of each row in cand (default cB.cand; for sampling).  FULL: additionally stores
// every logit of each row in cB.logits (scoring).  LSE (TL == 1): also the log-sum-exp of the row's
// true logits; the last block leaves misc[LM_LSE] = lse and misc[LM_BEST] = the best true logit.
// Draft rows: s_tok and the TOPK ids are token ids (mapped through cW.draft_ids).
constexpr int LM_LSE = 240, LM_BEST = 241, LM_WLSE = 208; // misc() slots ([LM_WLSE, +2 * NW): per-warp lse)
template <int TL, bool TOPK = false, bool FULL = false, bool LSE = false>
__device__ __noinline__ bool lm_head(int t0, bool draft, uint8_t * /*sm*/, int * s_tok, float2 * cand = nullptr) {
    static_assert(!LSE || TL == 1, "LSE: one row");
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const SX sx = make_sx(sm, U_D);
    float * m = misc(sm);
    // the draft head reads lm_head rows through cW.draft_ids (row r of head_d is lm_head row draft_ids[r])
    const int * ids = draft ? cW.draft_ids : nullptr;
    if (ids) {
        const int r0 = 2 * gwarp();
        if (r0 < cW.head_d.N)
            for (int k = 0; k < 2; k++) {
                const uint8_t * p = cW.head.w + (size_t) __ldg(ids + r0 + k) * cW.head.rb;
                for (int off = lane * 128; off < 896; off += 32 * 128) asm volatile("prefetch.global.L2 [%0];" ::"l"(p + off));
            }
    } else {
        pf_rows(draft ? cW.head_d : cW.head, 2 * gwarp(), 2, 896);
    }
    fill_xs<TL>(cB.qn, t0, true, sx, m);
    float bv[TL];
    int bi[TL];
#pragma unroll
    for (int t = 0; t < TL; t++) {
        bv[t] = -INFINITY;
        bi[t] = TOPK ? -1 : 0;
    }
    float lm_m = -INFINITY, lm_s = 0.f; // LSE: running max / sum of exp (warp-uniform)
    const Mat head = draft ? cW.head_d : cW.head;
    for (int r = 2 * gwarp(); r < head.N; r += 2 * nwarps()) {
        float a[2][TL];
        zero_acc<TL, 2>(a);
        {
            const uint8_t * rp[2];
            const __half * dp[2]; // Q6_K: the rows' scale (meta) streams
            const int p0 = ids ? __ldg(ids + r) : r, p1 = ids ? __ldg(ids + r + 1) : r + 1;
            rp[0] = head.w + (size_t) p0 * head.rb;
            rp[1] = head.w + (size_t) p1 * head.rb;
            dp[0] = head.d + (size_t) p0 * head.nd;
            dp[1] = head.d + (size_t) p1 * head.nd;
            rows_t<TL, 5, 2, T_Q6K>(rp, dp, sx, a);
        }
        wsum_acc<TL, 2>(a);
        if constexpr (LSE) {
            const float v0 = a[0][0] * m[0], v1 = a[1][0] * m[0];
            const float mx = fmaxf(lm_m, fmaxf(v0, v1));
            lm_s = lm_s * __expf(lm_m - mx) + __expf(v0 - mx) + __expf(v1 - mx);
            lm_m = mx;
        }
        if constexpr (FULL) {
            if (lane == 0) {
#pragma unroll
                for (int t = 0; t < TL; t++) {
                    cB.logits[(size_t) (t0 + t) * VOCAB + r] = a[0][t] * m[t];
                    cB.logits[(size_t) (t0 + t) * VOCAB + r + 1] = a[1][t] * m[t];
                }
            }
        }
#pragma unroll
        for (int t = 0; t < TL; t++) {
            if constexpr (TOPK) { // true logits: apply the final RMS norm scale (irrelevant to the argmax)
                topk_insert(bv[t], bi[t], a[0][t] * m[t], r);
                topk_insert(bv[t], bi[t], a[1][t] * m[t], r + 1);
            } else {
                if (a[0][t] > bv[t]) { bv[t] = a[0][t]; bi[t] = r; }
                if (a[1][t] > bv[t]) { bv[t] = a[1][t]; bi[t] = r + 1; }
            }
        }
    }
    if constexpr (LSE) {
        if (lane == 0) {
            m[LM_WLSE + 2 * warp] = lm_m;
            m[LM_WLSE + 2 * warp + 1] = lm_s;
        }
    }
    if constexpr (TOPK) {
        // merge the warps' lists (shared memory, the activations are no longer needed), then the blocks'
        __syncthreads();
        float2 * wl = reinterpret_cast<float2 *>(sm); // [NW][TL][32]
#pragma unroll
        for (int t = 0; t < TL; t++) wl[(warp * TL + t) * 32 + lane] = make_float2(bv[t], __int_as_float(bi[t]));
        __syncthreads();
        if (warp < TL) {
            float v = -INFINITY;
            int id = -1;
            for (int w = 0; w < NW; w++)
                for (int j = 0; j < 32; j++) {
                    const float2 e = wl[(w * TL + warp) * 32 + j];
                    if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
                }
            cB.blk_best[((size_t) blockIdx.x * MAXT + warp) * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
        }
    } else {
        float * wv = m + 16;                                    // [NW][MAXT]
        int * wi = reinterpret_cast<int *>(m + 16 + NW * MAXT); // [NW][MAXT]
        if (lane == 0) {
#pragma unroll
            for (int t = 0; t < TL; t++) {
                wv[warp * MAXT + t] = bv[t];
                wi[warp * MAXT + t] = bi[t];
            }
        }
        __syncthreads();
        if (tid < TL) {
            float b = -INFINITY;
            int ix = 0;
            for (int w = 0; w < NW; w++) {
                const float v = wv[w * MAXT + tid];
                const int id = wi[w * MAXT + tid];
                if (v > b || (v == b && id < ix)) { b = v; ix = id; }
            }
            cB.blk_best[blockIdx.x * MAXT + tid] = make_float2(b, __int_as_float(ix));
        }
    }
    if constexpr (LSE) { // block's (max, sum of exp); the per-warp values were published before a barrier above
        if (tid == NT - 1) {
            float M = -INFINITY, S = 0.f;
            for (int w = 0; w < NW; w++) M = fmaxf(M, m[LM_WLSE + 2 * w]);
            for (int w = 0; w < NW; w++) S += m[LM_WLSE + 2 * w + 1] * __expf(m[LM_WLSE + 2 * w] - M);
            cB.blk_lse[blockIdx.x] = make_float2(M, S);
        }
    }
    __threadfence();
    __syncthreads();
    int * s_last = reinterpret_cast<int *>(m + 200);
    if (tid == 0) s_last[0] = (atomicAdd(cB.cnt, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!s_last[0]) return false;
    __threadfence();
    if constexpr (LSE) {
        if (tid == NT - 1) {
            float M = -INFINITY, S = 0.f;
            for (int k = 0; k < (int) gridDim.x; k++) M = fmaxf(M, __ldcg(&cB.blk_lse[k]).x);
            for (int k = 0; k < (int) gridDim.x; k++) {
                const float2 e = __ldcg(&cB.blk_lse[k]);
                S += e.y * __expf(e.x - M);
            }
            m[LM_LSE] = M + __logf(S);
        }
    }
    if constexpr (TOPK) {
        if (warp < TL) {
            float v = -INFINITY;
            int id = -1;
            for (int k = 0; k < (int) gridDim.x; k++)
                for (int j = 0; j < 32; j++) {
                    const float2 e = __ldcg(&cB.blk_best[((size_t) k * MAXT + warp) * TOPK_MAX + j]);
                    if (!topk_insert(v, id, e.x, __float_as_int(e.y))) break;
                }
            if (draft && cW.draft_ids && id >= 0) id = cW.draft_ids[id];
            (cand ? cand : cB.cand)[warp * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
            const int best = __shfl_sync(0xffffffffu, id, 0);
            if (lane == 0) {
                s_tok[warp] = best;
                if (LSE) m[LM_BEST] = v;
            }
        }
    } else if (tid < TL) {
        float b = -INFINITY;
        int ix = 0;
        for (int k = 0; k < (int) gridDim.x; k++) {
            const float2 e = __ldcg(&cB.blk_best[k * MAXT + tid]);
            const int id = __float_as_int(e.y);
            if (e.x > b || (e.x == b && id < ix)) { b = e.x; ix = id; }
        }
        s_tok[tid] = (draft && cW.draft_ids) ? cW.draft_ids[ix] : ix;
        if (LSE) m[LM_BEST] = b * m[0];
    }
    if (tid == 0) *cB.cnt = 0;
    __syncthreads();
    return true;
}

// ---- sampling (single thread) ----
// counter-based uniform random number in [0, 1)
__device__ __forceinline__ float urand(unsigned long long seed, unsigned long long ctr) {
    unsigned long long z = seed + (ctr + 1) * 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    return (float) (z >> 40) * (1.0f / 16777216.0f);
}

// sampling distribution over one row's candidates (descending), llama.cpp order: top_k, top_p, min_p,
// temperature; returns the number kept, p[0..K) normalised
__device__ int cand_probs(const float2 * c, const Launch & P, float (&p)[TOPK_MAX]) {
    int K = P.top_k > 0 ? min(P.top_k, TOPK_MAX) : TOPK_MAX;
    while (K > 1 && !(c[K - 1].x > -INFINITY)) K--;
    const float mx = c[0].x;
    float s = 0.f;
    for (int j = 0; j < K; j++) s += (p[j] = expf(c[j].x - mx));
    if (P.top_p < 1.0f) {
        float cum = 0.f;
        for (int j = 0; j < K; j++) {
            cum += p[j] / s;
            if (cum >= P.top_p) { K = j + 1; break; }
        }
    }
    if (P.min_p > 0.0f) {
        int k = 1;
        while (k < K && p[k] >= P.min_p * p[0]) k++;
        K = k;
    }
    s = 0.f;
    for (int j = 0; j < K; j++) s += (p[j] = expf((c[j].x - mx) / P.temp));
    for (int j = 0; j < K; j++) p[j] /= s;
    return K;
}

// draw from p (optionally without token excl, renormalised)
__device__ int cand_draw(const float2 * c, const float (&p)[TOPK_MAX], int K, int excl, float u) {
    float s = 0.f;
    for (int j = 0; j < K; j++)
        if (__float_as_int(c[j].y) != excl) s += p[j];
    float x = u * s;
    int last = __float_as_int(c[0].y);
    for (int j = 0; j < K; j++) {
        const int id = __float_as_int(c[j].y);
        if (id == excl) continue;
        last = id;
        x -= p[j];
        if (x < 0.f) return id;
    }
    return last;
}

// q'(id) of a sampled draft's proposal list (prob, id) ... terminated by id -1. Sums every entry of the
// token: the draft head can hold a token twice (dv_add pads the draft vocabulary to an even row count with
// a duplicate), and the proposal probability of the token is then the sum.
__device__ __forceinline__ float dq_prob(const float2 * q, int id) {
    float s = 0.f;
    for (int j = 0; j < TOPK_MAX; j++) {
        const int qi = __float_as_int(q[j].y);
        if (qi < 0) break;
        if (qi == id) s += q[j].x;
    }
    return s;
}

// Sampled draft (single thread): q' = the sampling distribution (top_k, top_p, min_p, temp) of the draft
// head's top logits src; writes q' to dst as (prob, token id) terminated by (0, -1) (dst may alias src) and
// draws the draft from it with its own random stream, keyed by the draft's absolute position.
__device__ int draft_sample(const Launch & P, const float2 * src, float2 * dst, unsigned long long pos) {
    float p[TOPK_MAX];
    const int K = cand_probs(src, P, p);
    float x = urand(P.seed ^ 0xD1B54A32D192ED03ull, pos);
    int pick = -1, last = __float_as_int(src[0].y);
    for (int j = 0; j < K; j++) {
        const float2 e = src[j];
        last = __float_as_int(e.y);
        if (pick < 0) {
            x -= p[j];
            if (x < 0.f) pick = last;
        }
        dst[j] = make_float2(p[j], e.y);
    }
    if (K < TOPK_MAX) dst[K] = make_float2(0.f, __int_as_float(-1));
    return pick >= 0 ? pick : last;
}

// Sampled targets for the verified rows.  Point-mass drafts (argmax of the MTP): draft token d at row
// i is accepted with probability p_i(d); on rejection the row's token is drawn from p_i without d.
// Sampled drafts (Ctl::dsmp, d ~ q'_i): accepted with probability min(1, p_i(d) / q'_i(d)); on
// rejection drawn from the residual norm(max(0, p_i - q'_i)).  Either way the output is distributed
// exactly as p, and the rows after a rejection are dropped.  Rows with no draft to verify (the bonus
// row, the last prompt row) are drawn from p_i.  tg then drives the greedy acceptance logic of
// main_control unchanged.
__device__ void sample_targets(const Launch & P, int T, int tl, int * tg) {
    const Ctl * c = cB.ctl;
    const unsigned long long n = (unsigned long long) c->n;
    float p[TOPK_MAX];
    for (int i = 0; i < tl; i++) {
        const float2 * cd = cB.cand + i * TOPK_MAX;
        const int K = cand_probs(cd, P, p);
        if (P.mode == MODE_DECODE && i < T - 1 && c->dsmp) {
            const int d = c->tok[i + 1];
            const float2 * q = cB.dq + i * TOPK_MAX;
            float pd = 0.f;
            for (int j = 0; j < K; j++)
                if (__float_as_int(cd[j].y) == d) pd += p[j];
            if (urand(P.seed, 2 * (n + i)) * dq_prob(q, d) < pd) {
                tg[i] = d;
                continue;
            }
            float r[TOPK_MAX], rs = 0.f;
            for (int j = 0; j < K; j++) rs += (r[j] = fmaxf(0.f, p[j] - dq_prob(q, __float_as_int(cd[j].y))));
            const float u = urand(P.seed, 2 * (n + i) + 1);
            if (!(rs > 0.f)) { // rounding only; the row must not return the rejected d
                tg[i] = cand_draw(cd, p, K, d, u);
                return;
            }
            float x = u * rs;
            int pick = -1;
            for (int j = 0; j < K; j++) {
                if (r[j] <= 0.f) continue;
                pick = __float_as_int(cd[j].y);
                x -= r[j];
                if (x < 0.f) break;
            }
            tg[i] = pick;
            return;
        }
        if (P.mode == MODE_DECODE && i < T - 1) {
            const int d = c->tok[i + 1];
            float pd = 0.f;
            for (int j = 0; j < K; j++)
                if (__float_as_int(cd[j].y) == d) pd += p[j];
            if (urand(P.seed, 2 * (n + i)) < pd) {
                tg[i] = d;
                continue;
            }
            tg[i] = cand_draw(cd, p, K, d, urand(P.seed, 2 * (n + i) + 1));
            return;
        }
        tg[i] = cand_draw(cd, p, K, -1, urand(P.seed, 2 * (n + i) + 1));
        return;
    }
}

// ---- main-model control (single thread) ----
__device__ void main_control(int mode, int last, int T, const int * tgt) {
    Ctl * c = cB.ctl;
    const int n = c->n;
    int a, bonus = -1;
    if (mode == MODE_DECODE) {
        a = 0;
        while (a < T - 1 && tgt[a] == c->tok[a + 1]) a++;
        for (int i = 0; i < a; i++) c->out[i] = c->tok[i + 1];
        c->out[a] = tgt[a];
        c->n_out = a + 1;
        c->n_acc = a;
        bonus = tgt[a];
        c->dn_c = n + 1;
    } else {
        a = T - 1;
        c->dn_c = n + T;
        c->n_acc = 0;
        if (last) {
            bonus = tgt[0];
            c->out[0] = bonus;
            c->n_out = 1;
        } else {
            c->n_out = 0;
        }
    }
    const int m = c->mtp_valid;
    const int P = bonus >= 0 ? n + a + 1 : n + T - 1;
    int R = 0;
    for (int p = m; p <= P && R < 8; p++, R++) {
        c->mtp_pos[R] = p;
        c->mtp_tok[R] = (p >= n && p <= n + a) ? c->tok[p - n] : bonus;
    }
    c->mtp_R = R;
    c->mtp_P = P;
    c->mtp_logits = bonus >= 0;
    c->mtp_valid = P + 1;
    if (bonus >= 0) {
        c->n = P;
        c->tok[0] = bonus;
    }
}

template <int T>
__device__ void main_forward(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = cB.ctl;
    const int n = vc->n;
    const int dn_c = vc->dn_c;
    const int n_commit = P.mode == MODE_DECODE ? 1 : T;

    embed_prologue<T>();
    gsync(epoch, PH_EMBED);
    int dl = 0;
    for (int L = 0; L < NL; L++) {
        const LayerW & W = cW.L[L];
        const float * wnext = (L + 1 < NL) ? cW.L[L + 1].attn_norm : cW.onorm;
        if (W.attn) {
            attn_proj<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_PROJ);
            attn_core<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_CORE);
            attn_combine<T>(n);
            gsync(epoch, PH_ATTN_COMB);
            resid_any<T, 6>(W.o, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_O);
        } else {
            dn_proj<T>(W, dl, n, dn_c, sm);
            gsync(epoch, PH_DN_PROJ);
            dn_core<T>(W, dl, n, dn_c, n_commit, sm);
            gsync(epoch, PH_DN_CORE);
            resid_any<T, 6>(W.sout, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_SOUT);
            dl++;
        }
        ffn_gu<T>(W, sm);
        gsync(epoch, PH_FFN_GU);
        resid_any<T, 17>(W.fd, cB.qa, cB.x, true, wnext, sm);
        gsync(epoch, PH_FFN_DOWN);
    }
    write_h_rows<T>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, sm);

    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    bool am_last;
    int tl = 0;
    const bool sampling = P.temp > 0.0f && !P.score;
    if (P.mode == MODE_DECODE || P.score) {
        if (sampling) am_last = lm_head<T, true>(0, false, sm, s_tok);
        else if (P.score == 2) am_last = lm_head<T, false, true>(0, false, sm, s_tok);
        else am_last = lm_head<T>(0, false, sm, s_tok);
        tl = T;
    } else if (P.last) {
        am_last = sampling ? lm_head<1, true>(T - 1, false, sm, s_tok) : lm_head<1>(T - 1, false, sm, s_tok);
        tl = 1;
    } else {
        am_last = blockIdx.x == 0;
    }
    if (am_last && threadIdx.x == 0) {
        int tg[MAXT];
        for (int i = 0; i < tl; i++) tg[i] = s_tok[i];
        if (sampling) sample_targets(P, T, tl, tg);
        for (int i = 0; i < tl; i++) cB.ctl->tgt[i] = tg[i];
        main_control(P.mode, P.last, T, P.score ? s_tok + T - 1 : tg);
        __threadfence();
    }
    gsync(epoch, PH_LMHEAD);
}

// one MTP pass over R consecutive positions; rows read from ctl (catch-up) or a draft row.
template <int R>
__device__ __noinline__ void mtp_pass(const Launch & P, int row0, bool draft_row, int draft_idx, bool logits, uint8_t * /*sm*/,
                                     unsigned & epoch) {
    uint8_t * const sm = smem;
    volatile Ctl * vc = cB.ctl;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int pos0;
    if (draft_row) pos0 = vc->mtp_P + draft_idx;
    else pos0 = vc->mtp_pos[row0];

    // M0: concat(enorm(embed(tok)), hnorm(h_prev)) -> qe
    if (blockIdx.x < R) {
        const int r = blockIdx.x;
        const int pos = pos0 + r;
        const int tok = draft_row ? vc->tok[draft_idx] : vc->mtp_tok[row0 + r];
        const float * hsrc = draft_row ? cB.hm : (pos >= 1 ? cB.h_hist + (size_t) ((pos - 1) % H_RING) * D : nullptr);
        float * se = reinterpret_cast<float *>(sm);
        float * sh = se + D;
        float * red = sh + D;
        const uint8_t * erow = stage_embd_row(tok, reinterpret_cast<uint8_t *>(red + 32));
        float ae = 0.f, ah = 0.f;
        for (int e = tid; e < D; e += NT) {
            const float v = dequant_q4k(erow, e);
            const float hv = hsrc ? __ldcg(hsrc + e) : 0.f;
            se[e] = v;
            sh[e] = hv;
            ae += v * v;
            ah += hv * hv;
        }
        ae = bsum(ae, red);
        ah = bsum(ah, red + 16);
        const float rse = rsqrtf(ae / (float) D + EPS), rsh = rsqrtf(ah / (float) D + EPS);
        for (int j = warp; j < 2 * U_D; j += NW) {
            const int e = 32 * (j % U_D) + lane;
            const float v = j < U_D ? se[e] * rse * cW.enorm[e] : sh[e] * rsh * cW.hnorm[e];
            q32_store(cB.qe, r, j, v, 0.f, false);
        }
    }
    gsync(epoch, PH_MTP_M0);
    const LayerW & W = cW.L[NL];
    resid_any<R, 10>(cW.eh, cB.qe, cB.xm, false, W.attn_norm, sm);
    gsync(epoch, PH_MTP_EH);
    attn_proj<R>(W, 16, pos0, sm);
    gsync(epoch, PH_MTP_ATTN_PROJ);
    attn_core<R>(W, 16, pos0, sm);
    gsync(epoch, PH_MTP_ATTN_CORE);
    attn_combine<R>(pos0);
    gsync(epoch, PH_MTP_ATTN_COMB);
    resid_any<R, 6>(W.o, cB.qo, cB.xm, true, W.post_norm, sm);
    gsync(epoch, PH_MTP_RESID_O);
    ffn_gu<R>(W, sm);
    gsync(epoch, PH_MTP_FFN_GU);
    resid_any<R, 17>(W.fd, cB.qa, cB.xm, true, cW.shnorm, sm);
    gsync(epoch, PH_MTP_FFN_DOWN);
    if (logits) {
        write_h_rows<R>(cB.xm, cB.qn, cW.shnorm, cB.hm, 0, false, sm);
        int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
        const int k = draft_row ? draft_idx : 0; // this pass produces draft k + 1 (tok[k + 1])
        // sampled drafts need the draft head's top logits (TOPK path); argmax drafts only the reduction
#if MK_SPEC_LOG
        float2 * const tl = cB.dlog + k * (TOPK_MAX + 1);
        const bool lastb = lm_head<1, true, false, true>(R - 1, true, sm, s_tok, tl);
#else
        float2 * const tl = cB.dq + k * TOPK_MAX;
        const bool lastb = P.spec_smp ? lm_head<1, true, false, true>(R - 1, true, sm, s_tok, tl)
                                      : lm_head<1, false, false, true>(R - 1, true, sm, s_tok);
#endif
        if (lastb && threadIdx.x == 0) {
            Ctl * c = cB.ctl;
            const float * mm = misc(sm);
            const float conf = __expf(mm[LM_BEST] - mm[LM_LSE]);
            const float chain = (k == 0 ? 1.f : c->dchain) * conf;
#if MK_SPEC_LOG
            cB.dlog[k * (TOPK_MAX + 1) + TOPK_MAX] = make_float2(mm[LM_LSE], mm[LM_BEST]);
#endif
            const float f = P.gate_chain ? chain : conf;
            c->tok[1 + k] = P.spec_smp ? draft_sample(P, tl, cB.dq + k * TOPK_MAX, (unsigned long long) c->mtp_P + 1 + k)
                                       : s_tok[0];
            c->dsmp = P.spec_smp;
            c->dconf[k] = conf;
            c->dchain = chain;
            c->nd_prod = f < P.gate_drop[k] ? k : k + 1;
            c->nd_stop = f < P.gate_stop[k] || f < P.gate_drop[k];
            __threadfence();
        }
        gsync(epoch, PH_MTP_LMHEAD);
    }
}

template <int T>
__device__ void run_main(const Launch & P, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    main_forward<T>(P, sm, epoch);
}

template <int T>
__device__ void bench_phase(int which, int layer, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    for (int it = 0; it < 4; it++) {
        const int L = layer + 4 * it; // different layers: weights are not L2-resident, code is
        const LayerW & W = cW.L[L];
        if (which == 0) ffn_gu<T>(W, sm);
        else if (which == 1) {
            resid_any<T, 17>(W.fd, cB.qa, cB.x, true, W.attn_norm, sm);
        } else if (which == 2) {
            if (!W.attn) dn_proj<T>(W, 0, 0, 0, sm);
        }
        gsync(epoch, PH_FFN_GU);
    }
}

__global__ void __launch_bounds__(NT, 1) mk_cycle(Launch P) {
    uint8_t * const sm = smem;
    unsigned epoch = 0;
    if (P.mode == MODE_BENCH) {
        switch (P.T) {
            case 1: bench_phase<1>(P.last, P.ndraft, sm, epoch); break;
            case 4: bench_phase<4>(P.last, P.ndraft, sm, epoch); break;
        }
        return;
    }
    if (cB.prof && blockIdx.x == 0 && threadIdx.x == 0) {
        cB.prof[0] = gtimer() << 8;
        s_bar_k = 0;
    }
    switch (P.T) {
        case 1: run_main<1>(P, sm, epoch); break;
        case 2: run_main<2>(P, sm, epoch); break;
        case 3: run_main<3>(P, sm, epoch); break;
        case 4: run_main<4>(P, sm, epoch); break;
        case 5: run_main<5>(P, sm, epoch); break;
    }
    if (P.ndraft == 0 || P.score) return;
    volatile Ctl * vc = cB.ctl;
    const int Rtot = vc->mtp_R;
    const bool lg = vc->mtp_logits;
    int done = 0;
    while (done < Rtot) {
        const int R = min(MAXT, Rtot - done);
        const bool last_pass = done + R == Rtot;
        const bool want = lg && last_pass && P.ndraft > 0;
        switch (R) {
            case 1: mtp_pass<1>(P, done, false, 0, want, sm, epoch); break;
            case 2: mtp_pass<2>(P, done, false, 0, want, sm, epoch); break;
            case 3: mtp_pass<3>(P, done, false, 0, want, sm, epoch); break;
            case 4: mtp_pass<4>(P, done, false, 0, want, sm, epoch); break;
            case 5: mtp_pass<5>(P, done, false, 0, want, sm, epoch); break;
        }
        done += R;
    }
    if (lg) {
        // nd_stop is written before the PH_MTP_LMHEAD barrier, so every block takes the same exit
        for (int i = 1; i < P.ndraft && !vc->nd_stop; i++) mtp_pass<1>(P, 0, true, i, true, sm, epoch);
    }
}

#include "prefill.cuh"

// MK_FAULT (tests of the error path only, see Impl::fault_*): a kernel that traps, i.e. a sticky error
__global__ void pk_fault() { __trap(); }

// silu(gate) * up from the fp32 outputs of two GEMMs, written like mmh<.., true>'s epilogue (fp16,
// hperm order): the gate / up pairs it cannot fuse (different types, or Q6_K / Q8_0).
__global__ void __launch_bounds__(256) pk_glu(const float * __restrict__ g, const float * __restrict__ u, __half * __restrict__ out, int ld) {
    // thread = 4 consecutive values (one float4 of g and u: a warp reads 512 contiguous bytes of each); within a
    // 16-group, values 4 j .. 4 j + 3 go to hperm slots 2 j, 2 j + 1 and 8 + 2 j, 9 + 2 j (two 4-byte stores)
    const size_t t = blockIdx.x;
    const float4 * g4 = reinterpret_cast<const float4 *>(g + t * ld);
    const float4 * u4 = reinterpret_cast<const float4 *>(u + t * ld);
    __half * o = out + t * ld;
    for (int k = threadIdx.x; k < ld / 4; k += blockDim.x) {
        const float4 a = g4[k], b = u4[k];
        const float gv[4] = {a.x, a.y, a.z, a.w}, uv[4] = {b.x, b.y, b.z, b.w};
        __half hv[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const float v = gv[e] / (1.0f + expf(-gv[e])) * uv[e];
            hv[e] = __float2half_rn(fminf(fmaxf(v, -65504.f), 65504.f));
        }
        __half * og = o + 16 * (k >> 2) + 2 * (k & 3);
        *reinterpret_cast<__half2 *>(og) = __halves2half2(hv[0], hv[1]);
        *reinterpret_cast<__half2 *>(og + 8) = __halves2half2(hv[2], hv[3]);
    }
}

// ------------------------------------------------------------------------------------------
// host side: loading / repacking
// ------------------------------------------------------------------------------------------

uint16_t f2h_bits(float f) {
    __half h = __float2half(f);
    return *reinterpret_cast<uint16_t *>(&h);
}

template <class F>
void parallel_rows(int64_t n, F f) {
    int nt = std::max(1u, std::thread::hardware_concurrency());
    std::vector<std::thread> th;
    for (int i = 0; i < nt; i++) {
        th.emplace_back([&, i] {
            for (int64_t r = i; r < n; r += nt) f(r);
        });
    }
    for (auto & t : th) t.join();
}

// GGUF Q6_K block (210 B: ql[128] | qh[64] | int8 scales[16] | fp16 d) from 256 integers q in [-32, 31],
// one scale per 16 of them and the bits of d.
void put_q6k(uint8_t * blk, const int * q, const int8_t * sc, uint16_t dbits) {
    uint8_t *ql = blk, *qh = blk + 128;
    for (int n = 0; n < 2; n++)
        for (int l = 0; l < 32; l++) {
            const int v0 = q[128 * n + l] + 32, v1 = q[128 * n + l + 32] + 32;
            const int v2 = q[128 * n + l + 64] + 32, v3 = q[128 * n + l + 96] + 32;
            ql[64 * n + l] = (uint8_t) ((v0 & 0xF) | ((v2 & 0xF) << 4));
            ql[64 * n + l + 32] = (uint8_t) ((v1 & 0xF) | ((v3 & 0xF) << 4));
            qh[32 * n + l] = (uint8_t) (((v0 >> 4) & 3) | (((v1 >> 4) & 3) << 2) | (((v2 >> 4) & 3) << 4) | (((v3 >> 4) & 3) << 6));
        }
    memcpy(blk + 192, sc, 16);
    memcpy(blk + 208, &dbits, 2);
}

// Q3_K or IQ3_S row -> Q6_K with the same values. Both store a small integer times d times an integer
// sub-block scale (Q3_K: d (sc - 32) q, q in [-4, 3], per 16; IQ3_S: d (1 + 2 s) q, |q| <= 15, per 32),
// so q = value / (d * scale) is exact and Q6_K holds it with the same d and scale. f: the row dequantized.
void reencode_q6k(uint32_t type, const uint8_t * src, const float * f, uint8_t * out, int64_t K) {
    for (int64_t b = 0; b < K / 256; b++) {
        const uint8_t * blk = src + b * 110;
        const float * y = f + b * 256;
        uint16_t dbits;
        int8_t sc[16];
        if (type == gguf::Q3_K) { // hmask[32] | qs[64] | scales[12] (6 bits each) | fp16 d
            memcpy(&dbits, blk + 108, 2);
            uint32_t aux[4];
            memcpy(aux, blk + 96, 12);
            const uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu, tmp = aux[2];
            aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
            aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
            aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
            aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
            for (int i = 0; i < 16; i++) sc[i] = (int8_t) ((int) reinterpret_cast<const uint8_t *>(aux)[i] - 32);
        } else { // IQ3_S: fp16 d | qs[64] | qh[8] | signs[32] | scales[4] (4 bits per 32)
            memcpy(&dbits, blk, 2);
            for (int i = 0; i < 16; i++) sc[i] = (int8_t) (1 + 2 * ((blk[106 + i / 4] >> (4 * ((i / 2) & 1))) & 0xf));
        }
        const float d = ggml_fp16_to_fp32(dbits);
        int q[256];
        for (int e = 0; e < 256; e++) {
            const float s = d * (float) sc[e / 16];
            q[e] = s == 0.0f ? 0 : (int) lrintf(y[e] / s);
        }
        put_q6k(out + b * 210, q, sc, dbits);
    }
}

// IQ4_NL row (18 B blocks: fp16 d | 16 bytes of 4-bit indices) -> Q8_0 (fp16 d | 32 int8) with the same
// d and q = kvalues_iq4nl[index]: the same values.
void reencode_iq4nl_q80(const uint8_t * src, uint8_t * out, int64_t K) {
    static const int8_t kv[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    for (int64_t b = 0; b < K / 32; b++) {
        const uint8_t * s = src + b * 18;
        uint8_t * o = out + b * 34;
        memcpy(o, s, 2);
        for (int j = 0; j < 16; j++) {
            o[2 + j] = (uint8_t) kv[s[2 + j] & 0xf];
            o[2 + j + 16] = (uint8_t) kv[s[2 + j] >> 4];
        }
    }
}

// GGUF IQ4_XS blocks (136 B: fp16 d, u16 scales_h, u8 scales_l[4], u8 qs[128]) -> the layout rows_iq4x and the
// prefill GEMM read: pairs of blocks in 272 B (16-byte aligned, no padding): the two 8-byte GGUF headers, then
// each block's 128 index bytes in Q4_K order (see rows_iq4x). nblocks must be even.
void repack_iq4xs(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t *s = src + i * 136, *qs = s + 8;
        uint8_t * pr = dst + (i >> 1) * 272;
        memcpy(pr + 8 * (i & 1), s, 8);
        uint8_t * o = pr + 16 + 128 * (i & 1);
        auto idx = [&](int v) {
            const int ib = v >> 5, j = v & 31;
            return j < 16 ? qs[16 * ib + j] & 0xf : qs[16 * ib + j - 16] >> 4;
        };
        for (int p = 0; p < 4; p++)
            for (int l = 0; l < 32; l++) o[32 * p + l] = (uint8_t) (idx(64 * p + l) | (idx(64 * p + 32 + l) << 4));
    }
}

// GGUF IQ4_NL (8 blocks of 18 B: fp16 d, u8 qs[16]) per 256 weights -> 144 B: the eight fp16 d, then the 128 index
// bytes in the order of repack_iq4xs (rows_iq4nl, prefill TIQ4NL)
void repack_iq4nl(const uint8_t * src, uint8_t * dst, size_t nblocks) {
    for (size_t i = 0; i < nblocks; i++) {
        const uint8_t * s = src + i * 144;
        uint8_t * o = dst + i * 144;
        for (int ib = 0; ib < 8; ib++) memcpy(o + 2 * ib, s + 18 * ib, 2);
        auto idx = [&](int v) {
            const uint8_t * qs = s + 18 * (v >> 5) + 2;
            const int j = v & 31;
            return j < 16 ? qs[j] & 0xf : qs[j - 16] >> 4;
        };
        for (int p = 0; p < 4; p++)
            for (int l = 0; l < 32; l++) o[16 + 32 * p + l] = (uint8_t) (idx(64 * p + l) | (idx(64 * p + 32 + l) << 4));
    }
}

} // namespace

struct MKEngine::Impl {
    gguf::File file;
    ModelW hw{};
    Bufs hb{};
    std::vector<void *> allocs;
    void * embd_host = nullptr;
    int ctx;
    int nsm = 0;
    float smp_temp = 0.f, smp_top_p = 1.f, smp_min_p = 0.f; // sampling (temp 0: greedy)
    int smp_top_k = TOPK_MAX;
    unsigned long long smp_seed = 0;
    int ndraft;          // MTP drafts the next launch produces (at most; the gate may stop earlier)
    int nd_avail;        // drafts produced by the previous launch (verified by the next decode cycle)
    bool all_acc = true; // the last decode cycle accepted all of its drafts
    int last_acc = 0;    // drafts the last decode cycle accepted
    // draft-depth policy (MKEngine::set_policy). Default: up to n_draft drafts, the on-device gate on the running
    // product c_k of the MTP's own top-1 probabilities stops drafting after draft k >= 2 when c_k < 0.4 and drops
    // the 4th draft (the T = 5 verify row) when c_4 < 0.4; sampled drafts when sampling. Tuned on replayed logs
    // + live A/B (notes/agent_spec.md). The previous behaviour: "depth=0,drop=0,stop=0,chain=0,h1=0".
    struct Policy {
        int depth = 1;       // 0: n_draft - 1 after a cycle with a rejection (at least 3), 1: always n_draft,
                             // 2: 1 + the accepted count of the last cycle (clamped to [dmin, n_draft])
        int dmin = 1;
        float gate_drop[4] = {0.f, 0.f, 0.f, 0.4f}, gate_stop[4] = {0.f, 0.4f, 0.4f, 0.4f}; // on-device early stop
        int gate_chain = 1;
        int h1 = 1; // sampling: sampled drafts (Launch::spec_smp)
    } pol;
#if MK_SPEC_LOG
    FILE * slog = nullptr;               // MK_SPEC_LOG_FILE: one JSON line per decode cycle
    std::vector<float2> slog_q;          // dlog of the drafts the next cycle verifies
    void slog_drafts() {
        if (!slog) return;
        slog_q.resize((size_t) MAXT * (TOPK_MAX + 1));
        CK(cudaMemcpy(slog_q.data(), hb.dlog, slog_q.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    }
#endif
    std::vector<int> seq; // the sequence so far: prompt(s) and emitted tokens (position = index)
    // checkpoint of the state right after the last prefill (recurrent state, conv ring, h ring, ctl):
    // lets a caller rewind to the end of the last prompt (KV rows before it are never rewritten)
    // The checkpoint lives in pinned host memory (0.16 GB of VRAM saved; a save or restore moves 0.17 GB over PCIe,
    // 8-13 ms). MK_CK_DEVICE=1 keeps it in device memory.
    // Neither copy is on the critical path (F9). Save: a device-to-device snapshot into the idle batched-prefill
    // scratch on stream 0 (~0.4 ms), then the device-to-host copy on the non-blocking stream ck_st, which runs
    // under the first decode launch (the decode kernels never touch the prefill scratch). Restore: the host-to-device
    // copy runs on ck_st in CK_NG groups of DeltaNet layers, issued once the next prefill has uploaded its tokens
    // (ck_issue_restore: nothing on stream 0 then queues behind it on the copy engine); stream 0 waits for a group's event
    // only before the first kernel that reads that group's state (pf_run: dn_prep of the group's first layer), so the
    // copy overlaps the embedding / GEMMs of the layers before it. Same bytes, same order of use: bit-identical.
    // If the scratch is too small for the snapshot (MK_PF_CAP below ~640), the save copies straight from the state
    // and stream 0 waits for it before the state is next modified (still off the prompt's critical path).
    float *ck_dstate = nullptr, *ck_conv = nullptr, *ck_hhist = nullptr;
    void * ck_host = nullptr;
    Ctl ck_ctl;
    int ck_pos = -1;
    double ck_ms = 0; // last save / restore enqueue time (MK_MEMLOG)
    static constexpr size_t CK_NS = (size_t) 48 * NVH * 128 * 128, CK_NC = (size_t) 48 * CONV_RING * QKV, CK_NH = (size_t) H_RING * D;
    static constexpr int CK_NG = 5;
    static constexpr int ck_gb[CK_NG + 1] = {0, 1, 4, 12, 24, 48}; // restore group g: DeltaNet layers [ck_gb[g], ck_gb[g + 1])
    cudaStream_t ck_st = nullptr;
    cudaEvent_t ck_ev_src = nullptr, ck_ev_saved = nullptr, ck_ev_grp[CK_NG] = {};
    bool ck_saving = false;      // a save's device-to-host copy may still be running on ck_st
    bool ck_save_direct = false; // ... and it reads the live state (no snapshot): stream 0 must wait before modifying it
    bool ck_rs_due = false;      // a restore whose copies are not issued yet (ck_issue_restore)
    int ck_rs_next = CK_NG;      // first restore group stream 0 has not waited for (CK_NG: no restore pending)
    struct CkChunk { char * dev; char * scr; char * host; size_t n; };
    std::vector<CkChunk> ck_plan; // snapshot: state -> prefill scratch (stream 0) -> pinned host (ck_st)

    // Allocated at construction: cudaHostAlloc of the 0.16 GB takes 0.1-0.5 s under WSL2, which used to land in the
    // first prompt's time (inside the first ck_save).
    void ck_alloc() {
        const char * cd = getenv("MK_CK_DEVICE");
        if (!(cd && cd[0] == '1') && cudaHostAlloc(&ck_host, (CK_NS + CK_NC + CK_NH) * 4, cudaHostAllocDefault) == cudaSuccess) {
            ck_dstate = static_cast<float *>(ck_host);
            ck_conv = ck_dstate + CK_NS;
            ck_hhist = ck_conv + CK_NC;
            CK(cudaStreamCreateWithFlags(&ck_st, cudaStreamNonBlocking));
            CK(cudaEventCreateWithFlags(&ck_ev_src, cudaEventDisableTiming));
            CK(cudaEventCreateWithFlags(&ck_ev_saved, cudaEventDisableTiming));
            for (auto & e : ck_ev_grp) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        } else {
            cudaGetLastError();
            ck_host = nullptr;
            ck_dstate = dalloc<float>(CK_NS, false);
            ck_conv = dalloc<float>(CK_NC, false);
            ck_hhist = dalloc<float>(CK_NH, false);
        }
        mem_mark(ck_host ? "DeltaNet checkpoint (host)" : "DeltaNet checkpoint");
    }
    // snapshot plan over the prefill scratch buffers (after pf_init); empty when they are too small
    void ck_make_plan() {
        ck_plan.clear();
        if (!ck_host) return;
        const size_t cap = hp.cap;
        const std::pair<void *, size_t> scr[] = {
            {hp.y0, cap * FF * 4}, {hp.y1, cap * FF * 4}, {hp.xh2, cap * FF * 2}, {hp.y2, cap * VD * 4}, {hp.x, cap * D * 4},
            {hp.xm, cap * D * 4},  {hp.dq, cap * QKV * 2}, {hp.xh, cap * 2 * D * 2}, {hp.qh, cap * NH * HD * 2}};
        const std::tuple<void *, void *, size_t> reg[] = {
            {hb.dstate, ck_dstate, CK_NS * 4}, {hb.conv, ck_conv, CK_NC * 4}, {hb.h_hist, ck_hhist, CK_NH * 4}};
        size_t si = 0, so = 0; // scratch buffer, offset in it
        for (const auto & r : reg) {
            size_t done = 0;
            while (done < std::get<2>(r)) {
                while (si < std::size(scr) && scr[si].second - so < 256) si++, so = 0;
                if (si == std::size(scr)) {
                    ck_plan.clear();
                    return;
                }
                const size_t n = std::min(std::get<2>(r) - done, (scr[si].second - so) & ~(size_t) 255);
                ck_plan.push_back({static_cast<char *>(std::get<0>(r)) + done, static_cast<char *>(scr[si].first) + so,
                                   static_cast<char *>(std::get<1>(r)) + done, n});
                done += n;
                so += n;
            }
        }
    }
    // host: the last save's device-to-host copy has finished (the watchdog covers the wait)
    void ck_wait_saved() {
        if (!ck_saving) return;
        wait_event(ck_ev_saved, "the checkpoint save", std::max(wd_limit, 120.0));
        ck_saving = ck_save_direct = false;
    }
    // stream 0: before modifying the live state, wait for a save that reads it directly (no snapshot)
    void ck_guard_state() {
        if (!(ck_saving && ck_save_direct)) return;
        CK(cudaStreamWaitEvent(0, ck_ev_saved, 0));
        ck_save_direct = false;
    }
    // Issues a due restore's host-to-device copies on ck_st (5 groups, an event after each). Deferred from ck_restore
    // to just after the next prefill's token upload: a host-to-device copy on stream 0 (ctl, token ids, draft
    // vocabulary rows) queued behind the 0.17 GB on the copy engine would stall the host for the whole restore.
    void ck_issue_restore() {
        if (!ck_rs_due) return;
        ck_rs_due = false;
        ck_wait_saved(); // (the copy reads the host buffer)
        // stream 0's earlier work on the state is complete before ck_st overwrites it
        CK(cudaEventRecord(ck_ev_src, 0));
        CK(cudaStreamWaitEvent(ck_st, ck_ev_src, 0));
        CK(cudaMemcpyAsync(hb.h_hist, ck_hhist, CK_NH * 4, cudaMemcpyHostToDevice, ck_st)); // group 0
        constexpr size_t SL = (size_t) NVH * 128 * 128, CL = (size_t) CONV_RING * QKV; // floats per DeltaNet layer
        for (int g = 0; g < CK_NG; g++) {
            const size_t l0 = ck_gb[g], nl = ck_gb[g + 1] - ck_gb[g];
            CK(cudaMemcpyAsync(hb.conv + l0 * CL, ck_conv + l0 * CL, nl * CL * 4, cudaMemcpyHostToDevice, ck_st));
            CK(cudaMemcpyAsync(hb.dstate + l0 * SL, ck_dstate + l0 * SL, nl * SL * 4, cudaMemcpyHostToDevice, ck_st));
            CK(cudaEventRecord(ck_ev_grp[g], ck_st));
        }
        ck_rs_next = 0;
    }
    // stream 0: wait for the restore groups up to the one holding DeltaNet layer dl (-1: all of them)
    void ck_wait_restore(int dl = -1) {
        ck_issue_restore();
        if (ck_rs_next >= CK_NG || (dl >= 0 && dl < ck_gb[ck_rs_next])) return;
        int g = CK_NG - 1;
        if (dl >= 0)
            while (ck_gb[g] > dl) g--;
        CK(cudaStreamWaitEvent(0, ck_ev_grp[g], 0));
        ck_rs_next = g + 1;
    }
    void ck_save() {
        ck_wait_restore();
        const auto t0 = std::chrono::steady_clock::now();
        if (!ck_host) { // device checkpoint: stream-ordered device copies
            CK(cudaMemcpyAsync(ck_dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(ck_conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(ck_hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToDevice, 0));
        } else {
            ck_wait_saved(); // the previous save still reading the scratch / writing the host buffer (normally long done)
            for (const auto & c : ck_plan) CK(cudaMemcpyAsync(c.scr, c.dev, c.n, cudaMemcpyDeviceToDevice, 0));
            CK(cudaEventRecord(ck_ev_src, 0));
            CK(cudaStreamWaitEvent(ck_st, ck_ev_src, 0));
            if (!ck_plan.empty()) {
                for (const auto & c : ck_plan) CK(cudaMemcpyAsync(c.host, c.scr, c.n, cudaMemcpyDeviceToHost, ck_st));
            } else {
                CK(cudaMemcpyAsync(ck_dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToHost, ck_st));
                CK(cudaMemcpyAsync(ck_conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToHost, ck_st));
                CK(cudaMemcpyAsync(ck_hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToHost, ck_st));
                ck_save_direct = true;
            }
            CK(cudaEventRecord(ck_ev_saved, ck_st));
            ck_saving = true;
        }
        ck_ctl = hctl; // prefill() read ctl back after its last launch: the host copy is the device's
        ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (memlog)
            fprintf(stderr, "[mk-mem] checkpoint save enqueued in %.2f ms (%s)\n", ck_ms,
                    !ck_host ? "device" : ck_plan.empty() ? "host, direct" : "host, via the prefill scratch");
        ck_pos = ck_ctl.dn_c;
    }
    void ck_restore() {
        ck_wait_restore();
        const auto t0 = std::chrono::steady_clock::now();
        if (!ck_host) {
            ck_guard_state();
            CK(cudaMemcpyAsync(hb.dstate, ck_dstate, CK_NS * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(hb.conv, ck_conv, CK_NC * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(hb.h_hist, ck_hhist, CK_NH * 4, cudaMemcpyDeviceToDevice, 0));
        } else {
            ck_wait_saved();
            ck_rs_due = true; // issued by the next pf_run after its token upload, or by whatever touches the state first
        }
        ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (memlog) fprintf(stderr, "[mk-mem] checkpoint restore enqueued in %.2f ms (%s)\n", ck_ms, ck_host ? "host" : "device");
        hctl = ck_ctl;
        hctl.mtp_valid = std::min(hctl.mtp_valid, ck_pos);
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        seq.resize(ck_pos);
    }
    // draft vocabulary: ids of head_d rows (base list + tokens added from the sequence)
    std::vector<int> dv_ids;
    std::vector<uint8_t> dv_in;
    int dv_base = 0;

    // Adds the tokens not yet in the draft head's vocabulary (appends their ids).
    void dv_add(const int * toks, size_t n) {
        if (dv_in.empty()) return; // full-vocabulary draft head
        const int n0 = (int) dv_ids.size();
        for (size_t i = 0; i < n && (int) dv_ids.size() < dv_base + DV_EXTRA - 1; i++)
            if (!dv_in[toks[i]]) {
                dv_in[toks[i]] = 1;
                dv_ids.push_back(toks[i]);
            }
        if ((int) dv_ids.size() == n0) return;
        if (dv_ids.size() & 1) dv_ids.push_back(dv_ids.back()); // the draft lm_head takes rows in pairs
        const int k = (int) dv_ids.size() - n0;
        Mat & m = hw.head_d;
        CK(cudaMemcpy(const_cast<int *>(hw.draft_ids) + n0, dv_ids.data() + n0, k * 4, cudaMemcpyHostToDevice));
        m.N = (int) dv_ids.size();
        CK(cudaMemcpyToSymbol(cW, &m.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    void dv_reset() {
        if (dv_in.empty() || (int) dv_ids.size() == dv_base) return;
        for (size_t i = dv_base; i < dv_ids.size(); i++) dv_in[dv_ids[i]] = 0;
        for (int i = 0; i < dv_base; i++) dv_in[dv_ids[i]] = 1; // a padding duplicate may have been cleared
        dv_ids.resize(dv_base);
        hw.head_d.N = dv_base;
        CK(cudaMemcpyToSymbol(cW, &hw.head_d.N, sizeof(int), offsetof(ModelW, head_d) + offsetof(Mat, N)));
    }
    Ctl hctl{};

    // Watchdog. Every wait for the device in the run-time paths goes through wait_device(), which records an
    // event and polls it with a deadline (MK_WATCHDOG: seconds per decode launch, default 60; prefill chunks and
    // checkpoint copies get at least 120 s; 0 = the runtime's plain blocking waits). The CUDA runtime's own waits
    // never time out: a GPU / driver fault the WSL2 guest is not told about (2026-10-12 04:43, nvlddmkm event 153
    // on the Windows side, no error in the guest) left the host thread spinning for 33 minutes with an idle GPU.
    // The watchdog turns that into a message and exit code 3. The first wd_spin_s of a wait spin (a decode launch
    // finishes in 20-35 ms, so no latency is added); after that it sleeps between polls.
    cudaEvent_t wd_ev = nullptr;
    double wd_limit = 60.0, wd_spin_s = 0.25;
    int wd_T = 0, wd_mode = -1;
    void wait_device(const char * what, double limit) {
        if (limit <= 0.0 || !wd_ev) {
            CK(cudaStreamSynchronize(0));
            return;
        }
        CK(cudaEventRecord(wd_ev, 0));
        wait_event(wd_ev, what, limit);
    }
    // the same deadline for an event recorded elsewhere (the checkpoint copy stream)
    void wait_event(cudaEvent_t ev, const char * what, double limit) {
        if (limit <= 0.0) {
            CK(cudaEventSynchronize(ev));
            return;
        }
        const auto t0 = std::chrono::steady_clock::now();
        for (;;) {
            const cudaError_t e = cudaEventQuery(ev);
            if (e == cudaSuccess) return;
            if (e != cudaErrorNotReady) CK(e);
            const double el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (el > limit) {
                const cudaError_t le = cudaPeekAtLastError();
                fprintf(stderr,
                        "[mk] watchdog: the GPU did not finish %s within %g s (mode %d, T %d, position %d, committed %d, "
                        "launches %ld, last CUDA error: %s). The device or its driver is stuck: exiting with code 3.\n",
                        what, limit, wd_mode, wd_T, hctl.n, hctl.dn_c, launches_seen, cudaGetErrorString(le));
                fflush(stderr);
                _exit(3);
            }
            if (el > wd_spin_s) std::this_thread::sleep_for(std::chrono::microseconds(el > 2.0 ? 2000 : 100));
        }
    }

    // MK_MEMLOG=1: device memory accounting. Each mark prints the bytes dalloc'd since the previous mark
    // and the drop of cudaMemGetInfo's free memory over the same span (the difference is allocation
    // granularity, module code, local memory or another process).
    const bool memlog = getenv("MK_MEMLOG") != nullptr;
    size_t alloc_bytes = 0, mark_alloc = 0, mark_free = 0;
    long launches_seen = 0, pf_runs_seen = 0;
    void mem_mark(const char * tag) {
        if (!memlog) return;
        CK(cudaDeviceSynchronize());
        size_t fr = 0, tot = 0;
        CK(cudaMemGetInfo(&fr, &tot));
        if (mark_free)
            fprintf(stderr, "[mk-mem] %-26s alloc %9.2f MiB   free drop %9.2f MiB   (free %9.1f MiB)\n", tag,
                    (alloc_bytes - mark_alloc) / 1048576.0, ((double) mark_free - (double) fr) / 1048576.0, fr / 1048576.0);
        else
            fprintf(stderr, "[mk-mem] %-26s free %.1f MiB of %.1f MiB\n", tag, fr / 1048576.0, tot / 1048576.0);
        mark_free = fr;
        mark_alloc = alloc_bytes;
    }

    size_t wbytes = 0;
    bool profiling = false;
    std::vector<double> ph_ns = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> ph_n = std::vector<long>(PH_COUNT, 0);
    double launch_ns = 0;
    long launches = 0;
    // min over launches of each barrier-to-barrier interval, keyed by (T, barrier index)
    std::vector<std::vector<double>> best_ns = std::vector<std::vector<double>>(MAXT + 1);
    std::vector<std::vector<int>> best_tag = std::vector<std::vector<int>>(MAXT + 1);
    std::vector<std::vector<double>> best_fill = std::vector<std::vector<double>>(MAXT + 1);
    std::vector<double> arr_first = std::vector<double>(PH_COUNT, 0.0), arr_med = std::vector<double>(PH_COUNT, 0.0),
                        arr_last = std::vector<double>(PH_COUNT, 0.0);
    std::vector<long> arr_n = std::vector<long>(PH_COUNT, 0);

    // Device buffers come from the default memory pool (stream-ordered allocator, never trimmed): it packs the
    // allocations, where cudaMalloc rounds each one up to whole 2 MiB pages (~0.5 GB over the ~900 weight tensors).
    // MK_CUDAMALLOC=1: plain cudaMalloc.
    bool pool_init = false, use_pool = true;
    template <class T> T * dalloc(size_t n, bool zero = true) {
        void * p;
        if (!pool_init) {
            pool_init = true;
            const char * cm = getenv("MK_CUDAMALLOC");
            int ok = 0;
            cudaDeviceGetAttribute(&ok, cudaDevAttrMemoryPoolsSupported, 0);
            use_pool = ok && !(cm && cm[0] == '1');
            if (use_pool) {
                cudaMemPool_t pool;
                CK(cudaDeviceGetDefaultMemPool(&pool, 0));
                unsigned long long thr = ~0ull; // keep freed memory in the pool (nothing is freed before exit anyway)
                CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr));
            }
        }
        if (use_pool) {
            CK(cudaMallocAsync(&p, n * sizeof(T) + 16, 0));
            CK(cudaStreamSynchronize(0));
        } else {
            CK(cudaMalloc(&p, n * sizeof(T) + 16));
        }
        if (zero) CK(cudaMemset(p, 0, n * sizeof(T) + 16));
        allocs.push_back(p);
        alloc_bytes += n * sizeof(T) + 16;
        return (T *) p;
    }

    // A tensor of the file in a type the kernels read. Q3_K and IQ3_S are re-encoded as Q6_K and
    // IQ4_NL as Q8_0, exactly: every weight is a small integer times an fp16 scale in both (checked
    // bit for bit against ggml's dequantization of the source row). `to == T_F32` dequantizes
    // (ssm_alpha / ssm_beta, which are folded into fp32 rows).
    struct Src {
        uint32_t type = 0;
        std::vector<uint64_t> ne;
        const uint8_t * data = nullptr;
        uint64_t nbytes = 0;
        std::vector<uint8_t> own; // the re-encoded copy, when there is one
    };
    int n_conv = 0;
    double conv_s = 0;

    Src want(const std::string & name, int to = -1) {
        const auto & t = file.get(name);
        Src s;
        s.type = t.type;
        s.ne = t.ne;
        s.data = t.data;
        s.nbytes = t.nbytes;
        const bool to_f32 = to == T_F32 && t.type != gguf::F32;
        const bool to_q6 = !to_f32 && (t.type == gguf::Q3_K || t.type == gguf::IQ3_S);
        const bool to_q8 = !to_f32 && t.type == gguf::IQ4_NL && getenv("MK_IQ4NL_Q8"); // native IQ4_NL unless MK_IQ4NL_Q8
        if (!to_f32 && !to_q6 && !to_q8) return s; // anything the kernels cannot read is refused by up_mat / need()
        const ggml_type from = (ggml_type) t.type;
        const ggml_type dst = to_f32 ? GGML_TYPE_F32 : to_q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
        const int64_t K = (int64_t) t.ne[0];
        int64_t rows = 1;
        for (size_t d = 1; d < t.ne.size(); d++) rows *= (int64_t) t.ne[d];
        const size_t rs_from = ggml_row_size(from, K), rs_to = ggml_row_size(dst, K);
        const auto t0 = std::chrono::steady_clock::now();
        s.own.resize(rs_to * (size_t) rows);
        const auto * tr = ggml_get_type_traits(from);
        const auto * tr_dst = ggml_get_type_traits(dst);
        std::atomic<bool> exact{true};
        parallel_rows(rows, [&](int64_t r) {
            thread_local std::vector<float> f, g;
            f.resize((size_t) K);
            const uint8_t * src = t.data + (size_t) r * rs_from;
            uint8_t * out = s.own.data() + (size_t) r * rs_to;
            tr->to_float(src, f.data(), K);
            if (to_f32) {
                memcpy(out, f.data(), (size_t) K * 4);
                return;
            }
            if (to_q6) reencode_q6k(t.type, src, f.data(), out, K);
            else reencode_iq4nl_q80(src, out, K);
            g.resize((size_t) K);
            tr_dst->to_float(out, g.data(), K);
            if (memcmp(f.data(), g.data(), (size_t) K * 4) != 0) exact = false;
        });
        if (!exact) throw std::runtime_error(name + ": re-encoding is not exact");
        s.type = (uint32_t) dst;
        s.data = s.own.data();
        s.nbytes = s.own.size();
        n_conv++;
        conv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        return s;
    }

    const float * up_f32(const std::string & name, size_t expect) {
        const auto & t = file.get(name);
        if (t.type != gguf::F32) throw std::runtime_error(name + " not f32");
        size_t n = t.nbytes / 4;
        if (expect && n != expect) throw std::runtime_error(name + " bad size");
        float * d = dalloc<float>(n, false);
        CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
        wbytes += t.nbytes;
        return d;
    }

    Mat up_mat(const std::string & name, int K, int N, int to = -1) {
        const Src t = want(name, to < 0 ? (int) file.get(name).type : to);
        if ((int) t.ne[0] != K || (int) t.ne[1] != N) throw std::runtime_error(name + " bad shape");
        Mat m{};
        m.K = K;
        m.N = N;
        m.type = (int) t.type;
        const int nb = K / 256;
        if (t.type == gguf::Q4_K || t.type == gguf::Q5_K) {
            m.rb = (int) gguf::type_row_bytes(t.type, K);
            uint8_t * d = dalloc<uint8_t>(t.nbytes, false);
            CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
            m.w = d;
            wbytes += t.nbytes;
        } else if (t.type == gguf::Q6_K) {
            // two streams (rows_q6k): 192 B of values per block, and per row a meta stream of 18 B per block in
            // pairs (sc even | sc odd | d even | d odd)
            if (nb & 1) throw std::runtime_error(name + ": Q6_K rows need an even number of 256-blocks");
            m.rb = nb * Q6_BLK;
            m.nd = nb * (Q6_META / 2); // halves
            std::vector<uint8_t> q((size_t) N * m.rb, 0), meta((size_t) N * nb * Q6_META, 0);
            const uint8_t * src = t.data;
            parallel_rows(N, [&](int64_t r) {
                for (int b = 0; b < nb; b++) {
                    const uint8_t * blk = src + ((size_t) r * nb + b) * 210;
                    const uint8_t * ql = blk;
                    const uint8_t * qh = blk + 128;
                    const int8_t * sc = (const int8_t *) (blk + 192);
                    uint16_t dbits;
                    memcpy(&dbits, blk + 208, 2);
                    uint8_t vals[256];
                    for (int nn = 0; nn < 2; nn++)
                        for (int l = 0; l < 32; l++) {
                            const uint8_t * L = ql + 64 * nn;
                            const uint8_t * H = qh + 32 * nn;
                            vals[128 * nn + l + 0] = (L[l] & 0xF) | (((H[l] >> 0) & 3) << 4);
                            vals[128 * nn + l + 32] = (L[l + 32] & 0xF) | (((H[l] >> 2) & 3) << 4);
                            vals[128 * nn + l + 64] = (L[l] >> 4) | (((H[l] >> 4) & 3) << 4);
                            vals[128 * nn + l + 96] = (L[l + 32] >> 4) | (((H[l] >> 6) & 3) << 4);
                        }
                    uint8_t * dst = q.data() + (size_t) r * m.rb + b * Q6_BLK;
                    for (int p = 0; p < 4; p++)
                        for (int i = 0; i < 32; i++)
                            dst[32 * p + i] = (vals[64 * p + i] & 0xF) | ((vals[64 * p + 32 + i] & 0xF) << 4);
                    for (int p = 0; p < 4; p++)
                        for (int hh = 0; hh < 2; hh++) {
                            uint32_t hA = 0, hB = 0;
                            for (int i = 0; i < 16; i++) {
                                const int sh = 8 * (i % 4) + 2 * (i / 4);
                                hA |= (uint32_t) (vals[64 * p + 16 * hh + i] >> 4) << sh;
                                hB |= (uint32_t) (vals[64 * p + 32 + 16 * hh + i] >> 4) << sh;
                            }
                            memcpy(dst + 128 + 8 * (2 * p + hh), &hA, 4);
                            memcpy(dst + 128 + 8 * (2 * p + hh) + 4, &hB, 4);
                        }
                    uint8_t * mp = meta.data() + ((size_t) r * nb + (b & ~1)) * Q6_META;
                    memcpy(mp + 16 * (b & 1), sc, 16);
                    memcpy(mp + 32 + 2 * (b & 1), &dbits, 2);
                }
            });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            uint8_t * dm = dalloc<uint8_t>(meta.size(), false);
            CK(cudaMemcpy(dm, meta.data(), meta.size(), cudaMemcpyHostToDevice));
            m.w = dq;
            m.d = reinterpret_cast<const __half *>(dm);
            wbytes += q.size() + meta.size();
        } else if (t.type == gguf::IQ4_XS) {
            m.type = T_IQ4X;
            if (nb & 1) throw std::runtime_error(name + ": IQ4_XS rows need an even number of 256-blocks");
            m.rb = nb * 136;
            std::vector<uint8_t> q((size_t) N * m.rb);
            parallel_rows(N, [&](int64_t r) { repack_iq4xs(t.data + (size_t) r * nb * 136, q.data() + (size_t) r * m.rb, nb); });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            m.w = dq;
            wbytes += q.size();
        } else if (t.type == gguf::IQ4_NL) {
            m.type = T_IQ4NL;
            m.rb = nb * 144;
            std::vector<uint8_t> q((size_t) N * m.rb);
            parallel_rows(N, [&](int64_t r) { repack_iq4nl(t.data + (size_t) r * nb * 144, q.data() + (size_t) r * m.rb, nb); });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            m.w = dq;
            wbytes += q.size();
        } else if (t.type == gguf::Q8_0) {
            const int nb32 = K / 32;
            m.rb = K;
            m.nd = nb32;
            std::vector<uint8_t> q((size_t) N * K);
            std::vector<uint16_t> ds((size_t) N * nb32);
            const uint8_t * src = t.data;
            parallel_rows(N, [&](int64_t r) {
                const uint8_t * row = src + (size_t) r * nb32 * 34;
                std::vector<int8_t> vals(K);
                for (int j = 0; j < nb32; j++) {
                    memcpy(&ds[(size_t) r * nb32 + j], row + 34 * j, 2);
                    memcpy(&vals[32 * j], row + 34 * j + 2, 32);
                }
                uint8_t * dst = q.data() + (size_t) r * K;
                for (int u = 0; u < K / 32; u++) {
                    const int b = u >> 3, p = (u >> 1) & 3, hh = u & 1;
                    memcpy(dst + 32 * u, &vals[256 * b + 64 * p + 16 * hh], 16);
                    memcpy(dst + 32 * u + 16, &vals[256 * b + 64 * p + 32 + 16 * hh], 16);
                }
            });
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            uint16_t * dd = dalloc<uint16_t>(ds.size(), false);
            CK(cudaMemcpy(dd, ds.data(), ds.size() * 2, cudaMemcpyHostToDevice));
            m.w = dq;
            m.d = (const __half *) dd;
            wbytes += q.size() + ds.size() * 2;
        } else {
            throw std::runtime_error(name + ": unsupported quant type " + std::to_string(t.type));
        }
        return m;
    }

    QBuf qbuf(int U) {
        QBuf q;
        q.U = U;
        q.xa = dalloc<int8_t>((size_t) MAXT * U * 16);
        q.xb = dalloc<int8_t>((size_t) MAXT * U * 16);
        q.meta = dalloc<uint2>((size_t) MAXT * U);
        q.ssq = dalloc<float>((size_t) MAXT * U);
        return q;
    }

    // MK_FAULT=cuda:<n> | sticky:<n> (tests of the error path only): the n-th prefill (1-based) fails half-way
    // through the batched layers, with a rejected CUDA call (non-sticky: an invalid memcpy direction; the state is
    // left half-updated) or a trapping kernel (sticky: the context is lost)
    int fault_kind = 0, fault_at = 0, fault_seen = 0;
    bool fault_now = false;
    void fault_inject() {
        fault_now = false;
        fprintf(stderr, "[mk] MK_FAULT: injecting a %s CUDA error\n", fault_kind == 1 ? "non-sticky" : "sticky");
        if (fault_kind == 1) CK(cudaMemcpy(hp.h, hp.h, 4, (cudaMemcpyKind) 99));
        else pk_fault<<<1, 1>>>();
    }

    int n_draft_max;
    int kv_mode; // KV cache format: 0 fp16, 1 q8_0, -1 auto (q8_0 when fp16 does not fit)
    Impl(const std::string & path, int ctx_, int ndraft_, const std::string & draft_vocab, int kv_mode_)
        : file(path), ctx(ctx_), ndraft(ndraft_), nd_avail(ndraft_), n_draft_max(ndraft_), kv_mode(kv_mode_) {
        if (file.strs["general.architecture"] != "qwen35") throw std::runtime_error("expected qwen35 architecture");
        if (file.ints["qwen35.block_count"] != NLA || file.ints["qwen35.embedding_length"] != D)
            throw std::runtime_error("unexpected model dimensions (this kernel is specialised for the 27B)");
        cudaDeviceProp prop;
        CK(cudaGetDeviceProperties(&prop, 0));
        nsm = prop.multiProcessorCount;
        CK(cudaEventCreateWithFlags(&wd_ev, cudaEventDisableTiming));
        if (const char * w = getenv("MK_WATCHDOG")) wd_limit = atof(w);
        if (const char * w = getenv("MK_WATCHDOG_SPIN")) wd_spin_s = atof(w);
        if (const char * f = getenv("MK_FAULT")) {
            const std::string s = f;
            const size_t c = s.find(':');
            fault_kind = s.compare(0, c, "cuda") == 0 ? 1 : s.compare(0, c, "sticky") == 0 ? 2 : 0;
            fault_at = c == std::string::npos ? 1 : atoi(s.c_str() + c + 1);
            if (!fault_kind || fault_at < 1) throw std::runtime_error("MK_FAULT: cuda:<n> or sticky:<n>");
        }
        if (memlog) {
            CK(cudaFree(0));
            mem_mark("context");
        }

        auto blk = [](int i, const char * s) { return "blk." + std::to_string(i) + "." + s; };
        // The kernels take the weight type as a template argument and dispatch on each matrix's
        // type at run time (gemv_pairs_any, resid_any, ffn_gu), for Q4_K, Q5_K, Q6_K, Q8_0 and IQ4_XS (plus what want()
        // re-encodes into those). Check every role so that an unsupported type is refused at load.
        auto tname = [](int t) -> std::string {
            switch (t) {
                case gguf::F32: return "F32";
                case gguf::F16: return "F16";
                case gguf::Q8_0: return "Q8_0";
                case gguf::Q3_K: return "Q3_K";
                case gguf::Q4_K: return "Q4_K";
                case gguf::Q5_K: return "Q5_K";
                case gguf::Q6_K: return "Q6_K";
                case gguf::IQ4_NL: return "IQ4_NL";
                case gguf::IQ3_S: return "IQ3_S";
                case gguf::IQ4_XS: return "IQ4_XS";
                default: return "type " + std::to_string(t);
            }
        };
        auto need = [&](const std::string & name, int got, std::initializer_list<int> ok) {
            for (int t : ok)
                if (got == t) return;
            std::string want;
            for (int t : ok) want += (want.empty() ? "" : " or ") + tname(t);
            throw std::runtime_error(name + " is " + tname(got) + ", the kernel needs " + want +
                                     " (see README: the GGUF)");
        };
        auto needq = [&](const std::string & name, int got) { need(name, got, {T_Q4K, T_Q5K, T_Q6K, T_Q8_0, T_IQ4X, T_IQ4NL}); };
        for (int i = 0; i < NLA; i++) {
            LayerW & L = hw.L[i];
            const bool mtp = i == NL;
            L.attn = mtp || ((i + 1) % 4 == 0);
            L.attn_norm = up_f32(blk(i, "attn_norm.weight"), D);
            L.post_norm = up_f32(blk(i, "post_attention_norm.weight"), D);
            if (L.attn) {
                L.q = up_mat(blk(i, "attn_q.weight"), D, QG);
                L.k = up_mat(blk(i, "attn_k.weight"), D, KVD);
                L.v = up_mat(blk(i, "attn_v.weight"), D, KVD);
                L.o = up_mat(blk(i, "attn_output.weight"), VD, D);
                needq(blk(i, "attn_q.weight"), L.q.type);
                needq(blk(i, "attn_k.weight"), L.k.type);
                needq(blk(i, "attn_v.weight"), L.v.type);
                needq(blk(i, "attn_output.weight"), L.o.type);
                L.qn = up_f32(blk(i, "attn_q_norm.weight"), HD);
                L.kn = up_f32(blk(i, "attn_k_norm.weight"), HD);
            } else {
                L.qkv = up_mat(blk(i, "attn_qkv.weight"), D, QKV);
                L.gate = up_mat(blk(i, "attn_gate.weight"), D, VD);
                L.sout = up_mat(blk(i, "ssm_out.weight"), VD, D);
                needq(blk(i, "attn_qkv.weight"), L.qkv.type);
                needq(blk(i, "attn_gate.weight"), L.gate.type);
                needq(blk(i, "ssm_out.weight"), L.sout.type);
                L.conv = up_f32(blk(i, "ssm_conv1d.weight"), 4 * QKV);
                L.dt = up_f32(blk(i, "ssm_dt.bias"), NVH);
                L.A = up_f32(blk(i, "ssm_a"), NVH);
                L.snorm = up_f32(blk(i, "ssm_norm.weight"), HK);
                const auto & raw_a = file.get(blk(i, "ssm_alpha.weight"));
                const auto & raw_b = file.get(blk(i, "ssm_beta.weight"));
                if (raw_a.type == gguf::Q8_0 && raw_b.type == gguf::Q8_0 && raw_a.ne[0] == D && raw_a.ne[1] == NVH &&
                    raw_b.ne[0] == D && raw_b.ne[1] == NVH && !getenv("MK_AB_F32")) {
                    // Q8_0 alpha / beta (UD) kept as int8 rows + fp16 block scales; the kernels form
                    // (q * d) * attn_norm on the fly, the same fp32 values as the folded rows below (MK_AB_F32=1)
                    std::vector<int8_t> q((size_t) 2 * NVH * D);
                    std::vector<uint16_t> ds((size_t) 2 * NVH * (D / 32));
                    for (int w = 0; w < 2; w++) {
                        const uint8_t * src = (w ? raw_b : raw_a).data;
                        for (size_t blk_ = 0; blk_ < (size_t) NVH * (D / 32); blk_++) {
                            memcpy(&ds[(size_t) w * NVH * (D / 32) + blk_], src + blk_ * 34, 2);
                            memcpy(&q[((size_t) w * NVH * (D / 32) + blk_) * 32], src + blk_ * 34 + 2, 32);
                        }
                    }
                    int8_t * dq = dalloc<int8_t>(q.size(), false);
                    CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
                    uint16_t * dd = dalloc<uint16_t>(ds.size(), false);
                    CK(cudaMemcpy(dd, ds.data(), ds.size() * 2, cudaMemcpyHostToDevice));
                    L.abq = dq;
                    L.abd = reinterpret_cast<const __half *>(dd);
                    L.ab = nullptr;
                    wbytes += q.size() + ds.size() * 2;
                } else {
                // alpha/beta projections in fp32 with the attn_norm weight folded in
                const Src ta = want(blk(i, "ssm_alpha.weight"), T_F32);
                const Src tb = want(blk(i, "ssm_beta.weight"), T_F32);
                const auto & tn = file.get(blk(i, "attn_norm.weight"));
                // read below as raw floats, so anything but F32 would be reinterpreted silently
                need(blk(i, "ssm_alpha.weight"), (int) ta.type, {T_F32});
                need(blk(i, "ssm_beta.weight"), (int) tb.type, {T_F32});
                if (ta.ne[0] != D || ta.ne[1] != NVH || tb.ne[0] != D || tb.ne[1] != NVH)
                    throw std::runtime_error(blk(i, "ssm_alpha/beta.weight") + " bad shape");
                std::vector<float> ab((size_t) 2 * NVH * D);
                const float * fa = (const float *) ta.data;
                const float * fb = (const float *) tb.data;
                const float * fn = (const float *) tn.data;
                for (int h = 0; h < NVH; h++)
                    for (int e = 0; e < D; e++) {
                        ab[(size_t) h * D + e] = fa[(size_t) h * D + e] * fn[e];
                        ab[(size_t) (NVH + h) * D + e] = fb[(size_t) h * D + e] * fn[e];
                    }
                float * d = dalloc<float>(ab.size(), false);
                CK(cudaMemcpy(d, ab.data(), ab.size() * 4, cudaMemcpyHostToDevice));
                L.ab = d;
                L.abq = nullptr;
                L.abd = nullptr;
                wbytes += ab.size() * 4;
                }
            }
            L.fg = up_mat(blk(i, "ffn_gate.weight"), D, FF);
            L.fu = up_mat(blk(i, "ffn_up.weight"), D, FF);
            L.fd = up_mat(blk(i, "ffn_down.weight"), FF, D);
            needq(blk(i, "ffn_gate.weight"), L.fg.type);
            needq(blk(i, "ffn_up.weight"), L.fu.type);
            needq(blk(i, "ffn_down.weight"), L.fd.type);
        }
        mem_mark("weights: 64 layers + MTP");
        hw.eh = up_mat(blk(NL, "nextn.eh_proj.weight"), 2 * D, D);
        needq(blk(NL, "nextn.eh_proj.weight"), hw.eh.type);
        hw.enorm = up_f32(blk(NL, "nextn.enorm.weight"), D);
        hw.hnorm = up_f32(blk(NL, "nextn.hnorm.weight"), D);
        hw.shnorm = up_f32(blk(NL, "nextn.shared_head_norm.weight"), D);
        hw.onorm = up_f32("output_norm.weight", D);
        hw.head = up_mat("output.weight", D, VOCAB);
        if (hw.head.type != T_Q6K) throw std::runtime_error("output.weight must be Q6_K");
        mem_mark("weights: eh, norms, lm_head");
        hw.head_d = hw.head;
        hw.draft_ids = nullptr;
        if (!draft_vocab.empty()) {
            FILE * f = fopen(draft_vocab.c_str(), "rb");
            if (!f) throw std::runtime_error("cannot open draft vocab " + draft_vocab);
            std::vector<int> ids;
            int id;
            while (fread(&id, 4, 1, f) == 1) {
                if (id < 0 || id >= VOCAB) throw std::runtime_error("bad id in draft vocab");
                ids.push_back(id);
            }
            fclose(f);
            if (ids.size() & 1) ids.push_back(ids.back());
            // room for DV_EXTRA more rows: tokens of the current sequence that are not in the list
            // are added on the fly (identifiers of pasted code, see dv_add)
            // the draft head reads lm_head rows through the id list (lm_head), so no copy of the rows is kept
            const int cap = (int) ids.size() + DV_EXTRA;
            int * did = dalloc<int>(cap, false);
            CK(cudaMemcpy(did, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice));
            Mat m = hw.head;
            m.N = (int) ids.size();
            hw.head_d = m;
            hw.draft_ids = did;
            dv_ids = ids;
            dv_base = (int) ids.size();
            dv_in.assign(VOCAB, 0);
            for (int t : ids) dv_in[t] = 1;
            fprintf(stderr, "[mk] MTP draft head restricted to %d tokens (%s)\n", m.N, draft_vocab.c_str());
        }
        mem_mark("draft head copy");
        {
            const auto & t = file.get("token_embd.weight");
            if (t.type != gguf::Q4_K) throw std::runtime_error("token_embd must be Q4_K");
            // mapped pinned host memory: a cycle reads only a few 2.8 KB rows over PCIe, saves 0.7 GB of VRAM
            const char * ed = getenv("MK_EMBD_DEVICE");
            if (!(ed && ed[0] == '1') &&
                cudaHostAlloc(&embd_host, t.nbytes, cudaHostAllocMapped | cudaHostAllocWriteCombined) == cudaSuccess) {
                memcpy(embd_host, t.data, t.nbytes);
                void * d = nullptr;
                CK(cudaHostGetDevicePointer(&d, embd_host, 0));
                hw.embd = (const uint8_t *) d;
                fprintf(stderr, "[mk] token_embd in mapped host memory (%.2f GB)\n", t.nbytes / 1e9);
            } else {
                cudaGetLastError();
                embd_host = nullptr;
                uint8_t * d = dalloc<uint8_t>(t.nbytes, false);
                CK(cudaMemcpy(d, t.data, t.nbytes, cudaMemcpyHostToDevice));
                hw.embd = d;
            }
        }
        mem_mark("token_embd");
        CK(cudaMemcpyToSymbol(cW, &hw, sizeof(ModelW)));
        mem_mark("cW symbol (module load)");
        if (n_conv)
            fprintf(stderr, "[mk] %d tensors re-encoded in %.1f s (exact: Q3_K / IQ3_S -> Q6_K, IQ4_NL -> Q8_0 with MK_IQ4NL_Q8; ssm_alpha / beta -> F32)\n",
                    n_conv, conv_s);

        hb.x = dalloc<float>(MAXT * D);
        hb.xm = dalloc<float>(MAXT * D);
        hb.qn = qbuf(U_D);
        hb.qo = qbuf(U_VD);
        hb.qa = qbuf(U_FF);
        hb.qe = qbuf(U_EH);
        hb.z = dalloc<float>(MAXT * VD);
        hb.ab = dalloc<float>(AB_SPLIT * MAXT * 2 * NVH);
        hb.qbuf = dalloc<float>(MAXT * QG);
        hb.kbuf = dalloc<float>(MAXT * KVD);
        hb.vbuf = dalloc<float>(MAXT * KVD);
        hb.conv = dalloc<float>((size_t) 48 * CONV_RING * QKV);
        hb.kvgb = dalloc<float>((size_t) 48 * KV_RING * KVGB);
        mem_mark("decode act bufs, conv, kvgb");
        hb.dstate = dalloc<float>((size_t) 48 * NVH * 128 * 128);
        mem_mark("GDN recurrent state");
        {
            const size_t rows = (size_t) N_ATT_SLOTS * NKV * ctx;
            size_t free_b = 0, total_b = 0;
            CK(cudaMemGetInfo(&free_b, &total_b));
            // fp16 needs 2 x rows x HD x 2 bytes; leave ~1.5 GB for the prefill buffers, checkpoint and runtime.
            // A non-empty MK_KV picks the quantized cache in auto mode too.
            const char * kv_spec = getenv("MK_KV");
            const bool want_kv = kv_spec && kv_spec[0];
            const bool q8 = kv_mode == 1 || (kv_mode < 0 && (want_kv || rows * HD * 4 + (size_t) 1536 * 1024 * 1024 > free_b));
            hb.kvq8 = q8;
            // MK_KV (q8 cache): per attention slot and K / V, q8_0 (default) or q4 (kv_store_q4), e.g. "V*=4,K16=4r":
            // comma-separated <K|V|*><slot>=<8|4>[r], slot 0..15 = attention layer 4 slot + 3, 16 = MTP block, * or
            // empty = all slots; r (K only): K and the queries Hadamard-rotated
            const size_t srows = (size_t) NKV * ctx; // rows of one slot
            for (int s = 0; s < N_ATT_SLOTS; s++) {
                hb.kvb[0][s] = hb.kvb[1][s] = 8;
                hb.kvr[s] = 0;
            }
            if (want_kv) {
                if (!q8) throw std::runtime_error("MK_KV needs the q8 KV cache (--kv q8 or auto, not f16)");
                const std::string str = kv_spec;
                size_t i = 0;
                while (i < str.size()) {
                    size_t j = str.find(',', i);
                    if (j == std::string::npos) j = str.size();
                    const std::string tok = str.substr(i, j - i);
                    i = j + 1;
                    if (tok.empty()) continue;
                    const size_t e = tok.find('=');
                    if (e == std::string::npos || e == 0 || !strchr("KV*", tok[0])) throw std::runtime_error("MK_KV: bad item " + tok);
                    const std::string sl = tok.substr(1, e - 1), md = tok.substr(e + 1);
                    if (md != "8" && md != "4" && md != "8r" && md != "4r") throw std::runtime_error("MK_KV: expected 8, 4, 8r or 4r in " + tok);
                    const bool rot = md.size() == 2;
                    if (rot && tok[0] == 'V') throw std::runtime_error("MK_KV: rotation (r) is for K only: " + tok);
                    const bool all = sl.empty() || sl == "*";
                    const int slot = all ? -1 : std::stoi(sl);
                    if (!all && (slot < 0 || slot >= N_ATT_SLOTS)) throw std::runtime_error("MK_KV: bad slot in " + tok);
                    for (int w = 0; w < 2; w++)
                        if (tok[0] == '*' || tok[0] == "KV"[w])
                            for (int s = 0; s < N_ATT_SLOTS; s++)
                                if (all || s == slot) {
                                    hb.kvb[w][s] = md[0] - '0';
                                    if (w == 0) hb.kvr[s] = rot;
                                }
                }
            }
            size_t vbytes[2] = {0, 0};
            int n4 = 0;
            for (int w = 0; w < 2; w++)
                for (int s = 0; s < N_ATT_SLOTS; s++) {
                    vbytes[w] += srows * HD * hb.kvb[w][s] / 8;
                    n4 += hb.kvb[w][s] == 4 || (w == 0 && hb.kvr[s]);
                }
            hb.kc = q8 ? (void *) dalloc<int8_t>(vbytes[0]) : (void *) dalloc<__half>(rows * HD);
            hb.vc = q8 ? (void *) dalloc<int8_t>(vbytes[1]) : (void *) dalloc<__half>(rows * HD);
            hb.ksc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
            hb.vsc = q8 ? dalloc<__half>(rows * (HD / 32)) : nullptr;
            size_t off[2] = {0, 0};
            for (int s = 0; s < N_ATT_SLOTS; s++)
                for (int w = 0; w < 2; w++) { // slots packed in order: all-q8 gives the same addresses as [slot][head][ctx][HD]
                    hb.kvp[w][s] = static_cast<int8_t *>(w ? hb.vc : hb.kc) + off[w];
                    hb.kvs[w][s] = q8 ? (w ? hb.vsc : hb.ksc) + s * srows * (HD / 32) : nullptr;
                    off[w] += srows * HD * hb.kvb[w][s] / 8;
                }
            const double gb = q8 ? (vbytes[0] + vbytes[1] + 2.0 * rows * (HD / 32) * 2) / 1e9 : rows * HD * 4.0 / 1e9;
            fprintf(stderr, "[mk] KV cache %s, %.2f GB for %d positions\n", q8 ? (n4 ? "q8_0 + q4 (MK_KV)" : "q8_0") : "fp16", gb, ctx);
            if (n4) {
                std::string k, v;
                for (int s = 0; s < N_ATT_SLOTS; s++) {
                    k += hb.kvb[0][s] == 4 ? (hb.kvr[s] ? 'R' : '4') : (hb.kvr[s] ? 'r' : '8');
                    v += hb.kvb[1][s] == 4 ? '4' : '8';
                }
                fprintf(stderr, "[mk] MK_KV bits per slot (layers 3, 7, .., 63, MTP; R = 4 rotated, r = 8 rotated): K %s  V %s\n", k.c_str(), v.c_str());
            }
        }
        mem_mark("KV cache");
        hb.apart = dalloc<float>((size_t) NKV * MAX_CH * AR * 256);
        hb.aml = dalloc<float2>((size_t) NKV * MAX_CH * AR);
        hb.h_hist = dalloc<float>((size_t) H_RING * D);
        mem_mark("attn partials, h ring");
        {
            std::vector<float2> rt((size_t) ctx * 32);
            for (int p = 0; p < ctx; p++)
                for (int i = 0; i < 32; i++) {
                    const double th = (double) p * std::pow((double) ROPE_BASE, -2.0 * i / 64.0);
                    rt[(size_t) p * 32 + i] = make_float2((float) std::cos(th), (float) std::sin(th));
                }
            float2 * d = dalloc<float2>(rt.size(), false);
            CK(cudaMemcpy(d, rt.data(), rt.size() * sizeof(float2), cudaMemcpyHostToDevice));
            hb.rope = d;
        }
        mem_mark("RoPE table");
        hb.hm = dalloc<float>(D);
        hb.blk_best = dalloc<float2>((size_t) nsm * MAXT * TOPK_MAX);
        hb.blk_lse = dalloc<float2>((size_t) nsm);
        hb.dlog = dalloc<float2>((size_t) MAXT * (TOPK_MAX + 1));
        hb.dq = dalloc<float2>((size_t) MAXT * TOPK_MAX);
        hb.cand = dalloc<float2>((size_t) MAXT * TOPK_MAX);
        hb.logits = nullptr; // 5 MB used by scoring launches only (score_logits): logits_alloc()
        hb.bar = dalloc<unsigned>(4);
        hb.cnt = dalloc<unsigned>(4);
        hb.ctl = dalloc<Ctl>(1);
        hb.ctx = ctx;
        profiling = getenv("MK_PROFILE") != nullptr;
        hb.prof = profiling ? dalloc<unsigned long long>(4096 + 1024 * 128) : nullptr;
        mem_mark("logits, sampling, ctl");
        if (const char * spec = getenv("MK_KV_EMU_SPEC")) {
#if MK_KV_EMU
            // comma-separated <which><slot>=<bits>[r][a]: which K, V or * (both); slot 0..15 (attention layer 4 slot + 3),
            // 16 (MTP) or * / empty (all); r: Hadamard rotation, a: q4_0-style scale (kv_store_emu)
            const std::string s = spec;
            size_t i = 0;
            while (i < s.size()) {
                size_t j = s.find(',', i);
                if (j == std::string::npos) j = s.size();
                const std::string tok = s.substr(i, j - i);
                i = j + 1;
                if (tok.empty()) continue;
                const size_t e = tok.find('=');
                if (e == std::string::npos || e == 0 || !strchr("KV*", tok[0])) throw std::runtime_error("MK_KV_EMU_SPEC: bad item " + tok);
                const std::string sl = tok.substr(1, e - 1), md = tok.substr(e + 1);
                int bits = 0, mode = 0;
                size_t k = 0;
                while (k < md.size() && isdigit((unsigned char) md[k])) bits = 10 * bits + (md[k++] - '0');
                for (; k < md.size(); k++) {
                    if (md[k] == 'r') mode |= 16;
                    else if (md[k] == 'a') mode |= 32;
                    else throw std::runtime_error("MK_KV_EMU_SPEC: bad mode " + md);
                }
                if (bits < 2 || bits > 8) throw std::runtime_error("MK_KV_EMU_SPEC: bits must be 2..8 in " + tok);
                mode |= bits;
                const bool all = sl.empty() || sl == "*";
                const int slot = all ? -1 : std::stoi(sl);
                if (!all && (slot < 0 || slot >= N_ATT_SLOTS)) throw std::runtime_error("MK_KV_EMU_SPEC: bad slot in " + tok);
                for (int w = 0; w < 2; w++)
                    if (tok[0] == '*' || tok[0] == "KV"[w])
                        for (int q = 0; q < N_ATT_SLOTS; q++)
                            if (all || q == slot) hb.kvemu[w][q] = mode == 8 ? 0 : mode;
            }
            fprintf(stderr, "[mk] MK_KV_EMU_SPEC %s: K", spec);
            for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[0][q]);
            fprintf(stderr, " | V");
            for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[1][q]);
            fprintf(stderr, "\n");
            if (!hb.kvq8) throw std::runtime_error("MK_KV_EMU_SPEC needs the q8 KV cache (--kv q8)");
            for (int w = 0; w < 2; w++)
                for (int s = 0; s < N_ATT_SLOTS; s++)
                    if (hb.kvb[w][s] != 8 || hb.kvr[s]) throw std::runtime_error("MK_KV_EMU_SPEC needs an all-q8 cache (no MK_KV)");
#else
            (void) spec;
            throw std::runtime_error("MK_KV_EMU_SPEC needs a build with -DMK_KV_EMU=1");
#endif
        }
        CK(cudaMemcpyToSymbol(cB, &hb, sizeof(Bufs)));

        CK(cudaFuncSetAttribute(mk_cycle, cudaFuncAttributeMaxDynamicSharedMemorySize, SM_TOTAL));
        int nb = 0;
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, mk_cycle, NT, SM_TOTAL));
        if (nb < 1) throw std::runtime_error("megakernel does not fit on an SM");
        // The driver reserves local memory for every thread slot of the device (1,536 per SM x 82 SMs), sized by
        // the larger of the stack-size limit (default 1 KB per thread = 123 MiB) and the launched kernels' own
        // (static) frames. No kernel here needs a dynamic stack, so the limit only adds a floor: drop it.
        {
            const char * sl = getenv("MK_STACK_LIMIT");
            CK(cudaDeviceSetLimit(cudaLimitStackSize, sl ? (size_t) atol(sl) : 0));
        }
        if (memlog) {
            mem_mark("mk_cycle attributes");
            cudaFuncAttributes fa{};
            CK(cudaFuncGetAttributes(&fa, mk_cycle));
            size_t stk = 0, heap = 0, fifo = 0;
            CK(cudaDeviceGetLimit(&stk, cudaLimitStackSize));
            CK(cudaDeviceGetLimit(&heap, cudaLimitMallocHeapSize));
            CK(cudaDeviceGetLimit(&fifo, cudaLimitPrintfFifoSize));
            fprintf(stderr, "[mk-mem] mk_cycle: %d regs, local %zu B/thread, const %zu B, static smem %zu B; limits: stack %zu B/thread, "
                            "malloc heap %zu B, printf fifo %zu B; %d SMs x %d threads/SM\n",
                    fa.numRegs, (size_t) fa.localSizeBytes, (size_t) fa.constSizeBytes, (size_t) fa.sharedSizeBytes, stk, heap, fifo, nsm,
                    prop.maxThreadsPerMultiProcessor);
        }
        // the DeltaNet checkpoint and the batched-prefill scratch are allocated here, not inside the first prompt's time
        ck_alloc();
        pf_init();
        ck_make_plan();
        fprintf(stderr, "[mk] weights %.2f GB on device, %d SMs, ctx %d, n_draft %d\n", wbytes / 1e9, nsm, ctx, ndraft);
    }

#if MK_SPEC_LOG
    // one JSON line per decode cycle: start position n, T, accepted a, the verified drafts d, the verify
    // rows' targets tg, per draft its MTP [lse, best logit, (logit, id) x TOPK_MAX], and when sampling the
    // main rows' candidates p [(logit, id) x TOPK_MAX]
    void slog_cycle(int n0, int T, const int * drafts, const Ctl & c) {
        if (!slog) return;
        const bool smp = smp_temp > 0.f;
        std::vector<float2> cand;
        if (smp) {
            cand.resize((size_t) T * TOPK_MAX);
            CK(cudaMemcpy(cand.data(), hb.cand, cand.size() * sizeof(float2), cudaMemcpyDeviceToHost));
        }
        auto id_of = [](float y) { int v; memcpy(&v, &y, 4); return v; };
        fprintf(slog, "{\"n\":%d,\"T\":%d,\"a\":%d,\"nd\":%d,\"d\":[", n0, T, c.n_acc, nd_avail);
        for (int i = 1; i < T; i++) fprintf(slog, "%s%d", i > 1 ? "," : "", drafts[i]);
        fprintf(slog, "],\"tg\":[");
        for (int i = 0; i < T; i++) fprintf(slog, "%s%d", i ? "," : "", c.tgt[i]);
        fprintf(slog, "],\"q\":[");
        for (int i = 0; i + 1 < T; i++) {
            const float2 * q = slog_q.data() + (size_t) i * (TOPK_MAX + 1);
            fprintf(slog, "%s[%.7g,%.7g", i ? "," : "", q[TOPK_MAX].x, q[TOPK_MAX].y);
            for (int j = 0; j < TOPK_MAX; j++) fprintf(slog, ",%.7g,%d", q[j].x, id_of(q[j].y));
            fprintf(slog, "]");
        }
        fprintf(slog, "]");
        if (smp) {
            fprintf(slog, ",\"p\":[");
            for (int i = 0; i < T; i++) {
                fprintf(slog, "%s[", i ? "," : "");
                for (int j = 0; j < TOPK_MAX; j++)
                    fprintf(slog, "%s%.7g,%d", j ? "," : "", cand[(size_t) i * TOPK_MAX + j].x, id_of(cand[(size_t) i * TOPK_MAX + j].y));
                fprintf(slog, "]");
            }
            fprintf(slog, "]");
        }
        fprintf(slog, "}\n");
        slog_drafts(); // the drafts this launch produced, verified by the next cycle
    }
#endif

    ~Impl() {
#if MK_SPEC_LOG
        if (slog) fclose(slog);
#endif
        try { // (CK throws; a destructor must not)
            print_profile();
            mem_mark("at exit");
            ck_wait_saved(); // a checkpoint copy may still read the scratch / write the host buffer
            ck_wait_restore();
        } catch (const std::exception & e) {
            fprintf(stderr, "[mk] teardown: %s\n", e.what());
        }
        for (void * p : allocs) use_pool ? cudaFreeAsync(p, 0) : cudaFree(p);
        cudaStreamSynchronize(0);
        if (ck_st) {
            cudaStreamSynchronize(ck_st);
            cudaStreamDestroy(ck_st);
            for (cudaEvent_t e : {ck_ev_src, ck_ev_saved}) cudaEventDestroy(e);
            for (cudaEvent_t e : ck_ev_grp) cudaEventDestroy(e);
        }
        if (embd_host) cudaFreeHost(embd_host);
        if (ck_host) cudaFreeHost(ck_host);
    }

    void launch(int mode, int T, int last, int score = 0) {
        Launch L{mode, T, last, ndraft, score, smp_temp, smp_top_p, smp_min_p, smp_top_k, smp_seed};
        for (int k = 0; k < 4; k++) {
            L.gate_drop[k] = pol.gate_drop[k];
            L.gate_stop[k] = pol.gate_stop[k];
        }
        L.gate_chain = pol.gate_chain;
        L.spec_smp = smp_temp > 0.f && pol.h1;
        void * args[] = {&L};
        ck_wait_restore(); // (cheap no-ops unless a checkpoint copy is pending on ck_st)
        ck_guard_state();
        CK(cudaMemsetAsync(hb.bar, 0, 16));
        if (profiling) CK(cudaMemsetAsync(hb.prof, 0, (4096 + 1024 * 128) * 8));
        CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
        wd_T = T;
        wd_mode = mode;
        wait_device(mode == MODE_DECODE ? "a decode launch" : "a prefill launch", wd_limit);
        launches_seen++;
        if (memlog && launches_seen == 1) mem_mark("first mk_cycle launch");
        if (profiling && mode == MODE_DECODE) {
            std::vector<unsigned long long> pr(4096 + 1024 * 128);
            CK(cudaMemcpy(pr.data(), hb.prof, pr.size() * 8, cudaMemcpyDeviceToHost));
            unsigned long long prev = pr[0] >> 8;
            auto & bn = best_ns[T];
            auto & bt = best_tag[T];
            auto & bf = best_fill[T];
            for (int k = 1; k < 4096 && pr[k]; k++) {
                const unsigned long long t = pr[k] >> 8;
                const int tag = (int) (pr[k] & 0xff);
                const double dt = (double) (t - prev);
                ph_ns[tag] += dt;
                ph_n[tag]++;
                launch_ns += dt;
                prev = t;
                const unsigned long long mark = pr[2048 + k] & ((1ull << 56) - 1);
                const double fill = mark ? (double) ((mark - (t - (unsigned long long) dt)) & ((1ull << 56) - 1)) : 0.0;
                if ((int) bn.size() < k) { bn.resize(k, 1e30); bt.resize(k, -1); bf.resize(k, 0.0); }
                if (k < 1024) {
                    std::vector<double> arr;
                    const unsigned long long t0 = t - (unsigned long long) dt;
                    for (int b = 0; b < nsm; b++) {
                        const unsigned long long a = pr[4096 + k * 128 + b];
                        if (a) arr.push_back((double) (((a & ((1ull << 56) - 1)) - t0) & ((1ull << 56) - 1)));
                    }
                    if (arr.size() == (size_t) nsm) {
                        std::sort(arr.begin(), arr.end());
                        arr_first[tag] += arr.front();
                        arr_med[tag] += arr[arr.size() / 2];
                        arr_last[tag] += arr.back();
                        arr_n[tag]++;
                    }
                }
                if (bt[k - 1] != tag) { bt[k - 1] = tag; bn[k - 1] = dt; bf[k - 1] = fill; } // sequence changed
                else if (dt < bn[k - 1]) { bn[k - 1] = dt; bf[k - 1] = fill; }
            }
            launches++;
        }
    }

    void print_profile() {
        if (!profiling || !launches) return;
        static const char * names[PH_COUNT] = {"embed", "attn_proj", "attn_core", "attn_comb", "resid_o", "dn_proj",
                                               "dn_core", "resid_sout", "ffn_gu", "ffn_down", "lm_head", "mtp_m0",
                                               "mtp_eh", "mtp_attn_proj", "mtp_attn_core", "mtp_attn_comb",
                                               "mtp_resid_o", "mtp_ffn_gu", "mtp_ffn_down", "mtp_lm_head"};
        fprintf(stderr, "[mk-profile] %ld decode launches, mean %.3f ms in barriers-to-barrier phases\n", launches,
                launch_ns / launches / 1e6);
        for (int i = 0; i < PH_COUNT; i++) {
            if (!ph_n[i]) continue;
            fprintf(stderr, "  %-16s total/launch %8.3f ms   per call %8.2f us   calls/launch %6.1f\n", names[i],
                    ph_ns[i] / launches / 1e6, ph_ns[i] / ph_n[i] / 1e3, (double) ph_n[i] / launches);
        }
        fprintf(stderr, "[mk-profile] block arrival at barrier (us after phase start; mean over launches): first / median / last\n");
        for (int i = 0; i < PH_COUNT; i++)
            if (arr_n[i])
                fprintf(stderr, "  %-16s %8.2f %8.2f %8.2f\n", names[i], arr_first[i] / arr_n[i] / 1e3, arr_med[i] / arr_n[i] / 1e3,
                        arr_last[i] / arr_n[i] / 1e3);
        for (int T = 1; T <= MAXT; T++) {
            if (best_ns[T].empty()) continue;
            std::vector<double> tot(PH_COUNT, 0.0), fil(PH_COUNT, 0.0);
            std::vector<int> cnt(PH_COUNT, 0);
            double all = 0;
            for (size_t k = 0; k < best_ns[T].size(); k++) {
                if (best_tag[T][k] < 0) continue;
                tot[best_tag[T][k]] += best_ns[T][k];
                fil[best_tag[T][k]] += best_fill[T][k];
                cnt[best_tag[T][k]]++;
                all += best_ns[T][k];
            }
            fprintf(stderr, "[mk-profile] T=%d min-composed launch %.3f ms (interference-free estimate)\n", T, all / 1e6);
            for (int i = 0; i < PH_COUNT; i++)
                if (cnt[i])
                    fprintf(stderr, "  %-16s %8.3f ms  %8.2f us/call (fill %5.2f us)  x%d\n", names[i], tot[i] / 1e6,
                            tot[i] / cnt[i] / 1e3, fil[i] / cnt[i] / 1e3, cnt[i]);
        }
    }

    void bench() {
        const char * names[3] = {"ffn_gu", "ffn_down", "dn_proj"};
        if (profiling) {
            // i-cache test: 4 back-to-back iterations of one phase inside one launch
            for (int which = 0; which < 3; which++) {
                std::vector<double> best(5, 1e30);
                for (int rep = 0; rep < 60; rep++) {
                    Launch L{MODE_BENCH, 4, which, which == 2 ? 0 : 1, 0};
                    void * args[] = {&L};
                    CK(cudaMemsetAsync(hb.bar, 0, 16));
                    CK(cudaMemsetAsync(hb.prof, 0, 4096 * 8));
                    CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                    std::vector<unsigned long long> pr(8);
                    CK(cudaMemcpy(pr.data(), hb.prof, 8 * 8, cudaMemcpyDeviceToHost));
                    for (int k = 1; k <= 4; k++) best[k] = std::min(best[k], (double) ((pr[k] >> 8) - (pr[k - 1] >> 8)));
                }
                fprintf(stderr, "[mk-bench] %-8s T=4 back-to-back iterations: %.1f / %.1f / %.1f / %.1f us\n", names[which],
                        best[1] / 1e3, best[2] / 1e3, best[3] / 1e3, best[4] / 1e3);
            }
            return;
        }
        for (int which = 0; which < 3; which++)
            for (int T : {1, 4}) {
                float best = 1e30f;
                cudaEvent_t a, b;
                cudaEventCreate(&a);
                cudaEventCreate(&b);
                for (int rep = 0; rep < 400; rep++) {
                    cudaEventRecord(a);
                    Launch L{MODE_BENCH, T, which, which == 2 ? 0 : (rep % 16), 0};
                    void * args[] = {&L};
                    CK(cudaMemsetAsync(hb.bar, 0, 16));
                    CK(cudaLaunchCooperativeKernel((void *) mk_cycle, dim3(nsm), dim3(NT), args, SM_TOTAL, 0));
                    cudaEventRecord(b);
                    CK(cudaEventSynchronize(b));
                    float ms;
                    cudaEventElapsedTime(&ms, a, b);
                    best = std::min(best, ms);
                }
                fprintf(stderr, "[mk-bench] %-8s T=%d: %.1f us/launch (1 layer + barrier + launch overhead)\n", names[which], T, best * 1000);
            }
    }

    // ---------------- batched prefill (prefill.cuh) ----------------
    PBufs hp{};
    bool pf_ready = false;

    void pf_init() {
        if (pf_ready) return;
        // MK_PF_CAP: tokens per prefill chunk (multiple of 128); scratch is ~0.33 MB per token
        int cap = 2048;
        if (const char * e = getenv("MK_PF_CAP")) cap = std::max(128, std::min(2048, atoi(e) / 128 * 128));
        hp.cap = cap;
        mem_mark("before prefill scratch");
        hp.x = dalloc<float>((size_t) cap * D);
        hp.xm = dalloc<float>((size_t) cap * D);
        hp.rs = dalloc<float>(cap);
        hp.xh = dalloc<__half>((size_t) cap * 2 * D); // GEMM inputs: K = D, VD (attention / delta-net out) or 2 D (MTP eh)
        hp.xh2 = dalloc<__half>((size_t) cap * FF);
        hp.y0 = dalloc<float>((size_t) cap * FF);
        hp.y1 = dalloc<float>((size_t) cap * FF);
        hp.y2 = dalloc<float>((size_t) cap * VD);
        hp.dq = dalloc<__half>((size_t) cap * QKV);
        hp.gb = dalloc<float>((size_t) cap * 2 * NVH);
        hp.ab = dalloc<float>((size_t) 8 * cap * 2 * NVH);
        hp.h = dalloc<float>((size_t) (cap + 1) * D);
        hp.qh = dalloc<__half>((size_t) cap * NH * HD);
        hp.tok = dalloc<int>(cap);
        hp.amax = dalloc<int>(cap);
        mem_mark("prefill scratch");
        CK(cudaMemcpyToSymbol(cP, &hp, sizeof(PBufs)));
        mmh_attr<pf::TQ4K, false>();
        mmh_attr<pf::TQ5K, false>();
        mmh_attr<pf::TQ6K, false>();
        mmh_attr<pf::TQ80, false>();
        mmh_attr<pf::TIQ4, false>();
        mmh_attr<pf::TIQ4NL, false>();
        mmh_attr<pf::TQ4K, true>();
        mmh_attr<pf::TQ5K, true>();
        mmh_attr<pf::TIQ4, true>();
        if (const char * e = getenv("MK_PF_BN")) { // force one GEMM token tile (tuning / tests); 0 = by prompt size
            pf_bn_force = atoi(e);
            if (pf_bn_force != 0 && pf_bn_force != 32 && pf_bn_force != 64 && pf_bn_force != 128)
                throw std::runtime_error("MK_PF_BN: 0, 32, 64 or 128");
        }
        CK(cudaFuncSetAttribute(pk_attn<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
        CK(cudaFuncSetAttribute(pk_attn<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
        CK(cudaFuncSetAttribute(pk_attn2<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
        CK(cudaFuncSetAttribute(pk_attn2<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
        CK(cudaFuncSetAttribute(pk_attn2<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
        CK(cudaFuncSetAttribute(pk_attn2<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
        CK(cudaFuncSetAttribute(pk_attn8<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
        CK(cudaFuncSetAttribute(pk_attn8<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
        int dev = 0, nsm = 0, occ = 0;
        CK(cudaGetDevice(&dev));
        CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, dev));
        CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, pk_attn2<true, false>, A2_NT, A2_SMEM));
        a2_slots = nsm * occ;
        // MK_ATTN=1: the original pk_attn; MK_ATTN_SPLIT=0: no tail split; MK_ATTN_PV16=0: fp32-accumulated P V
        // (pk_attn2 with both off is bit-identical to pk_attn)
        if (const char * e = getenv("MK_ATTN")) attn_old = atoi(e) == 1;
        if (const char * e = getenv("MK_ATTN_SPLIT")) attn_split = atoi(e) != 0;
        if (const char * e = getenv("MK_ATTN_PV16")) attn_pv16 = atoi(e) != 0;
        for (int w = 0; w < 2; w++)
            for (int s = 0; s < N_ATT_SLOTS; s++)
                if (attn_old && hb.kvb[w][s] != 8) throw std::runtime_error("MK_ATTN=1 (the original prefill attention) reads q8_0 only: no MK_KV");
        // MK_ATTN_S8 (q8 cache, with PV16): 1 = int8 Q K^T, Q scales per 32-dim block (pk_attn8); 2 = per row; 0 = off
        if (const char * e = getenv("MK_ATTN_S8")) attn_s8 = atoi(e);
        if (attn_s8) {
            int occ8 = 0;
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ8, pk_attn8<false>, A2_NT, A8_SMEM));
            if (occ8 != occ) attn_s8 = 0; // the split plan assumes pk_attn2's residency
        }
        mem_mark("prefill kernel attributes");
        pf_ready = true;
    }

    // causal attention of the chunk's T queries (positions n ..) over the cache of attention slot `slot` -> y2
    int a2_slots = 0;
    bool attn_old = false, attn_split = true, attn_pv16 = true;
    int attn_s8 = 1;
    void attn(int T, int n, int slot) {
        if (attn_old) {
            if (hb.kvq8) pk_attn<true><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n, slot);
            else pk_attn<false><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n, slot);
            return;
        }
        // y1 (k, v of the layer) is free between pk_attn_prep and the FFN: partials of split items
        const int N = (T + A2_Q - 1) / A2_Q * NKV;
        int nfull = N, S = 1;
        if (attn_split) a2_plan(T, n, a2_slots, (size_t) hp.cap * FF, nfull, S);
        const int grid = nfull + (N - nfull) * S;
        if (hb.kvq8) {
            // pk_attn8 reads plain q8_0 K and V only; MK_KV q4 or rotated-K slots stay on pk_attn2
            const bool s8 = attn_pv16 && hb.kvb[0][slot] == 8 && hb.kvb[1][slot] == 8 && !hb.kvr[slot];
            if (s8 && attn_s8 == 1) pk_attn8<false><<<grid, A2_NT, A8_SMEM>>>(T, n, slot, nfull, S, hp.y1);
            else if (s8 && attn_s8 == 2) pk_attn8<true><<<grid, A2_NT, A8_SMEM>>>(T, n, slot, nfull, S, hp.y1);
            else if (attn_pv16) pk_attn2<true, true><<<grid, A2_NT, A2_SMEM>>>(T, n, slot, nfull, S, hp.y1);
            else pk_attn2<true, false><<<grid, A2_NT, A2_SMEM>>>(T, n, slot, nfull, S, hp.y1);
        } else {
            if (attn_pv16) pk_attn2<false, true><<<grid, A2_NT, A2_SMEM>>>(T, n, slot, nfull, S, hp.y1);
            else pk_attn2<false, false><<<grid, A2_NT, A2_SMEM>>>(T, n, slot, nfull, S, hp.y1);
        }
        if (N > nfull) pk_attn2_comb<<<N - nfull, 256>>>(T, nfull, S, hp.y1);
    }

    // GEMM token tile (pf::mmh NB): every NB gives bit-identical results, so this is a pure speed choice.
    // At 128 a short prompt pads to 128 columns and the CTA is tensor-pipe bound on the padding (42 tokens: 2.1 us
    // per 64-K stage, 86 of 128 columns wasted); 32 / 64 cut the MMA work and add CTAs.
    // The choice is a small cost model over waves of CTAs (nrt row tiles x token tiles) on the nsm SMs, with the
    // per-stage time of a lone CTA at NB = 32 / 64 / 128 (0.80 / 0.95 / 1.44, relative; the model's Q4_K .. Q8_0
    // shapes, microbenchmark notes/ttft/gemm_nb.cu). NB 64 / 128 run one CTA per SM (shared memory), NB 32 two (a
    // pair takes 1.77x as long as one CTA). It picks 32 for <= 32 tokens; up to 64 tokens 64, or 32 when the GEMM is
    // narrow (40 row tiles x 2 token tiles still fit one wave); up to 128 tokens 128, or 64 when narrow; above that
    // mostly 128. Chunks of more than 384 tokens always use 128 (the original kernel).
    int pf_bn_force = 0; // MK_PF_BN
    int pf_nb(int ntok, int nrt) const {
        if (pf_bn_force) return pf_bn_force;
        if (ntok > 384) return 128;
        const int n32 = nrt * ((ntok + 31) / 32), n64 = nrt * ((ntok + 63) / 64), n128 = nrt * ((ntok + 127) / 128);
        const double t32 = 0.80 * (n32 <= nsm ? 1.0 : 1.77 * ((n32 + 2 * nsm - 1) / (2 * nsm)));
        const double t64 = 0.95 * ((n64 + nsm - 1) / nsm), t128 = 1.44 * ((n128 + nsm - 1) / nsm);
        return t32 <= t64 && t32 <= t128 ? 32 : t64 <= t128 ? 64 : 128;
    }
    template <int TYPE, bool GLU> void mmh_attr() {
        CK(cudaFuncSetAttribute(pf::mmh<TYPE, GLU, 32>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<TYPE, 32>()));
        CK(cudaFuncSetAttribute(pf::mmh<TYPE, GLU, 64>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<TYPE, 64>()));
        CK(cudaFuncSetAttribute(pf::mmh<TYPE, GLU, 128>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<TYPE, 128>()));
    }
    // nrt row tiles (CTAs along N) x token tiles of pf_nb tokens
    template <int TYPE, bool GLU>
    void mmh_launch(int nrt, const pf::WMat & W, const __half * X, int ntok, const pf::Out & O, const uint8_t * W2 = nullptr) {
        switch (pf_nb(ntok, nrt)) {
            case 32: pf::mmh<TYPE, GLU, 32><<<dim3(nrt, (ntok + 31) / 32), 256, pf::hm_smem<TYPE, 32>()>>>(W, X, ntok, O, W2); break;
            case 64: pf::mmh<TYPE, GLU, 64><<<dim3(nrt, (ntok + 63) / 64), 256, pf::hm_smem<TYPE, 64>()>>>(W, X, ntok, O, W2); break;
            default: pf::mmh<TYPE, GLU, 128><<<dim3(nrt, (ntok + 127) / 128), 256, pf::hm_smem<TYPE, 128>()>>>(W, X, ntok, O, W2); break;
        }
    }

    // y[t][row] (+)= W x_t for the first ntok fp16 rows of xh (token offset t0)
    void gemm(const Mat & m, int ntok, float * y, int ldy, int mode, int t0 = 0, const __half * xin = nullptr) {
        const pf::WMat W{m.w, m.d, m.type, m.K, m.N, m.rb, m.nd};
        const __half * X = (xin ? xin : hp.xh) + (size_t) t0 * m.K;
        const pf::Out O{y, ldy, mode, nullptr};
        const int nrt = m.N / pf::BM;
        switch (m.type) {
            case T_Q4K: mmh_launch<pf::TQ4K, false>(nrt, W, X, ntok, O); break;
            case T_Q5K: mmh_launch<pf::TQ5K, false>(nrt, W, X, ntok, O); break;
            case T_Q6K: mmh_launch<pf::TQ6K, false>(nrt, W, X, ntok, O); break;
            case T_Q8_0: mmh_launch<pf::TQ80, false>(nrt, W, X, ntok, O); break;
            case T_IQ4X: mmh_launch<pf::TIQ4, false>(nrt, W, X, ntok, O); break;
            case T_IQ4NL: mmh_launch<pf::TIQ4NL, false>(nrt, W, X, ntok, O); break;
            default: throw std::runtime_error("gemm: unsupported weight type");
        }
    }

    // xh2[t] = silu(Wg x_t) * (Wu x_t) for the first ntok rows of xh: one fused GEMM when gate and
    // up share a type with in-row scales, else two GEMMs into y0 / y1 and pk_glu
    static bool glu_fused(const Mat & g, const Mat & u) {
        return g.type == u.type && g.rb == u.rb && (g.type == T_Q4K || g.type == T_Q5K || g.type == T_IQ4X);
    }
    void gemm_glu(const Mat & g, const Mat & u, int ntok) {
        const pf::WMat W{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd};
        const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
        const int nrt = g.N / (pf::BM / 2);
        if (glu_fused(g, u)) {
            switch (g.type) {
                case T_Q4K: mmh_launch<pf::TQ4K, true>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_Q5K: mmh_launch<pf::TQ5K, true>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_IQ4X: mmh_launch<pf::TIQ4, true>(nrt, W, hp.xh, ntok, O, u.w); return;
                default: break;
            }
        }
        gemm(g, ntok, hp.y0, g.N, pf::OUT_STORE);
        gemm(u, ntok, hp.y1, u.N, pf::OUT_STORE);
        pk_glu<<<ntok, 256>>>(hp.y0, hp.y1, hp.xh2, g.N);
    }

    // MK_PF_PROFILE=1: per-category GPU time of the batched prefill (events around each launch)
    struct PfEv { const char * cat; cudaEvent_t a, b; };
    std::vector<PfEv> pf_ev;
    bool pf_prof = getenv("MK_PF_PROFILE") != nullptr;
    template <class F> void pf_time(const char * cat, F f) {
        if (!pf_prof) { f(); return; }
        // MK_PF_PROFILE=2: idempotent kernels are timed alone (5 repeats, min)
        static const char * idem[] = {"attn_prep", "attn_out", "dn_out", "norm_quant", "dn_ab", "attn", "embed", "gemm_ffn_gu", "gemm_ffn_gu2", "gemm_dn_in", "gemm_attn_qkv"};
        if (getenv("MK_PF_PROFILE")[0] == '2' && std::any_of(std::begin(idem), std::end(idem), [&](const char * c) { return !strcmp(c, cat); })) {
            float best = 1e30f;
            for (int r = 0; r < 5; r++) {
                cudaEvent_t a, b;
                CK(cudaEventCreate(&a));
                CK(cudaEventCreate(&b));
                CK(cudaDeviceSynchronize());
                CK(cudaEventRecord(a));
                f();
                CK(cudaEventRecord(b));
                CK(cudaEventSynchronize(b));
                float ms;
                CK(cudaEventElapsedTime(&ms, a, b));
                best = std::min(best, ms);
                cudaEventDestroy(a);
                cudaEventDestroy(b);
            }
            pf_iso[cat] += best;
            return;
        }
        PfEv e{cat, nullptr, nullptr};
        CK(cudaEventCreate(&e.a));
        CK(cudaEventCreate(&e.b));
        CK(cudaEventRecord(e.a));
        f();
        CK(cudaEventRecord(e.b));
        pf_ev.push_back(e);
    }
    std::map<std::string, double> pf_iso;
    void pf_report() {
        if (!pf_prof || pf_ev.empty()) return;
        CK(cudaDeviceSynchronize());
        // per category: sum, and median-per-launch x launches (robust to time-slicing by other GPU users)
        std::vector<std::string> cats;
        std::vector<std::vector<float>> ts;
        for (auto & e : pf_ev) {
            float ms;
            CK(cudaEventElapsedTime(&ms, e.a, e.b));
            auto it = std::find(cats.begin(), cats.end(), std::string(e.cat));
            if (it == cats.end()) { cats.push_back(e.cat); ts.push_back({}); it = cats.end() - 1; }
            ts[it - cats.begin()].push_back(ms);
            cudaEventDestroy(e.a);
            cudaEventDestroy(e.b);
        }
        pf_ev.clear();
        double all = 0, all_med = 0;
        std::vector<std::tuple<double, double, std::string, size_t>> rows;
        for (size_t i = 0; i < cats.size(); i++) {
            auto v = ts[i];
            double sum = 0;
            for (float x : v) sum += x;
            std::sort(v.begin(), v.end());
            const double med = v[v.size() / 2] * v.size();
            rows.emplace_back(med, sum, cats[i], v.size());
            all += sum;
            all_med += med;
        }
        for (auto & kv : pf_iso) fprintf(stderr, "  [isolated] %-12s %8.1f ms\n", kv.first.c_str(), kv.second);
        pf_iso.clear();
        std::sort(rows.rbegin(), rows.rend());
        fprintf(stderr, "[mk-prefill-profile] %.1f ms in kernels (median-based %.1f ms)\n", all, all_med);
        for (auto & r : rows)
            fprintf(stderr, "  %-14s median-based %8.1f ms %5.1f%%   sum %8.1f ms   x%zu\n", std::get<2>(r).c_str(), std::get<0>(r),
                    100 * std::get<0>(r) / all_med, std::get<1>(r), std::get<3>(r));
    }

    // Runs positions n0 .. n0 + toks.size() - 1 through the trunk (and the MTP block's K/V).
    // score: instead of MTP, greedy argmax of every position -> pred, and / or the logits of the
    // positions >= first -> sink.
    void pf_run(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink = nullptr,
                int first = 0) {
        pf_init();
        ck_wait_saved(); // the scratch may still be the source of the last checkpoint save
        ck_guard_state();
        const int N = (int) toks.size(), cap = hp.cap;
        for (int c0 = 0; c0 < N; c0 += cap) {
            const int T = std::min(cap, N - c0), n = n0 + c0;
            CK(cudaMemcpy(hp.tok, toks.data() + c0, T * sizeof(int), cudaMemcpyHostToDevice));
            ck_issue_restore(); // a due checkpoint restore streams in from here on (waited for per layer group below)
            pf_time("embed", [&] { pk_embed<<<T, 256>>>(T); });
            int dl = 0;
            for (int L = 0; L < NL; L++) {
                const LayerW & W = hw.L[L];
                if (fault_now && L == NL / 2) fault_inject();
                pf_time("norm_quant", [&] { pk_norm_quant<<<T, 256>>>(hp.x, W.attn_norm); });
                if (W.attn) {
                    pf_time("gemm_attn_qkv", [&] { gemm(W.q, T, hp.y0, QG, pf::OUT_STORE); });
                    pf_time("gemm_attn_qkv", [&] { gemm(W.k, T, hp.y1, 2 * KVD, pf::OUT_STORE); gemm(W.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE); });
                    pf_time("attn_prep", [&] { pk_attn_prep<<<T, 256>>>(n, L / 4, W.qn, W.kn, 1); });
                    pf_time("attn", [&] { attn(T, n, L / 4); });
                    pf_time("attn_out", [&] { pk_attn_out<<<T, 256>>>(); });
                    pf_time("gemm_attn_o", [&] { gemm(W.o, T, hp.x, D, pf::OUT_ADD); });
                } else {
                    pf_time("gemm_dn_in", [&] { gemm(W.qkv, T, hp.y0, QKV, pf::OUT_STORE); gemm(W.gate, T, hp.y1, VD, pf::OUT_STORE); });
                    pf_time("dn_ab", [&] { pk_ab<<<dim3((T + 63) / 64, AB_SPLITS), 256>>>(W.ab, W.abq, W.abd, W.attn_norm, T); });
                    ck_wait_restore(dl); // this layer's conv ring / state may still be arriving from the checkpoint
                    pf_time("dn_prep", [&] { pk_dn_prep<<<T, 256>>>(T, n, dl, W.conv, W.dt, W.A); pk_conv_ring<<<std::min(T, CONV_RING), 256>>>(T, n, dl); });
                    pf_time("dn_scan", [&] { pk_dn_scan<<<dim3(NVH, 2), 256>>>(T, dl); });
                    pf_time("dn_out", [&] { pk_dn_out<<<T, 256>>>(W.snorm); });
                    pf_time("gemm_dn_out", [&] { gemm(W.sout, T, hp.x, D, pf::OUT_ADD); });
                    dl++;
                }
                pf_time("norm_quant", [&] { pk_norm_quant<<<T, 256>>>(hp.x, W.post_norm); });
                // (profile: gate / up pairs that run as two GEMMs + pk_glu are reported as gemm_ffn_gu2)
                pf_time(glu_fused(W.fg, W.fu) ? "gemm_ffn_gu" : "gemm_ffn_gu2", [&] { gemm_glu(W.fg, W.fu, T); });
                pf_time("gemm_ffn_down", [&] { gemm(W.fd, T, hp.x, D, pf::OUT_ADD, 0, hp.xh2); });
            }
            wd_T = T;
            wd_mode = MODE_PREFILL;
            wait_device("a prefill chunk", std::max(wd_limit, 120.0));
            if (pred || sink) {
                pk_norm_quant<<<T, 256>>>(hp.x, hw.onorm);
                std::vector<float> rows;
                const int tile = std::min(128, cap * FF / VOCAB); // logits rows that fit in y0
                for (int s = 0; s < T; s += tile) {
                    const int nt = std::min(tile, T - s);
                    const int lo = std::max(s, first - n); // rows from lo on are at positions >= first
                    if (!pred && lo >= s + nt) continue;
                    gemm(hw.head, nt, hp.y0, VOCAB, pf::OUT_STORE, s);
                    if (pred) pk_argmax<<<nt, 256>>>(s);
                    if (sink && lo < s + nt) {
                        rows.resize((size_t) (s + nt - lo) * VOCAB);
                        CK(cudaMemcpy(rows.data(), hp.y0 + (size_t) (lo - s) * VOCAB, rows.size() * sizeof(float), cudaMemcpyDeviceToHost));
                        (*sink)(n + lo, s + nt - lo, rows.data());
                    }
                }
                if (pred) {
                    std::vector<int> am(T);
                    CK(cudaMemcpy(am.data(), hp.amax, T * sizeof(int), cudaMemcpyDeviceToHost));
                    pred->insert(pred->end(), am.begin(), am.end());
                }
                continue;
            }
            if (c0 == 0) { // hp.h row 0: the hidden state before the chunk, read only by pk_mtp_in (after every restore wait)
                if (n0 == 0) CK(cudaMemsetAsync(hp.h, 0, D * sizeof(float), 0));
                else CK(cudaMemcpyAsync(hp.h, hb.h_hist + (size_t) ((n0 - 1) % H_RING) * D, D * sizeof(float), cudaMemcpyDeviceToDevice, 0));
            }
            pf_time("mtp", [&] {
                pk_final_h<<<T, 256>>>(T, n);
                pk_mtp_in<<<T, 256>>>();
                gemm(hw.eh, T, hp.xm, D, pf::OUT_STORE);
                const LayerW & M = hw.L[NL];
                pk_norm_quant<<<T, 256>>>(hp.xm, M.attn_norm);
                gemm(M.k, T, hp.y1, 2 * KVD, pf::OUT_STORE);
                gemm(M.v, T, hp.y1 + KVD, 2 * KVD, pf::OUT_STORE);
                pk_attn_prep<<<T, 256>>>(n, 16, M.qn, M.kn, 0);
            });
            CK(cudaMemcpyAsync(hp.h, hp.h + (size_t) T * D, D * sizeof(float), cudaMemcpyDeviceToDevice));
        }
        CK(cudaGetLastError());
        wait_device("the prefill tail", std::max(wd_limit, 120.0));
        if (memlog && pf_runs_seen++ == 0) mem_mark("first prefill run");
        pf_report();
    }

    // cB.logits ([MAXT][VOCAB] fp32, 5 MB): only scoring launches (Launch::score == 2) write it
    void logits_alloc() {
        if (hb.logits) return;
        hb.logits = dalloc<float>((size_t) MAXT * VOCAB);
        CK(cudaMemcpyToSymbol(cB, &hb.logits, sizeof(hb.logits), offsetof(Bufs, logits)));
        mem_mark("scoring logits");
    }

    void reset() {
        ck_wait_restore();
        ck_guard_state();
        CK(cudaMemset(hb.dstate, 0, sizeof(float) * (size_t) 48 * NVH * 128 * 128));
        CK(cudaMemset(hb.h_hist, 0, sizeof(float) * (size_t) H_RING * D));
        memset(&hctl, 0, sizeof(hctl));
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        CK(cudaMemset(hb.cnt, 0, 16));
    }
};

MKEngine::MKEngine(const std::string & path, int ctx, int n_draft, const std::string & draft_vocab, int kv_mode)
    : p_(nullptr), n_draft_(n_draft) {
    if (n_draft < 0 || n_draft > MAXT - 1) throw std::runtime_error("n_draft must be in [0, " + std::to_string(MAXT - 1) + "]");
    p_ = new Impl(path, ctx, n_draft, draft_vocab, kv_mode);
    reset();
    if (const char * pol = getenv("MK_POLICY")) set_policy(pol);
#if MK_SPEC_LOG
    if (const char * f = getenv("MK_SPEC_LOG_FILE")) p_->slog = fopen(f, "a");
#endif
}

// "key=value,...": depth=0|1|2 (see Impl::Policy), dmin=N; on-device early stop (Launch::gate_*): drop=F,
// stop=F (every draft), dropK=F / stopK=F (draft K = 1..4 only), chain=0|1; h1=0|1 (sampled drafts when
// sampling); "" = defaults
void MKEngine::set_policy(const std::string & spec) {
    Impl::Policy pol;
    float dk[4] = {-1.f, -1.f, -1.f, -1.f}, sk[4] = {-1.f, -1.f, -1.f, -1.f}; // per-draft overrides
    size_t i = 0;
    while (i < spec.size()) {
        size_t j = spec.find(',', i);
        if (j == std::string::npos) j = spec.size();
        const std::string kv = spec.substr(i, j - i);
        i = j + 1;
        if (kv.empty()) continue;
        const size_t e = kv.find('=');
        if (e == std::string::npos) throw std::runtime_error("policy: expected key=value, got " + kv);
        const std::string k = kv.substr(0, e), v = kv.substr(e + 1);
        if (k == "depth") pol.depth = std::stoi(v);
        else if (k == "dmin") pol.dmin = std::stoi(v);
        else if (k == "drop") for (float & x : pol.gate_drop) x = std::stof(v);
        else if (k == "stop") for (float & x : pol.gate_stop) x = std::stof(v);
        else if (k.size() == 5 && (k.compare(0, 4, "drop") == 0 || k.compare(0, 4, "stop") == 0) && k[4] >= '1' &&
                 k[4] <= '4') // dropK / stopK: draft K alone (e.g. drop4: the T = 5 verify row)
            (k[0] == 'd' ? dk : sk)[k[4] - '1'] = std::stof(v);
        else if (k == "chain") pol.gate_chain = std::stoi(v);
        else if (k == "h1") pol.h1 = std::stoi(v);
        else throw std::runtime_error("policy: unknown key " + k);
    }
    for (int j = 0; j < 4; j++) {
        if (dk[j] >= 0.f) pol.gate_drop[j] = dk[j];
        if (sk[j] >= 0.f) pol.gate_stop[j] = sk[j];
    }
    p_->pol = pol;
}

MKEngine::~MKEngine() { delete p_; }

void MKEngine::bench_phases() { p_->bench(); }

void MKEngine::reset() {
    p_->reset();
    p_->seq.clear();
    p_->ck_pos = -1;
    p_->dv_reset();
    stats = MKStats();
}

int MKEngine::position() const { return p_->hctl.n; }

int MKEngine::prefill(const std::vector<int> & tokens) {
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    if (p_->fault_at && ++p_->fault_seen == p_->fault_at) p_->fault_now = true;
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    std::vector<int> & seq = p_->seq;
    if ((int) (seq.size() + tokens.size()) + 8 > p_->ctx) throw std::runtime_error("prompt exceeds context");
    seq.insert(seq.end(), tokens.begin(), tokens.end());
    p_->dv_add(tokens.data(), tokens.size());
    // Everything from the committed delta-net position on is fed: on a continued sequence that is
    // the last emitted token (not yet processed) and any accepted drafts not yet folded into the
    // recurrent state, followed by the new tokens.  Their KV / MTP rows are simply rewritten.
    const int n0 = c.dn_c;
    const std::vector<int> feed(seq.begin() + n0, seq.end());
    c.mtp_valid = std::min(c.mtp_valid, n0);
    size_t i = 0;
    const char * mode = getenv("MK_PREFILL");
    const bool batched = !(mode && std::string(mode) == "mk");
    if (batched && feed.size() >= 2) {
        // all but the last token in batched kernels; the last one goes through the megakernel,
        // which produces the first output token and the MTP drafts
        const std::vector<int> head(feed.begin(), feed.end() - 1);
        p_->pf_run(head, n0, nullptr);
        i = head.size();
        c.dn_c = n0 + (int) i;
        c.mtp_valid = c.dn_c;
    }
    while (i < feed.size()) {
        const int T = att_cap_T(c.dn_c, (int) std::min<size_t>(MAXT, feed.size() - i));
        const bool last = i + T == feed.size();
        c.n = c.dn_c;
        for (int k = 0; k < T; k++) c.tok[k] = feed[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->ndraft = p_->n_draft_max;
        p_->launch(MODE_PREFILL, T, last ? 1 : 0);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        i += T;
    }
    p_->nd_avail = p_->ndraft > 0 ? std::min(p_->ndraft, c.nd_prod) : 0;
    p_->all_acc = true;
    p_->last_acc = p_->nd_avail;
    CK(cudaGetLastError());
#if MK_SPEC_LOG
    p_->slog_drafts();
#endif
    p_->ck_save();
    seq.push_back(c.out[0]);
    return c.out[0];
}

void MKEngine::set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed) {
    if (temp > 0.f && (top_k <= 0 || top_k > TOPK_MAX))
        throw std::runtime_error("sampling needs 1 <= top_k <= " + std::to_string(TOPK_MAX));
    p_->smp_temp = temp;
    p_->smp_top_p = top_p;
    p_->smp_top_k = top_k;
    p_->smp_min_p = min_p;
    p_->smp_seed = seed;
}

const std::vector<int> & MKEngine::sequence() const { return p_->seq; }
int MKEngine::committed() const { return p_->hctl.dn_c; }
int MKEngine::checkpoint_pos() const { return p_->ck_pos; }
bool MKEngine::rewind_to_checkpoint() {
    if (p_->ck_pos < 0 || p_->ck_pos > (int) p_->seq.size()) return false;
    p_->ck_restore();
    return true;
}

void MKEngine::truncate(int len) {
    std::vector<int> & seq = p_->seq;
    if (len < p_->hctl.dn_c || len > (int) seq.size())
        throw std::runtime_error("truncate: can only drop tokens emitted after the committed position");
    seq.resize(len);
}

std::vector<int> MKEngine::score(const std::vector<int> & tokens) {
    reset();
    std::vector<int> pred;
    const char * mode = getenv("MK_PREFILL");
    if (!(mode && std::string(mode) == "mk")) {
        p_->pf_run(tokens, 0, &pred);
        reset();
        return pred;
    }
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    size_t i = 0;
    while (i < tokens.size()) {
        const int T = att_cap_T((int) i, (int) std::min<size_t>(MAXT, tokens.size() - i));
        c.n = (int) i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 1);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        for (int k = 0; k < T; k++) pred.push_back(c.tgt[k]);
        i += T;
    }
    return pred;
}

int MKEngine::n_vocab() { return VOCAB; }

void MKEngine::score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink) {
    const int N = (int) tokens.size();
    if (N + 8 > p_->ctx) throw std::runtime_error("sequence exceeds context");
    reset();
    p_->logits_alloc();
    mk_from = std::min(std::max(mk_from, 0), N);
    if (mk_from > 0)
        p_->pf_run(std::vector<int>(tokens.begin(), tokens.begin() + mk_from), 0, nullptr, first < mk_from ? &sink : nullptr, first);
    Ctl & c = p_->hctl;
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    c.dn_c = mk_from;
    std::vector<float> rows((size_t) MAXT * VOCAB);
    for (int i = mk_from; i < N;) {
        const int T = att_cap_T(i, std::min(MAXT, N - i));
        c.n = i;
        for (int k = 0; k < T; k++) c.tok[k] = tokens[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->launch(MODE_PREFILL, T, 0, 2);
        CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
        const int s = std::max(0, first - i);
        if (s < T) {
            CK(cudaMemcpy(rows.data(), p_->hb.logits + (size_t) s * VOCAB, (size_t) (T - s) * VOCAB * sizeof(float), cudaMemcpyDeviceToHost));
            sink(i + s, T - s, rows.data());
        }
        i += T;
    }
    reset();
}

int MKEngine::step(std::vector<int> & out) {
    Ctl & c = p_->hctl;
    if (c.n + MAXT + 2 >= p_->ctx) throw std::runtime_error("context full");
    // verify the previous launch's drafts, minus any that would put this launch's rows across an
    // attention-partition boundary (once per ATT_BUCKET positions; keeps greedy output depth-independent)
    const int T = att_cap_T(c.n, 1 + p_->nd_avail);
    // Draft depth (an upper bound: the on-device gate may stop drafting earlier). Default (depth 0):
    // after a cycle that rejected a draft, draft one token less (but at least 3), since a deeper draft
    // is then unlikely to be accepted; the full depth again after a fully accepted cycle.
    const Impl::Policy & pol = p_->pol;
    int nd = n_draft_;
    if (pol.depth == 0) nd = (!p_->all_acc && n_draft_ > 3) ? n_draft_ - 1 : n_draft_;
    else if (pol.depth == 2) nd = std::min(n_draft_, std::max(std::min(pol.dmin, n_draft_), 1 + p_->last_acc));
    p_->ndraft = nd;
#if MK_SPEC_LOG
    const int n0 = c.n;
    int drafts[MAXT];
    for (int i = 1; i < T; i++) drafts[i] = c.tok[i];
#endif
    p_->launch(MODE_DECODE, T, 0);
    CK(cudaMemcpy(&c, p_->hb.ctl, sizeof(Ctl), cudaMemcpyDeviceToHost));
    p_->nd_avail = nd > 0 ? std::min(nd, c.nd_prod) : 0;
#if MK_SPEC_LOG
    p_->slog_cycle(n0, T, drafts, c);
#endif
    for (int i = 0; i < c.n_out; i++) out.push_back(c.out[i]);
    p_->seq.insert(p_->seq.end(), c.out, c.out + c.n_out);
    p_->dv_add(c.out, c.n_out);
    p_->all_acc = c.n_acc == T - 1;
    p_->last_acc = c.n_acc;
    stats.cycles++;
    stats.drafted += T - 1;
    stats.accepted += c.n_acc;
    stats.last_T = T;
    stats.last_nd = p_->nd_avail;
    stats.last_acc = c.n_acc;
    return c.n_out;
}

std::vector<float> MKEngine::last_hidden() {
    std::vector<float> h(D);
    const int pos = p_->hctl.dn_c - 1;
    p_->ck_wait_restore();
    CK(cudaMemcpy(h.data(), p_->hb.h_hist + (size_t) (pos % H_RING) * D, D * 4, cudaMemcpyDeviceToHost));
    return h;
}
