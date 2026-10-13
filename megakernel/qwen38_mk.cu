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
#include <climits>
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
// Decode attention routing (compile-time, for A/B builds; both default on, all choices give bit-identical partials):
// MK_DA_ROUTE=1: chunks of one K/V tile (L_ctx <= ATT_BUCKET) run attn_core_t (no pipeline fill: faster there);
// MK_DA_Q4=1: q4 MK_KV slots run the pipelined kernel too (attn_core_q8m, attn_core_t's q4 -> fp16 values).
#ifndef MK_DA_ROUTE
#define MK_DA_ROUTE 1
#endif
#ifndef MK_DA_Q4
#define MK_DA_Q4 1
#endif
// MK_WV=1: the main-pass GEMV phases run the int8 tensor-core verify GEMM (wv.cuh) on a load-time fragment-ordered
// tile-major weight layout, at every T (T-invariant, not bit-identical to the dp4a GEMV). Default 0: today's dp4a path.
#ifndef MK_WV
#define MK_WV 0
#endif
// MK_WV_FMK (default 1): Q4_K / Q5_K main-pass matrices fragment-ordered like IQ4_XS. 0: tile-major whole blocks (the
// prefill reads whole 16-B chunks, but decode loses: -d 0 -5.8%, -d 4 +-0 vs dp4a instead of +3% / +7%; notes/agent_wvplumb.md)
#ifndef MK_WV_FMK
#define MK_WV_FMK 1
#endif
// MK_MAXT: the most tokens of one verify pass (1 + drafts). Default 5 (depth 4). 6..8 (depth 5..7) need MK_WV=1: the dp4a
// ffn_down cannot hold T >= 6 rows, so every main-layer ffn_down runs on wv (its Q6_K / Q3_K / IQ3_S too, tile-major),
// the conv ring has 16 slots, and the attention cores run two head-group passes at T >= 6 (48 query rows do not fit).
#ifndef MK_MAXT
#define MK_MAXT 5
#endif
#if MK_MAXT < 5 || MK_MAXT > 8 || (MK_MAXT > 5 && !MK_WV)
#error "MK_MAXT must be 5, or 6..8 with MK_WV=1"
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
constexpr int MAXT = MK_MAXT;
constexpr int MAXT_PF = 5; // rows of prefill-mode and scoring launches (T >= 6 runs only as a decode verify pass)
constexpr int NT = 512, NW = NT / 32;
constexpr float EPS = 1e-6f;
constexpr float ROPE_BASE = 10000000.0f;
// conv ring: dn_proj writes positions n .. n + T - 1 while dn_core reads n - 3 .. n + T - 1 (T + 3 slots)
constexpr int CONV_RING = MAXT > 5 ? 16 : 8, KV_RING = 16, H_RING = 16;
static_assert(MAXT + 3 <= CONV_RING && 2 * MAXT - 1 <= KV_RING, "rings");
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
// attn_core_q8 (decode attention, all-q8 slots): q8 K / V ring stages, double-buffered S and P
constexpr int AQ_NSK = 3, AQ_NSV = 4; // K(t) is issued in interval t - 2 and read in t; V(t) issued in t - 1, read in t + 2
constexpr int AQ_NSKM = 4;            // MIX (low-bit MK_KV slots): K(t) issued in t - 3, expanded to int8 in t - 1, read in t
#ifndef MK_XP_SM
#define MK_XP_SM 0 // MIX tile expansion: rows per tile done by the softmax warps (0: all 12 non-S warps round-robin)
#endif
#ifndef MK_AQ_EARLY
#define MK_AQ_EARLY 1 // MIX: issue the interval's K / V copies at the top of the interval (0: after the MMA work, as q8)
#endif
constexpr int AQ_LDS = ATP + 4;       // S row stride (floats): the softmax quads of 8 rows hit distinct banks; the pad
                                      // columns 32 .. 34 hold the rows' m, l and alpha (the 99 KB budget with AQ_NSKM)
constexpr int AQ_STG = ATP * KVQ_ROW; // one K or V ring stage
constexpr int SM_ATTN_Q8 = 32 * (256 + 8) * 2 + (AQ_NSKM + AQ_NSV) * AQ_STG + 2 * 32 * AQ_LDS * 4 + 2 * 32 * (ATP + 8) * 2;
static_assert(SM_ATTN_Q8 + 16 <= 99 * 1024, "attention ring budget (mk_cycle has a few bytes of static shared memory)");
constexpr int SM_ATTN_MAX = SM_ATTN > SM_ATTN_Q8 ? SM_ATTN : SM_ATTN_Q8;
constexpr int SM_TOTAL = (SM_MISC + 2048 > SM_ATTN_MAX ? SM_MISC + 2048 : SM_ATTN_MAX);
static_assert(SM_TOTAL <= 99 * 1024, "shared memory budget");

enum { T_F32 = 0, T_Q8_0 = 8, T_Q3K = 11, T_Q4K = 12, T_Q5K = 13, T_Q6K = 14, T_IQ4NL = 20, T_IQ3S = 21, T_IQ4X = 23 };
enum { MODE_PREFILL = 0, MODE_DECODE = 1, MODE_BENCH = 2 };

struct Mat {
    const uint8_t * w;
    const __half * d; // per-row block scales for repacked Q6_K / Q8_0
    int type, K, N, rb, nd;
#if MK_WV
    int fm; // 1: stored in the wv layout (wv.cuh; main-pass matrices of the wv types), read by the wv phases and mmh<.., 1>
#endif
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
    int mtp_pos[MAXT + 3], mtp_tok[MAXT + 3]; // MTP catch-up rows (<= T + 1)
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
    int kvrv[N_ATT_SLOTS]; // V stored Hadamard-rotated (MK_KV "6r" on V): the head's attention output is rotated back
    // MK_KV_WIN=W (recent-window ring): a side (w, slot) whose main buffer is low-bit also keeps its most recent rows as
    // q8 in a ring of kvrc rows per kv head: kvrp [kv head][kvrc][HD] values, kvrs [kv head][kvrc][HD / 32] scales
    // (nullptr: no ring). A tile at position p is read from the ring when p >= kv_ws(L_ctx); kvrc = (Wb + 1) ATT_BUCKET
    // with Wb = ceil(W / ATT_BUCKET), so the ring always holds every position the window can name.
    int8_t * kvrp[2][N_ATT_SLOTS];
    __half * kvrs[2][N_ATT_SLOTS];
    int kvwin, kvrc;
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
    unsigned short * pcnt; // [VOCAB] presence / frequency penalty: occurrences of each token in the response (saturating)
    unsigned * bar;
    unsigned * cnt;
    Ctl * ctl;
    unsigned long long * prof; // optional per-barrier timestamps (block 0)
    int ctx;
#if MK_KV_EMU
    int kvemu[2][N_ATT_SLOTS]; // per slot, K / V: 0 plain q8, else the kv_store_emu mode (MK_KV_EMU_SPEC)
    int kvemu_sink;            // positions < kvemu_sink stay plain q8 (attention sinks; MK_KV_EMU_SPEC "sink=N")
    int kvemu_win;             // > 0: rows are stored q8 and demoted to their mode once older than kvemu_win ("win=N")
#endif
    int dattn;   // decode attention of all-q8 slots: 1 attn_core_q8 (default), 0 attn_core_t (MK_DATTN=0)
    int mtp_win; // MK_MTP_WIN > 0: the MTP block's decode attention sees the last mtp_win positions only (drafts only)
#if MK_WV
    float * wvpart;  // wv split-K partial sums [WV_S][MAXT][D] (resid_wv)
    unsigned * wvcnt; // [2][D / 32] arrivals per 32-row group (out, ffn_down); the last arrival resets it to 0
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
    float gate_drop[MAXT - 1], gate_stop[MAXT - 1];
    int gate_chain;
    // sampling: draft from the MTP's own sampling distribution q' and verify with min(1, p / q') plus the
    // residual max(0, p - q') (distribution-exact; higher acceptance than argmax drafts at temp 1)
    int spec_smp;
    // presence / frequency penalty (vLLM's semantics, OpenAI's parameters): the logit of token v is lowered by
    // freq * c(v) + pres * (c(v) > 0), c(v) = how often v occurs in this response's generated tokens before the row's
    // position (Bufs::pcnt: the committed ones; a verify row adds the drafts in its own context). Applied to the main
    // decode rows and to the MTP draft head before the argmax / top-k; 0 / 0: off (the kernels without the penalty).
    float pres, freq;
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

#if MK_WV
// wv activation layout (wv.cuh, wv_fill): tokens per C-column group (8, or 4 for the per-16 split of X16), groups, rows
template <int T, bool X16> struct WvCfg {
    static constexpr int TPG = X16 ? 4 : 8;
    static constexpr int NG = (T + TPG - 1) / TPG;
    static constexpr int NTOKP = NG * TPG;
};
// the dp4a view (rows_* with XP = 2) of wv_fill<T, false, US>'s activation units: token stride US + 2
template <int T, int US>
__device__ __forceinline__ SX wv_sx() {
    constexpr int LDU = US + 2, NTOKP = WvCfg<T, false>::NTOKP;
    SX s;
    s.xa = reinterpret_cast<int4 *>(smem);
    s.xb = s.xa + NTOKP * LDU;
    s.xm = reinterpret_cast<uint2 *>(s.xb + NTOKP * LDU);
    s.U = LDU;
    return s;
}
#endif

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

template <int T, int IT, int R, int XP = 0>
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

template <int T, int IT, int R, int XP = 0>
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q6_K, 210 B per 256 weights like the GGUF block, in two streams per row:
//   rows (Mat::w, Q6_BLK = 192 B per block): qlo[128] (q4k-style nibbles) | qhi[64] (8 B per unit);
//   meta (Mat::d, 18 B per block, in pairs of blocks: sc[16] even | sc[16] odd | d even | d odd, 36 B).
// A unit reads its 4 scales sc[4p..4p+3] and the pair's two d (one 4-byte load each).
constexpr int Q6_BLK = 192, Q6_META = 18;
template <int T, int IT, int R, int XP = 0>
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// repacked Q8_0: per unit 32 B (A 16 int8 | B 16 int8), scales per natural 32-block
template <int T, int IT, int R, int XP = 0>
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
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
template <int T, int IT, int R, int XP = 0>
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
        accR0<T, R, 32 * IT + XP>(s, u, A, B, fA, fB, acc);
    }
}

// IQ4_NL re-laid-out at load into 144-byte blocks of 256 weights (repack_iq4nl): the fp16 scales d of the eight
// 32-weight blocks, then the 128 index bytes in Q4_K order (as IQ4_XS).  A weight is d * kvalues_iq4nl[index].
// Same integer products and the same float arithmetic (accR with zero mins) as the Q8_0 path these tensors were
// re-encoded for before, so the results are bit-identical to it at 0.53x the bytes.
template <int T, int IT, int R, int XP = 0>
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
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// Native Q3_K (repack_q3k): per row a value stream of Q3_BLK = 96 B per 256-block, unit u' = 2p + hh at 8 u' (the 2-bit
// lows: A's 16 in the first word, B's in the second, element 4j + m at bits 2j of byte m) and its high bits at 64 + 4 u'
// (byte m: bit j = A element 4j + m, bit 4 + j = B), plus the Q6_K meta stream (sc - 32 as int8, d; Mat::d). A weight is
// d (sc - 32) (u3 - 4), u3 = lows + 4 high. The GEMV feeds u3 + 28 = the Q6_K value (q + 32) the load-time re-encode
// stored, with the same scales and the same float arithmetic as rows_q6k: bit-identical to it at 114 / 210 the bytes.
constexpr int Q3_BLK = 96;
template <int T, int IT, int R, int XP = 0>
__device__ __forceinline__ void rows_q3k(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                         float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 2) : IT) // (4 at T < 5 spilled 84-152 B in resid32<1, 17 / 10, Q3_K>)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, hh = u & 1, w = u & 7;
        int2 lo[R];
        int hi[R], sc4[R], dd[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * Q3_BLK;
            const uint8_t * mp = reinterpret_cast<const uint8_t *>(dp[k]) + (b >> 1) * (2 * Q6_META);
            lo[k] = ldw2(blk + 8 * w);
            hi[k] = ldw1(blk + 64 + 4 * w);
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
            const unsigned x = (unsigned) lo[k].x, y = (unsigned) lo[k].y, h = (unsigned) hi[k];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                A[k][j] = (int) ((((x >> (2 * j)) & 0x03030303u) | (((h >> j) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
                B[k][j] = (int) ((((y >> (2 * j)) & 0x03030303u) | (((h >> (4 + j)) & 0x01010101u) << 2)) + 0x1c1c1c1cu);
            }
        }
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
    }
}

// iq3s_grid (ggml): 512 entries of four magnitudes {1, 3, .., 15}; filled at load from ggml's own dequantization
__device__ uint32_t g_iq3s_grid[512];
// (through L1, kept there: the weight streams bypass L1, so the 2 KB table stays resident)
__device__ __forceinline__ unsigned ldg_l1(const uint32_t * p) {
    unsigned r;
    asm("ld.global.nc.L1::evict_last.u32 %0, [%1];" : "=r"(r) : "l"(p));
    return r;
}
__device__ __forceinline__ unsigned iq3s_grid(unsigned idx) { return ldg_l1(g_iq3s_grid + idx); }
// Native IQ3_S (repack_iq3s): per row a value stream of 96 B per 256-block, unit u' = 2p + hh at 8 u' (the low bytes of
// the four grid indices of A (sub-block 2p, indices 4 hh ..) then of B (sub-block 2p + 1)) and its signs at 64 + 4 u'
// (byte m: bit c = A element 4c + m, bit 4 + c = B), and a meta stream of 16 B per block (Mat::d): qh[8] (byte u': the
// index high bits, A's in bits 0-3, B's in 4-7) | scales[4] (as in the file: byte p = sub-block 2p | 2p + 1 << 4) | fp16 d
// | pad. A weight is d (1 + 2 s) (+-grid). As rows_q3k, the GEMV feeds 32 +- grid = the Q6_K value of the load-time
// re-encode with the same scales and arithmetic as rows_q6k: bit-identical to it at 112 / 210 the bytes.
constexpr int IQ3S_META = 16;
template <int T, int IT, int R, int XP = 0>
__device__ __forceinline__ void rows_iq3s(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                          float (&acc)[R][T]) {
    const int lane = threadIdx.x & 31;
#pragma unroll(IT > 8 ? (T >= 5 ? 1 : 2) : IT) // (4 at T < 5 spilled 200-280 B in resid32<*, 17 / 10, IQ3_S>)
    for (int i = 0; i < IT; i++) {
        const int u = lane + 32 * i, b = u >> 3, p = (u >> 1) & 3, w = u & 7;
        int2 qs[R], sd[R];
        int sg[R], qh[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const uint8_t * blk = rp[k] + b * Q3_BLK;
            const uint8_t * mp = reinterpret_cast<const uint8_t *>(dp[k]) + b * IQ3S_META;
            qs[k] = ldw2(blk + 8 * w);
            sg[k] = ldw1(blk + 64 + 4 * w);
            qh[k] = ldw1(mp + 4 * (w >> 2));
            sd[k] = ldw2(mp + 8);
        }
        int A[R][4], B[R][4];
        float fA[R], gA[R], fB[R], gB[R];
#pragma unroll
        for (int k = 0; k < R; k++) {
            const float d = __half2float(__ushort_as_half((unsigned short) (sd[k].y & 0xffff)));
            const unsigned sc = (unsigned) sd[k].x >> (8 * p);
            fA[k] = d * (float) (int) (1 + 2 * (sc & 0xfu));
            fB[k] = d * (float) (int) (1 + 2 * ((sc >> 4) & 0xfu));
            gA[k] = 32.0f * fA[k];
            gB[k] = 32.0f * fB[k];
            const unsigned h = (unsigned) qh[k] >> (8 * (w & 3)), x = (unsigned) qs[k].x, y = (unsigned) qs[k].y,
                           sv = (unsigned) sg[k];
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const unsigned ga = iq3s_grid(((x >> (8 * c)) & 0xffu) | (((h >> c) & 1u) << 8));
                const unsigned gb = iq3s_grid(((y >> (8 * c)) & 0xffu) | (((h >> (4 + c)) & 1u) << 8));
                const unsigned ma = ((sv >> c) & 0x01010101u) * 0xffu, mb = ((sv >> (4 + c)) & 0x01010101u) * 0xffu;
                // 32 +- g per byte: (g ^ m) - m is -g in the bytes where m = 0xff (the borrows cancel in the sum)
                A[k][c] = (int) (0x20202020u + (ga ^ ma) - ma);
                B[k][c] = (int) (0x20202020u + (gb ^ mb) - mb);
            }
        }
        accR<T, R, 32 * IT + XP>(s, u, A, B, fA, gA, fB, gB, acc);
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
         : TYPE == T_Q3K || TYPE == T_IQ3S ? (u0 >> 3) * Q3_BLK
                                          : 32 * u0;
}
// R rows of one weight type given their row pointers (and, for Q8_0, their scale rows)
template <int T, int IT, int R, int TYPE, int XP = 0>
__device__ __forceinline__ void rows_t(const uint8_t * const (&rp)[R], const __half * const (&dp)[R], const SX & s,
                                       float (&acc)[R][T]) {
    if constexpr (TYPE == T_Q4K) rows_q4k<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_Q5K) rows_q5k<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_Q6K) rows_q6k<T, IT, R, XP>(rp, dp, s, acc);
    else if constexpr (TYPE == T_IQ4X) rows_iq4x<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_IQ4NL) rows_iq4nl<T, IT, R, XP>(rp, s, acc);
    else if constexpr (TYPE == T_Q3K) rows_q3k<T, IT, R, XP>(rp, dp, s, acc);
    else if constexpr (TYPE == T_IQ3S) rows_iq3s<T, IT, R, XP>(rp, dp, s, acc);
    else rows_q80<T, IT, R, XP>(rp, dp, s, acc);
}
template <int T, int IT, int R, int TYPE, int XP = 0>
__device__ __forceinline__ void mrows(const Mat & m, int row0, const SX & s, float (&acc)[R][T], int u0 = 0) {
    const uint8_t * rp[R];
    const __half * dp[R];
#pragma unroll
    for (int k = 0; k < R; k++) {
        rp[k] = m.w + (size_t) (row0 + k) * m.rb + unit_bytes<TYPE>(u0);
        // Q6_K / Q3_K: meta (9 halves per block), IQ3_S: meta (8 halves per block), Q8_0: d
        dp[k] = m.d + (size_t) (row0 + k) * m.nd +
                (TYPE == T_Q6K || TYPE == T_Q3K ? (u0 >> 3) * (Q6_META / 2) : TYPE == T_IQ3S ? (u0 >> 3) * (IQ3S_META / 2) : u0);
    }
    rows_t<T, IT, R, TYPE, XP>(rp, dp, s, acc);
}

// Row pairs [lo, hi) of a single-matrix item range, distributed over all warps of the grid.
// out(row0, acc) receives the reduced sums for rows row0, row0 + 1.
template <int T, int IT, int TYPE, int XP = 0, class Out>
__device__ __noinline__ void gemv_pairs(const Mat m, int lo, int hi, Out out) {
#if MK_WV
    const SX sx = XP ? wv_sx<T, 32 * IT>() : make_sx(nullptr, 32 * IT); // activations already in shared memory
#else
    const SX sx = make_sx(nullptr, 32 * IT); // activations already in shared memory
#endif
    const int W = nwarps(), gw = gwarp();
    const int first = lo + (((gw - lo) % W) + W) % W;
    for (int it = first; it < hi; it += W) {
        const int r = 2 * (it - lo);
        float a[2][T];
        zero_acc<T, 2>(a);
        mrows<T, IT, 2, TYPE, XP>(m, r, sx, a);
        wsum_acc<T, 2>(a);
        out(r, a);
    }
}

// Runtime dispatch on a matrix's type (uniform across the grid). Plain switches: a capturing
// lambda would take the address of the caller's locals and push them to local memory.
// (mk_cycle's stack frame is the deepest call chain incl. the callees' spills: a spilling GEMV instantiation of any
// type raises the local memory of the whole kernel, ~0.12 MiB per byte. A second out-of-line switch for the newer types
// was worse, 816 -> 1232 B: notes/agent_unsloth.md.)
template <int T, int IT, class Out>
__device__ __forceinline__ void gemv_pairs_any(const Mat & m, int lo, int hi, Out out) {
    switch (m.type) {
        case T_Q4K: gemv_pairs<T, IT, T_Q4K>(m, lo, hi, out); break;
        case T_Q5K: gemv_pairs<T, IT, T_Q5K>(m, lo, hi, out); break;
        case T_Q6K: gemv_pairs<T, IT, T_Q6K>(m, lo, hi, out); break;
        case T_Q8_0: gemv_pairs<T, IT, T_Q8_0>(m, lo, hi, out); break;
        case T_IQ4NL: gemv_pairs<T, IT, T_IQ4NL>(m, lo, hi, out); break;
        case T_Q3K: gemv_pairs<T, IT, T_Q3K>(m, lo, hi, out); break;
        case T_IQ3S: gemv_pairs<T, IT, T_IQ3S>(m, lo, hi, out); break;
        default: gemv_pairs<T, IT, T_IQ4X>(m, lo, hi, out); break;
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

// token embedding rows -> x, plus quantised attn_norm input for layer 0.  Block s of row t handles the 16 units of
// super-blocks 2s, 2s+1 (one per warp; 10 blocks per row, U_D / 16): a single block per row spent ~12 us in its
// serial per-unit quantise chain while the grid waited (the row fetch itself is ~free, MK_EMBD_DEVICE A/B)
constexpr int EMB_UB = 16; // units per embed block
static_assert(U_D % EMB_UB == 0 && EMB_UB == NW, "embed_prologue: one unit per warp");
__device__ __forceinline__ void stage_embd_part(int tok, int b0, int nbytes, uint8_t * dst) {
    if (threadIdx.x < nbytes / 16)
        reinterpret_cast<uint4 *>(dst)[threadIdx.x] = reinterpret_cast<const uint4 *>(cW.embd + (size_t) tok * EMBD_RB + b0)[threadIdx.x];
    __syncthreads();
}
template <int T>
__device__ __noinline__ void embed_prologue() {
    constexpr int NB = U_D / EMB_UB; // blocks per row
    const int t = blockIdx.x / NB, s = blockIdx.x - t * NB;
    if (t >= T) return;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int tok = ((volatile Ctl *) cB.ctl)->tok[t];
    // units 16 s .. 16 s + 15 = elements [512 s, 512 s + 512) = Q4_K super-blocks 2 s, 2 s + 1 (288 bytes)
    stage_embd_part(tok, 2 * s * 144, 2 * 144, smem);
    const float * w0 = cW.L[0].attn_norm;
    const int j = EMB_UB * s + warp, e = 32 * j + lane;
    const float v = dequant_q4k(smem, e - 512 * s);
    cB.x[t * D + e] = v;
    q32_store(cB.qn, t, j, v * w0[e], v * v, true);
}

// ffn_down (K = FF) whole-K rows out of line (half_rows) at the T of mask bit T: inlined into resid32's item loop,
// resid32<2..4,17,*> spilled 20-88 B and the phase ran 0.17-0.25 ms per pass slower (notes/agent_mtpcut.md). T = 5
// splits K anyway; the K = 5120 / 6144 roles stay inline (out of line they got slower).
#ifndef MK_FD_NOINL
#define MK_FD_NOINL 0x1c
#endif
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
            if constexpr (IT == 17 && ((MK_FD_NOINL >> T) & 1)) half_rows<T, IT, TYPE>(W, 32 * j + 2 * warp, sx, a, 0);
            else mrows<T, IT, 2, TYPE>(W, 32 * j + 2 * warp, sx, a);
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
        case T_Q3K: resid32<T, IT, T_Q3K>(W, in, x, add, wnext, sm); break;
        case T_IQ3S: resid32<T, IT, T_IQ3S>(W, in, x, add, wnext, sm); break;
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
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const uint8_t * gw = W.fg.w;
    const uint8_t * uw = W.fu.w;
    const int rb = W.fg.rb;
    for (int j = gwarp(); j < FF / 16; j += nwarps()) {
        float act[T];
#pragma unroll
        for (int t = 0; t < T; t++) act[t] = 0.f;
        // one row per matrix per step at every T: the 2 + 2-row step T <= 2 used to take was slower (T = 1 / 2 ffn_gu
        // 10.2 / 10.9 -> 9.0 / 9.4 ms per pass on UD-Q4_K_XL, where it spilled; notes/agent_mtpcut.md)
        constexpr int RG = 1;
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
template <int T, int TYPE, int XP = 0>
__device__ __noinline__ LaneT<T> rows16(const Mat m, int j) {
    const int lane = threadIdx.x & 31;
#if MK_WV
    const SX sx = XP ? wv_sx<T, U_D>() : make_sx(nullptr, U_D);
#else
    const SX sx = make_sx(nullptr, U_D);
#endif
    const float * s_rs = misc(nullptr);
    constexpr int R = 2; // at every T (4 rows at T <= 2 spilled 44-160 B and ran 1.1-1.5 ms per pass slower)
    LaneT<T> res;
#pragma unroll
    for (int t = 0; t < T; t++) res.v[t] = 0.f;
#pragma unroll 1
    for (int r = 0; r < 16; r += R) {
        float a[R][T];
        zero_acc<T, R>(a);
        mrows<T, 5, R, TYPE, XP>(m, 16 * j + r, sx, a);
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
        case T_Q3K: return rows16<T, T_Q3K>(m, j);
        case T_IQ3S: return rows16<T, T_IQ3S>(m, j);
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

#if MK_WV
// MK_WV builds: one store functor type for every gemv_pairs call site. A lambda per site made one gemv_pairs instance per
// (T, type, site): ~47% of the module's PTX, and ptxas's memory grows with it (> 25 GB for the merged MK_WV build;
// notes/agent_wvplumb.md "wvplumb2"). Rows t of base (or, ring >= 0, the conv ring rows (ring + t) % CONV_RING), scaled by
// the RMS-norm scale s_rs[t]: the stores of dn_proj's / attn_proj's lambdas, the same values.
struct PairOut {
    float * base;
    int stride, ring;
    template <int T> __device__ __forceinline__ void operator()(int r, const float (&a)[2][T]) const {
        const int lane = threadIdx.x & 31;
        const float * s_rs = misc(nullptr);
#pragma unroll
        for (int t = 0; t < T; t++)
            if ((lane >> 1) == t) {
                const int row = ring >= 0 ? (ring + t) % CONV_RING : t;
                base[row * stride + r + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * s_rs[t];
            }
    }
};
#endif

// delta-net input projections: alpha/beta (fp32), qkv -> conv ring, z
template <int T>
__device__ __noinline__ void dn_proj(const LayerW & W, int dl, int n, int dn_c, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    const SX sx = make_sx(sm, U_D);
    float * s_rs = misc(sm);
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
#if MK_WV
    gemv_pairs_any<T, 5>(W.qkv, n_ab, n_ab + n_qkv, PairOut{conv, QKV, n});
    gemv_pairs_any<T, 5>(W.gate, n_ab + n_qkv, n_ab + n_qkv + n_z, PairOut{cB.z, VD, -1});
#else
    gemv_pairs_any<T, 5>(W.qkv, n_ab, n_ab + n_qkv, [=](int r, const float (&a)[2][T]) {
        const int lane = threadIdx.x & 31;
#pragma unroll
        for (int t = 0; t < T; t++)
            if ((lane >> 1) == t)
                conv[((n + t) % CONV_RING) * QKV + r + (lane & 1)] = ((lane & 1) ? a[1][t] : a[0][t]) * misc(nullptr)[t];
    });
    gemv_pairs_any<T, 5>(W.gate, n_ab + n_qkv, n_ab + n_qkv + n_z,
                         [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.z, VD, r, a); });
#endif
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

// q5 / q6 / q7 store of one KV row (HD bits / 8 bytes): the q4 word of each lane (low nibbles of the 2^bits-level index
// n = q + 2^(bits-1), kv_store_q4's nibble order) in the first 128 bytes, then the high bits, bits - 4 per value: lane l's
// 8 values at bytes 128 + l (bits - 4) .., value k at bit (bits - 4) k. q4_0-style scale d = (signed max) / -2^(bits-1),
// value = (n - 2^(bits-1)) d. bits = 4 would give kv_store_q4's bytes.
__device__ __forceinline__ void kv_store_qn(int8_t * q, __half * sc, const float (&v)[8], int bits) {
    const int lane = threadIdx.x & 31, h = 1 << (bits - 1), hb = bits - 4;
    float am = 0.f, sv = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++)
        if (fabsf(v[k]) > am || (fabsf(v[k]) == am && v[k] > sv)) { am = fabsf(v[k]); sv = v[k]; }
#pragma unroll
    for (int m = 1; m <= 2; m <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, m), os = __shfl_xor_sync(0xffffffffu, sv, m);
        if (oa > am || (oa == am && os > sv)) { am = oa; sv = os; }
    }
    const float d = sv / (float) -h, id = d != 0.f ? 1.0f / d : 0.f;
    unsigned lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const unsigned n = (unsigned) min(2 * h - 1, (int) (v[k] * id + ((float) h + 0.5f)));
        lo |= (n & 15u) << (k < 4 ? 8 * k : 8 * (k - 4) + 4);
        hi |= (n >> 4) << (hb * k);
    }
    *reinterpret_cast<unsigned *>(q + lane * 4) = lo;
    for (int k = 0; k < hb; k++) q[128 + lane * hb + k] = (int8_t) ((hi >> (8 * k)) & 0xffu);
    if ((lane & 3) == 0) sc[lane >> 2] = __float2half_rn(d);
}

// one K (w = 0) or V (w = 1) row (kv head g, position pos) of attention slot `slot` in its MK_KV format; with a recent-window
// ring (MK_KV_WIN) the row is also stored q8 in the ring (same rotation: every tile of a slot must share the basis)
__device__ __forceinline__ void kv_store(int w, int slot, int g, int pos, const float (&vin)[8]) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (w == 0 ? cB.kvr[slot] : cB.kvrv[slot]) fwht256(v);
    const int bits = cB.kvb[w][slot];
    const size_t row = (size_t) g * cB.ctx + pos;
    __half * sc = cB.kvs[w][slot] + row * (HD / 32);
    if (bits == 8) kv_store_q8(cB.kvp[w][slot] + row * HD, sc, v);
    else if (bits == 4) kv_store_q4(cB.kvp[w][slot] + row * (HD / 2), sc, v);
    else kv_store_qn(cB.kvp[w][slot] + row * (HD * bits / 8), sc, v, bits);
    if (cB.kvrp[w][slot]) {
        const size_t rr = (size_t) g * cB.kvrc + pos % cB.kvrc;
        kv_store_q8(cB.kvrp[w][slot] + rr * HD, cB.kvrs[w][slot] + rr * (HD / 32), v);
    }
}

// MK_KV_WIN: the window start for context length L_ctx (positions >= ws are read from the q8 ring). Stepped by ATT_BUCKET
// like the attention partition, so all rows of one launch (att_cap_T) and both draft depths see the same ring / main
// split of the tiles: greedy output stays independent of the draft depth. The window covers Wb ATT_BUCKET + 1 ..
// (Wb + 1) ATT_BUCKET positions (>= W), the ring holds (Wb + 1) ATT_BUCKET rows.
__device__ __forceinline__ int kv_ws(int L_ctx) {
    if (cB.kvwin <= 0) return 0;
    const int ws = (L_ctx + ATT_BUCKET - 1) / ATT_BUCKET - 1 - (cB.kvwin + ATT_BUCKET - 1) / ATT_BUCKET;
    return ws > 0 ? ws * ATT_BUCKET : 0;
}
// bits of the tile at position p0 (a multiple of ATP) of side w of a slot: 8 from the ring, else the main format
__device__ __forceinline__ int kv_tbits(int w, int slot, int p0, int ws) {
    return (cB.kvrp[w][slot] != nullptr && p0 >= ws) ? 8 : cB.kvb[w][slot];
}
// ws % kvrc for kv_src (computed once per launch / item: a runtime integer modulo is ~25 dependent instructions)
__device__ __forceinline__ int kv_wsm(int ws) { return cB.kvrc > 0 ? ws % cB.kvrc : 0; }
// the tile's rows: q / s = values / scales of its first row (ring or main buffer), returns its bits. wsm = kv_wsm(ws):
// a ring tile's row is p0 % kvrc = wsm + (p0 - ws) folded once (positions ws .. L_ctx - 1 span at most kvrc rows)
__device__ __forceinline__ int kv_src(int w, int slot, int g, int p0, int ws, int wsm, const int8_t *& q, const __half *& s) {
    if (cB.kvrp[w][slot] != nullptr && p0 >= ws) {
        int r0 = wsm + (p0 - ws);
        if (r0 >= cB.kvrc) r0 -= cB.kvrc;
        const size_t rr = (size_t) g * cB.kvrc + r0;
        q = cB.kvrp[w][slot] + rr * HD;
        s = cB.kvrs[w][slot] + rr * (HD / 32);
        return 8;
    }
    const int bits = cB.kvb[w][slot];
    const size_t row = (size_t) g * cB.ctx + p0;
    q = cB.kvp[w][slot] + row * (HD * bits / 8);
    s = cB.kvs[w][slot] + row * (HD / 32);
    return bits;
}

#if MK_KV_EMU

// Lloyd-Max (MSE-optimal) codebooks for N(0,1), positive half (symmetric), 2..6 bits: notes/kvq/lloyd_max.py.
// MSE / variance: 0.1175 (2 bits), 0.03455 (3), 0.009501 (4), 0.002505 (5), 0.000644 (6).
__constant__ float LM_LEVELS[62] = {
    0.4527800f, 1.5104176f,                                                                                     // 2 bits
    0.2450942f, 0.7560053f, 1.3439093f, 2.1519457f,                                                             // 3
    0.1283950f, 0.3880483f, 0.6567591f, 0.9423405f, 1.2562312f, 1.6180464f, 2.0690172f, 2.7325896f,             // 4
    0.0658897f, 0.1980518f, 0.3313783f, 0.4666995f, 0.6049336f, 0.7471357f, 0.8945651f, 1.0487833f,             // 5
    1.2118044f, 1.3863403f, 1.5762281f, 1.7872332f, 2.0287284f, 2.3177394f, 2.6911196f, 3.2607325f,
    0.0335182f, 0.1006038f, 0.1678377f, 0.2353205f, 0.3031553f, 0.3714493f, 0.4403149f, 0.5098710f,             // 6
    0.5802446f, 0.6515730f, 0.7240059f, 0.7977081f, 0.8728625f, 0.9496748f, 1.0283779f, 1.1092386f,
    1.1925660f, 1.2787219f, 1.3681356f, 1.4613233f, 1.5589154f, 1.6616948f, 1.7706536f, 1.8870784f,
    2.0126849f, 2.1498391f, 2.3019465f, 2.4741863f, 2.6750484f, 2.9200457f, 3.2428991f, 3.7463072f,
};
__constant__ int LM_OFF[5] = {0, 2, 6, 14, 30}; // bits 2..6
// fixed random signs for mode 's' (randomised Hadamard): bit 8 lane + k flips value k of the lane
__constant__ unsigned EMU_SIGNS[8] = {0x6818f096u, 0x0205cd15u, 0xca000ab0u, 0x0ce98e8du, 0x005d3defu, 0xf897127cu, 0x0b0b0cf8u, 0x8fd7fa69u};

// A K / V row rounded to what a low-bit cache format would reconstruct, written in the q8 layout (accuracy experiments,
// MK_KV_EMU_SPEC). mode: bits (2..8) | 16 Hadamard rotation (WHT) first | 32 q4_0-style scale d = (signed max) / -2^(b-1)
// (else symmetric d = max |x| / (2^(b-1) - 1)) | 64 Lloyd-Max codebook for the (near-Gaussian, rotated) coordinates with
// scale = rms of the group | 128 scale group = the whole 256-value row (else 32 values) | 256 random signs before the
// WHT (randomised Hadamard; undone after the inverse). The scale is fp16. A plain integer grid without rotation is
// stored exactly (integers + scale, as a real kernel would: "4a" is bit-identical to kv_store_q4); everything else is
// dequantised, transformed back and stored as ordinary q8 (adds q8 rounding). Positions < cB.kvemu_sink stay plain q8.
__device__ __noinline__ void kv_store_emu(int8_t * q, __half * sc, const float (&vin)[8], int mode, int pos) {
    const int lane = threadIdx.x & 31, bits = mode & 15;
    const bool rot = mode & (16 | 256), q40 = mode & 32, lm = mode & 64, rowg = mode & 128, sgn = mode & 256;
    const bool ta = mode & (2048 | 4096); // TurboAngle (t: 4-bit log norms, T: 8-bit linear norms); 1024: demotion pass
    if (pos < cB.kvemu_sink || (cB.kvemu_win > 0 && !(mode & 1024))) {
        kv_store_q8(q, sc, vin);
        return;
    }
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = vin[k];
    if (sgn) {
#pragma unroll
        for (int k = 0; k < 8; k++)
            if ((EMU_SIGNS[lane >> 2] >> ((lane & 3) * 8 + k)) & 1u) v[k] = -v[k];
    }
    if (rot) fwht256(v);
    float r[8]; // the reconstruction
    if (ta) {
        // TurboAngle (2603.27467): after the randomised Hadamard, consecutive pairs -> (norm, angle); angles are uniform
        // on the circle and quantised to 2^bits bins; the 128 pair norms of the row are min-max quantised (t: 4 bits in
        // log space, T: 8 bits linear) with the row's min / max kept in fp32
        const int nb = 1 << bits;
        const bool lg = mode & 2048;
        const float ql = (float) ((1 << (lg ? 4 : 8)) - 1);
        float rn[4], th[4], mn = INFINITY, mx = -INFINITY;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float x = v[2 * j], y = v[2 * j + 1];
            th[j] = atan2f(y, x);
            const float nrm = sqrtf(x * x + y * y);
            rn[j] = lg ? logf(fmaxf(nrm, 1e-30f)) : nrm;
            mn = fminf(mn, rn[j]);
            mx = fmaxf(mx, rn[j]);
        }
#pragma unroll
        for (int m = 1; m <= 16; m <<= 1) {
            mn = fminf(mn, __shfl_xor_sync(0xffffffffu, mn, m));
            mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, m));
        }
        const float span = mx - mn, qs = span > 0.f ? ql / span : 0.f, iqs = span > 0.f ? span / ql : 0.f;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const float u = mn + (float) __float2int_rn((rn[j] - mn) * qs) * iqs;
            const float nrm = lg ? expf(u) : u;
            int ka = __float2int_rn(th[j] * ((float) nb / 6.283185307f));
            ka = ((ka % nb) + nb) % nb;
            const float a = (float) ka * (6.283185307f / (float) nb);
            r[2 * j] = nrm * cosf(a);
            r[2 * j + 1] = nrm * sinf(a);
        }
    } else {
    // group statistics: largest |x| and that x (ties: the larger x, the same in every lane of the group), sum of squares
    float am = 0.f, sv = 0.f, ss = 0.f;
#pragma unroll
    for (int k = 0; k < 8; k++) {
        ss += v[k] * v[k];
        if (fabsf(v[k]) > am || (fabsf(v[k]) == am && v[k] > sv)) { am = fabsf(v[k]); sv = v[k]; }
    }
    const int gm = rowg ? 16 : 2;
    for (int m = 1; m <= gm; m <<= 1) {
        const float oa = __shfl_xor_sync(0xffffffffu, am, m), os = __shfl_xor_sync(0xffffffffu, sv, m);
        if (oa > am || (oa == am && os > sv)) { am = oa; sv = os; }
        ss += __shfl_xor_sync(0xffffffffu, ss, m);
    }
    const int h = 1 << (bits - 1);
    const float d = lm ? sqrtf(ss / (rowg ? 256.f : 32.f)) : q40 ? sv / (float) -h : am / (float) (h - 1);
    const __half dh = __float2half_rn(d);
    const float df = __half2float(dh);
    if (lm) {
        const float * lv = LM_LEVELS + LM_OFF[bits - 2];
        const float id = df > 0.f ? 1.0f / df : 0.f;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const float a = fabsf(v[k]) * id;
            int i = 0; // nearest level: thresholds are the midpoints
            for (int j = 1; j < h; j++) i = a >= 0.5f * (lv[j - 1] + lv[j]) ? j : i;
            r[k] = copysignf(df * lv[i], v[k]);
        }
    } else {
        const float id = d != 0.f ? 1.0f / d : 0.f;
        int qi[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            qi[k] = q40 ? min(2 * h - 1, (int) (v[k] * id + ((float) h + 0.5f))) - h : max(1 - h, min(h - 1, __float2int_rn(v[k] * id)));
            r[k] = df * (float) qi[k];
        }
        if (!rot && !rowg) { // exact store of the integers and the fp16 scale
            unsigned w[2] = {0u, 0u};
#pragma unroll
            for (int k = 0; k < 8; k++) w[k >> 2] |= ((unsigned) qi[k] & 0xffu) << (8 * (k & 3));
            *reinterpret_cast<uint2 *>(q + lane * 8) = make_uint2(w[0], w[1]);
            if ((lane & 3) == 0) sc[lane >> 2] = dh;
            return;
        }
    }
    } // grid modes
    if (rot) {
        fwht256(r);
        if (sgn) {
#pragma unroll
            for (int k = 0; k < 8; k++)
                if ((EMU_SIGNS[lane >> 2] >> ((lane & 3) * 8 + k)) & 1u) r[k] = -r[k];
        }
    }
    kv_store_q8(q, sc, r);
}

// Recent-window emulation ("win=N"): rows are stored as plain q8 and demoted to their slot's mode once they are older
// than N positions. Re-quantises rows of positions p0 .. p1 - 1 of every emulated K / V slot (one warp per row; q8
// in, kv_store_emu with the force bit out; sink rows are re-stored as q8, which is idempotent).
__global__ void __launch_bounds__(256) k_kv_demote(int p0, int p1) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = p1 - p0;
    for (int i = blockIdx.x * 8 + warp; i < n * NKV * N_ATT_SLOTS * 2; i += gridDim.x * 8) {
        const int w = i & 1, slot = (i >> 1) % N_ATT_SLOTS, g = ((i >> 1) / N_ATT_SLOTS) % NKV;
        const int p = p0 + (i >> 1) / (N_ATT_SLOTS * NKV);
        const int mode = cB.kvemu[w][slot];
        if (!mode) continue;
        const size_t row = ((size_t) slot * NKV + g) * cB.ctx + p;
        int8_t * q = static_cast<int8_t *>(w ? cB.vc : cB.kc) + row * HD;
        __half * sc = (w ? cB.vsc : cB.ksc) + row * (HD / 32);
        const uint2 qq = *reinterpret_cast<const uint2 *>(q + lane * 8);
        const float d = __half2float(sc[lane >> 2]);
        float v[8];
#pragma unroll
        for (int k = 0; k < 8; k++) v[k] = d * (float) (int8_t) ((k < 4 ? qq.x : qq.y) >> (8 * (k & 3)));
        __syncwarp();
        kv_store_emu(q, sc, v, mode | 1024, p);
    }
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

// the high bits of one lane word's 8 values of a q5 / q6 / q7 row: hb = bits - 4 bytes at p, value k at bit hb k
__device__ __forceinline__ unsigned kv_hi(const uint8_t * p, int hb) {
    unsigned x = p[0];
    if (hb > 1) x |= (unsigned) p[1] << 8;
    if (hb > 2) x |= (unsigned) p[2] << 16;
    return x;
}
// the hb-bit fields k0 .. k0 + 3 of x as the high nibbles of 4 bytes (OR into a word of 4 low nibbles)
__device__ __forceinline__ unsigned kv_hi4(unsigned x, int k0, int hb) {
    const unsigned m = (1u << hb) - 1u;
    x >>= hb * k0;
    return ((x & m) | (((x >> hb) & m) << 8) | (((x >> (2 * hb)) & m) << 16) | (((x >> (3 * hb)) & m) << 24)) << 4;
}
// the four HB-bit fields k = 0 .. 3 of x (field k at bit HB k) -> bits 4 .. 4 + HB - 1 of byte k (the high nibbles of a
// word of four low nibbles, as kv_hi4 with a compile-time width)
template <int HB>
__device__ __forceinline__ unsigned kv_spread4(unsigned x) {
    if constexpr (HB == 1) return ((x & 0xfu) * 0x02040810u) & 0x10101010u; // bit k -> bit 8 k + 4 (no two products collide)
    if constexpr (HB == 2) {
        // two IMAD + two LOP3 (the shifted copies never overlap, so the multiply is the OR): x * 0x10010 puts fields
        // 0, 1 at bits 4 .. 7 and 2, 3 at 20 .. 23 (mask), then * 0x41 moves field 1 to 12 .. 13 and 3 to 28 .. 29
        x = ((x & 0xffu) * 0x10010u) & 0x00f000f0u;
        return (x * 0x41u) & 0x30303030u;
    }
    if constexpr (HB == 3) {
        x &= 0xfffu;
        x = (x | (x << 10)) & 0x003f003fu;
        x = (x | (x << 5)) & 0x07070707u;
        return x << 4;
    }
    return 0u;
}
// Staged q4 .. q7 K / V rows (kv_store_q4 / kv_store_qn's layout: lane word w = the low nibbles of values 8 w .. 8 w + 7, then
// at 128 + w (BITS - 4) their high bits) expanded IN PLACE to the q8 layout: 256 int8 values q = n - 2^(BITS - 1), the scales
// untouched. The fp16 product d q is the same number the low-bit decode computes (qn_to_f16 / a2_qnh), so every consumer of
// the q8 layout (aq_q8x2, a8_imma with the scale applied after the int8 dot) sees exactly the low-bit values.
// One warp per row, up to 3 rows (r0, r0 + rs, r0 + 2 rs, all < np) per call, loads of all rows first (one shared-memory
// latency instead of a load -> ALU -> store chain per row), then every lane has read its words before any lane writes (the
// output overlaps the input).
// SIGNED = false leaves the biased bytes n (0 .. 2^BITS - 1) in place: the fp16 consumers subtract the bias for free in
// their magic-number conversion (aq_q8x2 / aq_q8x2p with the tile's xm / sub); the int8 IMMA consumer (pk_attn8) needs q.
template <int BITS, bool SIGNED = true>
__device__ __forceinline__ void kv_rows_expand(uint8_t * st, int np, int r0, int rs, int ld) {
    constexpr int HB = BITS - 4;
    const int lane = threadIdx.x & 31;
    unsigned wd[3], hi[3];
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int r = r0 + k * rs;
        wd[k] = 0u;
        hi[k] = 0u;
        if (r < np) {
            const uint8_t * row = st + r * ld;
            wd[k] = *reinterpret_cast<const unsigned *>(row + 4 * lane);
            if constexpr (HB == 1) hi[k] = row[128 + lane];
            if constexpr (HB == 2) hi[k] = *reinterpret_cast<const unsigned short *>(row + 128 + 2 * lane);
            if constexpr (HB == 3) hi[k] = (unsigned) row[128 + 3 * lane] | ((unsigned) row[129 + 3 * lane] << 8) | ((unsigned) row[130 + 3 * lane] << 16);
        }
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 3; k++) {
        const int r = r0 + k * rs;
        if (r < np) {
            unsigned lo = wd[k] & 0x0f0f0f0fu, hv = (wd[k] >> 4) & 0x0f0f0f0fu; // values 0 .. 3 / 4 .. 7 (biased, n = q + 2^(BITS-1))
            if constexpr (HB > 0) {
                lo |= kv_spread4<HB>(hi[k]);
                hv |= kv_spread4<HB>(hi[k] >> (4 * HB));
            }
            if constexpr (SIGNED) {
                // per byte n - 2^(BITS-1) mod 256, without borrows between bytes (n < 128: bit 7 is free to absorb them)
                constexpr unsigned bias = (1u << (BITS - 1)) * 0x01010101u;
                lo = ((lo | 0x80808080u) - bias) ^ 0x80808080u;
                hv = ((hv | 0x80808080u) - bias) ^ 0x80808080u;
            }
            *reinterpret_cast<uint2 *>(st + r * ld + 8 * lane) = make_uint2(lo, hv);
        }
    }
}
// rows r0, r0 + rs, .. < np of a staged low-bit tile (row stride ld bytes) -> q8 layout (SIGNED: int8 q; else the biased
// bytes n, see kv_rows_expand); bits is block-uniform
template <bool SIGNED = true>
__device__ __forceinline__ void kv_tile_expand(uint8_t * st, int bits, int np, int r0, int rs, int ld) {
    switch (bits) {
    case 4:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<4, SIGNED>(st, np, r, rs, ld);
        break;
    case 5:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<5, SIGNED>(st, np, r, rs, ld);
        break;
    case 6:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<6, SIGNED>(st, np, r, rs, ld);
        break;
    case 7:
        for (int r = r0; r < np; r += 3 * rs) kv_rows_expand<7, SIGNED>(st, np, r, rs, ld);
        break;
    default:
        break;
    }
}

// 8 q4 .. q7 values (lane word w in kv_store_q4's nibble order, high bits hi) + scale -> 4 fp16 pairs (q4: q4_to_f16)
__device__ __forceinline__ void qn_to_f16(unsigned w, unsigned hi, float d, int bits, __half2 (&h)[4]) {
    const int hb = bits - 4, mid = 1 << (bits - 1);
    const unsigned m = (1u << hb) - 1u;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int j0 = 2 * k, j1 = 2 * k + 1;
        const int a = (int) (((w >> (j0 < 4 ? 8 * j0 : 8 * (j0 - 4) + 4)) & 0xfu) | (((hi >> (hb * j0)) & m) << 4)) - mid;
        const int b = (int) (((w >> (j1 < 4 ? 8 * j1 : 8 * (j1 - 4) + 4)) & 0xfu) | (((hi >> (hb * j1)) & m) << 4)) - mid;
        h[k] = __floats2half2_rn(d * (float) a, d * (float) b);
    }
}

// staged q8 / q4..q7 K/V rows (KVQ_ROW bytes: values, then scales at 256) -> fp16 rows of stride ld; rows x 2 tensors
__device__ __forceinline__ void kv_tile_to_f16(const uint8_t * src, __half * dK, __half * dV, int rows, int ld, int kbits, int vbits) {
    for (int i = threadIdx.x; i < 2 * rows * 32; i += blockDim.x) {
        const int which = i / (rows * 32), r = (i >> 5) % rows, c = i & 31;
        const uint8_t * row = src + (size_t) (which * rows + r) * KVQ_ROW;
        const float d = __half2float(reinterpret_cast<const __half *>(row + 256)[c >> 2]);
        const int bits = which ? vbits : kbits;
        __half2 h[4];
        if (bits == 4) q4_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), d, h);
        else if (bits != 8) qn_to_f16(*reinterpret_cast<const unsigned *>(row + 4 * c), kv_hi(row + 128 + c * (bits - 4), bits - 4), d, bits, h);
        else q8_to_f16(*reinterpret_cast<const uint2 *>(row + 8 * c), d, h);
        *reinterpret_cast<int4 *>((which ? dV : dK) + (size_t) r * ld + 8 * c) = *reinterpret_cast<int4 *>(h);
    }
}

// MK_KV_WIN after a checkpoint restore to position p1: the ring rows of positions p0 .. p1 - 1 may have been overwritten by
// the abandoned continuation, so they are refilled from the main buffer (its low-bit values, re-stored as q8: the window
// is at the main format's precision until those rows age out). One warp per row.
__global__ void __launch_bounds__(256) k_ring_refill(int p0, int p1) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int n = p1 - p0;
    for (int i = blockIdx.x * 8 + warp; i < n * NKV * N_ATT_SLOTS * 2; i += gridDim.x * 8) {
        const int w = i & 1, slot = (i >> 1) % N_ATT_SLOTS, g = ((i >> 1) / N_ATT_SLOTS) % NKV;
        const int p = p0 + (i >> 1) / (N_ATT_SLOTS * NKV);
        if (!cB.kvrp[w][slot]) continue;
        const int bits = cB.kvb[w][slot];
        const size_t row = (size_t) g * cB.ctx + p;
        const uint8_t * rp = reinterpret_cast<const uint8_t *>(cB.kvp[w][slot] + row * (HD * bits / 8));
        const float d = __half2float(cB.kvs[w][slot][row * (HD / 32) + (lane >> 2)]);
        float v[8];
        if (bits == 8) {
            const uint2 qq = *reinterpret_cast<const uint2 *>(rp + lane * 8);
#pragma unroll
            for (int k = 0; k < 8; k++) v[k] = d * (float) (int8_t) ((k < 4 ? qq.x : qq.y) >> (8 * (k & 3)));
        } else { // q4..q7: lane word (values 0..3 low nibbles, 4..7 high) + bits - 4 high bits per value (kv_store_qn)
            const unsigned wd = *reinterpret_cast<const unsigned *>(rp + lane * 4);
            const int hb = bits - 4;
            const unsigned hi = hb > 0 ? kv_hi(rp + 128 + lane * hb, hb) : 0u;
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const int nb = (int) ((wd >> (k < 4 ? 8 * k : 8 * (k - 4) + 4)) & 0xfu) | (int) (((hi >> (hb * k)) & ((1u << hb) - 1u)) << 4);
                v[k] = d * (float) (nb - (1 << (bits - 1)));
            }
        }
        __syncwarp();
        const size_t rr = (size_t) g * cB.kvrc + p % cB.kvrc;
        kv_store_q8(cB.kvrp[w][slot] + rr * HD, cB.kvrs[w][slot] + rr * (HD / 32), v);
    }
}

// async copy of q8 / q4..q7 K/V rows r < np of one tile (Kq / Ks, Vq / Vs: the tile's first row, kv_src) into the staging
// layout (rows >= np: zero scales)
__device__ __forceinline__ void kv_stage(uint8_t * dst, const int8_t * Kq, const int8_t * Vq, const __half * Ks, const __half * Vs,
                                         int rows, int np, int kbits, int vbits) {
    for (int i = threadIdx.x; i < 2 * rows * 17; i += blockDim.x) {
        const int which = i / (rows * 17), r = (i / 17) % rows, c = i % 17;
        uint8_t * d = dst + (size_t) (which * rows + r) * KVQ_ROW + 16 * c;
        if (r < np) {
            const int bits = which ? vbits : kbits; // a row of HD bits / 8 bytes = 2 bits chunks of 16
            if (c < 16) {
                if (c < 2 * bits) cp_async16(d, (which ? Vq : Kq) + (size_t) r * (HD * bits / 8) + 16 * c);
            } else {
                cp_async16(d, (which ? Vs : Ks) + (size_t) r * (HD / 32));
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
// MK_MTP_WIN=N (measurement knob, default off): the MTP slot's decode attention covers the first tile of positions
// (sinks, StreamingLLM) and the last N ones; ws = the window start (0: everything, also when the window would start
// inside the first two tiles). The chunk partition is attn_dims over [ws, L_ctx) plus one sink chunk [0, ATP).
__device__ __forceinline__ int mtp_ws(int slot, int n) {
    if (slot != N_ATT_SLOTS - 1 || cB.mtp_win <= 0) return 0;
    const int ws = n + 1 - cB.mtp_win;
    return ws > 2 * ATP ? ws : 0;
}
__device__ __forceinline__ void attn_dims_w(int L_ctx, int ws, int & nch, int & cs) {
    attn_dims(L_ctx - ws, nch, cs);
    if (ws > 0) nch++;
}
__device__ __forceinline__ void attn_chunk(int c, int ws, int cs, int L_ctx, int & c0, int & c1) {
    if (ws > 0 && c == 0) {
        c0 = 0;
        c1 = ATP;
        return;
    }
    c0 = ws + (c - (ws > 0 ? 1 : 0)) * cs;
    c1 = min(L_ctx, c0 + cs);
}

// attention q/k/v projections
template <int T>
__device__ __noinline__ void attn_proj(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    uint8_t * const sm = smem;
    const int lane = threadIdx.x & 31;
    (void) lane;
    const SX sx = make_sx(sm, U_D);
    fill_xs<T>(cB.qn, 0, true, sx, misc(sm));
    const int nq = QG / 2, nk = KVD / 2;
#if MK_WV
    gemv_pairs_any<T, 5>(W.q, 0, nq, PairOut{cB.qbuf, QG, -1});
    gemv_pairs_any<T, 5>(W.k, nq, nq + nk, PairOut{cB.kbuf, KVD, -1});
    gemv_pairs_any<T, 5>(W.v, nq + nk, nq + 2 * nk, PairOut{cB.vbuf, KVD, -1});
#else
    gemv_pairs_any<T, 5>(W.q, 0, nq, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.qbuf, QG, r, a); });
    gemv_pairs_any<T, 5>(W.k, nq, nq + nk, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.kbuf, KVD, r, a); });
    gemv_pairs_any<T, 5>(W.v, nq + nk, nq + 2 * nk, [=](int r, const float (&a)[2][T]) { store_pair_t<T>(cB.vbuf, KVD, r, a); });
#endif
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
    const int ws = mtp_ws(slot, n); // MK_MTP_WIN (ws <= n: this cycle's rows are always inside the window)
    int nch, CS;
    attn_dims_w(L_ctx, ws, nch, CS);
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
    const int wsr = Q8 ? kv_ws(L_ctx) : 0; // MK_KV_WIN: tiles at positions >= wsr come from the q8 ring (kv_src)
    const int wsm = Q8 ? kv_wsm(wsr) : 0;
    const int mt = warp >> 3, wn = warp & 7; // MMA work split: warp -> (row tile, column group)

    auto issue_tile = [&](int g, int tb, int np_, int st) {
        if constexpr (Q8) {
            const int8_t *tk, *tv;
            const __half *tks, *tvs;
            const int kb_ = kv_src(0, slot, g, tb, wsr, wsm, tk, tks), vb_ = kv_src(1, slot, g, tb, wsr, wsm, tv, tvs);
            kv_stage(sKVq + st * 2 * TP * KVQ_ROW, tk, tv, tks, tvs, TP, np_, kb_, vb_);
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
        int c0, c1;
        attn_chunk(c, ws, CS, L_ctx, c0, c1);
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
            if (Q8 && cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot], pp);
            else
#endif
            if constexpr (Q8) kv_store(0, slot, g, pp, v);
            else kv_store_f16(static_cast<__half *>(cB.kc) + row * HD, v);
            load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
            if (Q8 && cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot], pp);
            else
#endif
            if constexpr (Q8) kv_store(1, slot, g, pp, v);
            else kv_store_f16(static_cast<__half *>(cB.vc) + row * HD, v);
            (void) srow;
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
            if constexpr (Q8) { // q8 stage -> the fp16 tile (its formats: ring q8 or the slot's, per tile)
                kv_tile_to_f16(sKVq + st * 2 * TP * KVQ_ROW, sKV, sKV + TP * LDH, TP, LDH, kv_tbits(0, slot, tb, wsr), kv_tbits(1, slot, tb, wsr));
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

// ---------------------------------------------------------------------------------------------------------------------
// Decode attention on the all-q8 cache: attn_core_t's per-row arithmetic (bit-identical partials, so also T-invariant),
// restructured for long contexts (one 32-position tile per SM every ~4.8 us before: 5 dependent stages and 5 block
// barriers per tile, ~290 GB/s of q8 KV at 101k; this kernel: ~600 GB/s for 3-5 query tokens, ~740 GB/s = the DRAM
// stream rate for 1-2). Microbenchmark: bench/dattn_bench.cu.
// - Three-deep software pipeline with ONE block barrier per tile: interval j computes S(j) = Q K(j)^T, the online softmax
//   of tile j - 1 and O = O * alpha + P(j - 2) V(j - 2) on specialised warps, and issues the next copies:
//     warps 0-3   S for positions 8 w .. 8 w + 7, all row tiles (one K fragment conversion feeds the MT chains)
//     warps 4-11  P V for dims 32 (w - 4) .. + 31, all row tiles (one V fragment conversion feeds the MT tiles)
//     warps 12-15 softmax, a quad of lanes per query row (8 positions per lane)
// - q8 K / V tiles stream through separate cp.async rings (K(t) issued in interval t - 2 and read in t: 3 stages; V(t)
//   issued in t - 1 and read in t + 2: 4 stages). No fp16 K / V tiles: both MMA B operands are converted from the q8
//   rows in registers with q8_to_f16's rounding (d * q, exact in fp32, rounded once to fp16; here one fp16 multiply).
// - P V B fragments: lane (g8, tig) of column group wn holds dims 32 wn + 4 g8 + jj of n8 tile jj, so one 32-bit q8 load
//   per position feeds 4 tiles. Each output element is the same MMA dot product (same operand values, same k order),
//   only its column slot in the MMA differs.
// - The softmax keeps attn_core_t's float operations and its butterfly sum tree (position bit 4 first, then 3 .. 0).
// q4 MK_KV slots run the MIX instantiation (attn_core_q8m: q4 K / V fragments with q4_to_f16's values); chunks of a
// single tile (L_ctx <= ATT_BUCKET) and MK_DATTN=0 run attn_core_t (same partials).

// d * q rounded once to fp16 for the two int8 values in bytes 0 and 1 of u: 0x6400 | (q ^ 0x80) is the fp16 1152 + q,
// minus 1152 is q exactly, and one fp16 multiply rounds the exact product (= q8_to_f16's value).
// xm / sub generalise it to a tile of biased bytes n = q + 2^(b-1) (MIX, kv_rows_expand<_, false>): xm = 0, sub = 1024 +
// 2^(b-1) (aq_sub) gives n - 2^(b-1) = q, the very same fp16 q; the all-q8 path's constants fold at compile time.
__device__ __forceinline__ unsigned aq_q8x2(unsigned u, __half2 d, unsigned xm = 0x8080u, unsigned sub = 0x64806480u) {
    const unsigned h = __byte_perm(u ^ xm, 0x64646464u, 0x4140);
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&h), *reinterpret_cast<const __half2 *>(&sub));
    const __half2 r = __hmul2(q, d);
    return *reinterpret_cast<const unsigned *>(&r);
}
// the same for bytes 0 / 2 or 1 / 3 of y (selector 0x4140 / 0x4342 after the caller's interleave; the caller applied xm)
__device__ __forceinline__ unsigned aq_q8x2p(unsigned y, unsigned sel, __half2 d, unsigned sub = 0x64806480u) {
    const unsigned h = __byte_perm(y, 0x64646464u, sel);
    const __half2 q = __hsub2(*reinterpret_cast<const __half2 *>(&h), *reinterpret_cast<const __half2 *>(&sub));
    const __half2 r = __hmul2(q, d);
    return *reinterpret_cast<const unsigned *>(&r);
}
// the magic-number subtrahend (as a half2 bit pattern) for a tile of bits b: q8 tiles hold q ^ 0x80 after xm (1152),
// expanded low-bit tiles hold n = q + 2^(b-1) (1024 + 2^(b-1): 0x6400 | 2^(b-1), exact below 2048)
__device__ __forceinline__ unsigned aq_sub(int bits) {
    const unsigned s = bits == 8 ? 0x6480u : 0x6400u | (1u << (bits - 1));
    return s * 0x10001u;
}
// one commit group's copies, issued by NTH threads (this one is thread t of them): K rows of a tile (kn valid rows,
// kn < 0: none) -> K stage dk, V rows of a tile -> dv; rows >= the valid count get zero scales (their values are
// never used: masked S, P = 0). MIX: q4..q7 rows (kbits / vbits) copy 2 bits value chunks (kv_stage's staging layout).
// Kq / Ks and Vq / Vs: the tiles' first rows (kv_src: the slot's main buffer, or the MK_KV_WIN q8 ring).
template <int NTH, bool MIX = false>
__device__ __forceinline__ void aq_issue(int t, uint8_t * dk, const int8_t * Kq, const __half * Ks, int kn, uint8_t * dv, const int8_t * Vq,
                                         const __half * Vs, int vn, int kbits = 8, int vbits = 8) {
#pragma unroll
    for (int k = 0; k < (2 * ATP * 17 + NTH - 1) / NTH; k++) {
        const int i = t + NTH * k;
        if (i < 2 * ATP * 17) {
            const int which = i >= ATP * 17, j = which ? i - ATP * 17 : i, r = j / 17, c = j - 17 * r;
            const int np = which ? vn : kn;
            if (np < 0) continue;
            uint8_t * d = (which ? dv : dk) + r * KVQ_ROW + 16 * c;
            if (r < np) {
                const int bits = MIX ? (which ? vbits : kbits) : 8;
                if (c < 16) {
                    if (c < 2 * bits) cp_async16(d, (which ? Vq : Kq) + (size_t) r * (HD * bits / 8) + 16 * c);
                } else cp_async16(d, (which ? Vs : Ks) + (size_t) r * (HD / 32));
            } else if (c == 16) {
                *reinterpret_cast<int4 *>(d) = make_int4(0, 0, 0, 0);
            }
        }
    }
}

// MIX = false: all-q8 slots (attn_core_q8). MIX = true (attn_core_q8m): K and V of the slot each q8 or q4 .. q7 (MK_KV), the
// low-bit tiles staged in their packed layout and expanded to int8 once in shared memory (kv_row_expand: q = n - 2^(bits-1)
// with the scale unchanged, by warps 4 - 15 one interval before the tile is read; the K pipeline is one stage deeper for
// it), so the MMA path is the q8 one for every tile and the partials equal attn_core_t's for every MK_KV format.
// T >= 6 (MK_MAXT > 5; all-q8 slots, MIX = false): 6 T > 32 query rows do not fit shared memory, so the items run twice,
// over query heads 0-2 and 3-5 of each kv head (HG = 3, partial rows ho T ..); the second pass re-reads K / V and skips the
// K / V store (the first pass of the same item, in the same block, wrote those rows). Per-row arithmetic is unchanged.
template <int T, bool MIX = false>
__device__ __forceinline__ void attn_core_q8_t(const LayerW & W, int slot, int n) {
    uint8_t * const sm = smem;
    constexpr int HG = T > 5 ? 3 : 6; // query heads per pass
    constexpr int R = HG * T;
    constexpr int MT = (R + 15) / 16; // 16-row MMA tiles of query rows
    constexpr int TP = ATP;
    constexpr int LDH = HD + 8;
    constexpr int LDP = TP + 8;
    constexpr int LDS = AQ_LDS;
    constexpr int NSK = MIX ? AQ_NSKM : AQ_NSK, NSV = AQ_NSV;
    static_assert(R <= 32 && TP == 32 && MT <= 2, "8 rows per softmax warp, 32-position tiles");
    const int L_ctx = n + T;
    const int ws = mtp_ws(slot, n); // MK_MTP_WIN
    int nch, CS;
    attn_dims_w(L_ctx, ws, nch, CS);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g8 = lane >> 2, tig = lane & 3;
    __half * sQ = reinterpret_cast<__half *>(sm);                 // [32][LDH]
    uint8_t * kring = reinterpret_cast<uint8_t *>(sQ + 32 * LDH); // NSK stages
    uint8_t * vring = kring + NSK * AQ_STG;                       // NSV stages
    float * sS = reinterpret_cast<float *>(vring + NSV * AQ_STG); // [2][32][LDS]; pad columns: m (32), l (33), alpha (34)
    __half * sP = reinterpret_cast<__half *>(sS + 2 * 32 * LDS);  // [2][32][LDP]
    auto sM = [&](int r) -> float & { return sS[r * LDS + 32]; };
    auto sL = [&](int r) -> float & { return sS[r * LDS + 33]; };
    auto sAl = [&](int b, int r) -> float & { return sS[(b * 32 + r) * LDS + 34]; };
    const size_t ctx = cB.ctx;
    const size_t slot0 = (size_t) slot * NKV * ctx; // first row of this slot
    (void) slot0;
    // MK_KV per-slot formats (MIX only): tiles are copied in the slot's format (per tile: kv_tbits, since with MK_KV_WIN the
    // tiles at positions >= wsr are the ring's q8 copies) and the low-bit ones expanded to int8 in their stage before use
    const int wsr = MIX ? kv_ws(L_ctx) : 0, wsm = MIX ? kv_wsm(wsr) : 0;

    for (int ho = 0; ho < 6; ho += HG) // (T <= 5: one pass, ho = 0)
    for (int item = blockIdx.x; item < NKV * nch; item += gridDim.x) {
        const int g = item / nch, c = item % nch;
        int c0, c1;
        attn_chunk(c, ws, CS, L_ctx, c0, c1);
        // ---- item prologue, attn_core_t's (q rows -> sQ, this cycle's K / V rows -> cache, m / l init)
        float qv[2][8], qw[8];
        float2 rc[2][8];
        load8(W.qn + lane * 8, qw);
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = min(warp + 16 * k2, R - 1);
            const int hq = r / T, t = r % T, head = g * 6 + ho + hq;
            load8(cB.qbuf + t * QG + head * 512 + lane * 8, qv[k2]);
            const float2 * tab = cB.rope + (size_t) (n + t) * 32 + (lane & 3) * 8;
#pragma unroll
            for (int k = 0; k < 8; k++) rc[k2][k] = __ldg(tab + k);
        }
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const int r = warp + 16 * k2;
            if (r < R) {
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
                if (cB.kvr[slot]) fwht256(v); // K of this slot is stored rotated (MK_KV "r")
                __half2 hq2[4];
#pragma unroll
                for (int k = 0; k < 4; k++) hq2[k] = __floats2half2_rn(v[2 * k] * 0.0625f, v[2 * k + 1] * 0.0625f);
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = *reinterpret_cast<int4 *>(hq2);
            } else {
                *reinterpret_cast<int4 *>(sQ + r * LDH + lane * 8) = make_int4(0, 0, 0, 0);
                if (2 * lane < TP) { // padded rows of both P buffers stay 0
                    *reinterpret_cast<unsigned *>(sP + r * LDP + 2 * lane) = 0u;
                    *reinterpret_cast<unsigned *>(sP + 32 * LDP + r * LDP + 2 * lane) = 0u;
                }
            }
        }
        const int pn0 = max(c0, n), pn1 = ho ? pn0 : min(c1, n + T); // (the first head-group pass stores)
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
            (void) row;
#if MK_KV_EMU
            if (cB.kvemu[0][slot]) kv_store_emu(static_cast<int8_t *>(cB.kc) + row * HD, cB.ksc + row * (HD / 32), v, cB.kvemu[0][slot], pp);
            else
#endif
            kv_store(0, slot, g, pp, v);
            load8(cB.vbuf + t * KVD + g * 256 + lane * 8, v);
#if MK_KV_EMU
            if (cB.kvemu[1][slot]) kv_store_emu(static_cast<int8_t *>(cB.vc) + row * HD, cB.vsc + row * (HD / 32), v, cB.kvemu[1][slot], pp);
            else
#endif
            kv_store(1, slot, g, pp, v);
            (void) srow;
        }
        if (tid < 32) {
            sM(tid) = -INFINITY;
            sL(tid) = 0.f;
        }
        __threadfence_block();
        __syncthreads();

        // ---- ring prologue: commit groups {K(0)}, {K(1), V(0)}; then per interval jt {K(jt + 2), V(jt + 1)}
        // (MIX: {K(0), K(1)}, {K(2), V(0)}; per interval {K(jt + 3), V(jt + 1)}, and the low-bit tiles that landed at the
        // end of interval jt - 1, K(jt + 1) and V(jt - 1), are expanded to int8 in interval jt)
        const int ntl = (c1 - c0 + TP - 1) / TP;
        auto tnp = [&](int t) { return t < ntl ? min(TP, c1 - c0 - t * TP) : -1; }; // valid rows of tile t, -1: none
        // tile sources (MK_KV_WIN: the q8 ring for tiles at positions >= wsr, else the slot's main buffer) and formats
        auto issue = [&](int t_, uint8_t * dk, int tk, uint8_t * dv, int tv, int nth_tid) {
            const int8_t *kq, *vq;
            const __half *ks, *vs;
            int kb_ = 8, vb_ = 8;
            if constexpr (MIX) {
                kb_ = kv_src(0, slot, g, c0 + tk * TP, wsr, wsm, kq, ks);
                vb_ = kv_src(1, slot, g, c0 + max(tv, 0) * TP, wsr, wsm, vq, vs);
            } else { // all-q8 slot: the plain layout
                const size_t kr = (size_t) g * ctx + c0 + tk * TP, vr = (size_t) g * ctx + c0 + max(tv, 0) * TP;
                kq = cB.kvp[0][slot] + kr * HD;
                ks = cB.kvs[0][slot] + kr * (HD / 32);
                vq = cB.kvp[1][slot] + vr * HD;
                vs = cB.kvs[1][slot] + vr * (HD / 32);
            }
            if (nth_tid == NT) aq_issue<NT, MIX>(t_, dk, kq, ks, tnp(tk), dv, vq, vs, tv < 0 ? -1 : tnp(tv), kb_, vb_);
            else aq_issue<NT - 128, MIX>(t_, dk, kq, ks, tnp(tk), dv, vq, vs, tv < 0 ? -1 : tnp(tv), kb_, vb_);
        };
        if constexpr (MIX) {
            issue(tid, kring, 0, vring, -1, NT);
            issue(tid, kring + AQ_STG, 1, vring, -1, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            issue(tid, kring + 2 * AQ_STG, 2, vring, 0, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
            const int kb0 = kv_tbits(0, slot, c0, wsr); // K(0) -> bytes now (all warps); K(1) in interval 0
            if (kb0 != 8) {
                kv_tile_expand<false>(kring, kb0, TP, warp, NW, KVQ_ROW);
                __syncthreads();
            }
        } else {
            issue(tid, kring, 0, vring, -1, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            issue(tid, kring + AQ_STG, 1, vring, 0, NT);
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
        }

        float acc[MT][4][4]; // P V warps: O rows 16 i + {g8, g8 + 8}, dims 32 wn + 8 tig + jj and + 4
#pragma unroll
        for (int i = 0; i < MT; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
#pragma unroll
                for (int k = 0; k < 4; k++) acc[i][j][k] = 0.f;

        for (int jt = 0; jt < ntl + 2; jt++) {
            // MIX: the copies of K(jt + 3) (into the stage of K(jt - 1)) and V(jt + 1) (into the stage of V(jt - 3), both read
            // in interval jt - 1) go out FIRST, by warps 4-15: a tile then has two whole intervals to land instead of one
            // (the all-q8 kernel issues after its MMA work; its intervals are DRAM-bound, so the lead time does not show)
            if constexpr (MIX && MK_AQ_EARLY) {
                if (warp >= 4 && jt + 1 < ntl) issue(tid - 128, kring + ((jt + 3) % NSK) * AQ_STG, jt + 3, vring + ((jt + 1) % NSV) * AQ_STG, jt + 1, NT - 128);
            }
            if (warp < 4) {
                // ---- S(jt) = Q K(jt)^T: B = positions 8 wn + g8 (all 16 k-steps converted first), all row tiles
                if (jt < ntl) {
                    const int wn = warp;
                    const uint8_t * krow = kring + (jt % NSK) * AQ_STG + (8 * wn + g8) * KVQ_ROW;
                    const uint4 scv = *reinterpret_cast<const uint4 *>(krow + 256);
                    const unsigned scw[4] = {scv.x, scv.y, scv.z, scv.w};
                    // MIX: a low-bit tile was expanded to biased bytes n = q + 2^(b-1) in interval jt - 1; its xm / sub make
                    // the conversion below yield the same fp16 q (aq_q8x2). All-q8: the constants fold.
                    const int kbj = MIX ? kv_tbits(0, slot, c0 + jt * TP, wsr) : 8;
                    const unsigned kxm = kbj == 8 ? 0x8080u : 0u, ksub = aq_sub(kbj);
                    unsigned kb[HD / 16][2];
#pragma unroll
                    for (int ks = 0; ks < HD / 16; ks++) {
                        const unsigned sw = __byte_perm(scw[ks >> 2], 0, (ks & 2) ? 0x3232 : 0x1010);
                        const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                        kb[ks][0] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig), dd, kxm, ksub);
                        kb[ks][1] = aq_q8x2(*reinterpret_cast<const unsigned short *>(krow + 16 * ks + 2 * tig + 8), dd, kxm, ksub);
                    }
                    float s[MT][4];
#pragma unroll
                    for (int i = 0; i < MT; i++)
#pragma unroll
                        for (int k = 0; k < 4; k++) s[i][k] = 0.f;
                    const unsigned qaddr = (unsigned) __cvta_generic_to_shared(sQ + (lane & 15) * LDH + (lane >> 4) * 8);
#pragma unroll
                    for (int ks = 0; ks < HD / 16; ks++) {
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            unsigned a[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                         : "r"(qaddr + 2 * (16 * i * LDH + 16 * ks)));
                            mma16816(s[i], a[0], a[1], a[2], a[3], kb[ks][0], kb[ks][1]);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < MT; i++) {
                        float * so = sS + (jt & 1) * 32 * LDS + (16 * i + g8) * LDS + 8 * wn + 2 * tig;
                        so[0] = s[i][0];
                        so[1] = s[i][1];
                        so[8 * LDS] = s[i][2];
                        so[8 * LDS + 1] = s[i][3];
                    }
                }
            } else if (warp < 12) {
                // ---- O = O * alpha + P(jt - 2) V(jt - 2): B = dims 32 wn + 4 g8 + jj of n8 tile jj, all row tiles
                if (jt >= 2) {
                    const int wn = warp - 4, jp = jt - 2, b = jp & 1;
#pragma unroll
                    for (int i = 0; i < MT; i++) {
                        const float al0 = sAl(b, 16 * i + g8), al1 = sAl(b, 16 * i + g8 + 8);
#pragma unroll
                        for (int j = 0; j < 4; j++) {
                            acc[i][j][0] *= al0;
                            acc[i][j][1] *= al0;
                            acc[i][j][2] *= al1;
                            acc[i][j][3] *= al1;
                        }
                    }
                    const unsigned paddr = (unsigned) __cvta_generic_to_shared(sP + b * 32 * LDP + (lane & 15) * LDP + (lane >> 4) * 8);
                    // (MIX: a low-bit V tile was expanded to biased bytes in interval jt - 1: xm / sub as for K)
                    const uint8_t * vst = vring + (jp % NSV) * AQ_STG + 32 * wn + 4 * g8;
                    const uint8_t * vsc = vring + (jp % NSV) * AQ_STG + 256 + 2 * wn;
                    const int vbj = MIX ? kv_tbits(1, slot, c0 + jp * TP, wsr) : 8;
                    const unsigned vxm = vbj == 8 ? 0x80808080u : 0u, vsub = aq_sub(vbj);
#pragma unroll
                    for (int ks = 0; ks < TP / 16; ks++) {
                        unsigned bf[2][4]; // [b0 / b1][tile jj]
#pragma unroll
                        for (int h = 0; h < 2; h++) {
                            const int p = 16 * ks + 2 * tig + 8 * h; // k rows p, p + 1 of the fragment
                            const unsigned xa = *reinterpret_cast<const unsigned *>(vst + p * KVQ_ROW) ^ vxm;
                            const unsigned xb = *reinterpret_cast<const unsigned *>(vst + (p + 1) * KVQ_ROW) ^ vxm;
                            const unsigned sw = (unsigned) *reinterpret_cast<const unsigned short *>(vsc + p * KVQ_ROW) |
                                                ((unsigned) *reinterpret_cast<const unsigned short *>(vsc + (p + 1) * KVQ_ROW) << 16);
                            const __half2 dd = *reinterpret_cast<const __half2 *>(&sw);
                            const unsigned y0 = __byte_perm(xa, xb, 0x5140), y1 = __byte_perm(xa, xb, 0x7362);
                            bf[h][0] = aq_q8x2p(y0, 0x4140, dd, vsub);
                            bf[h][1] = aq_q8x2p(y0, 0x4342, dd, vsub);
                            bf[h][2] = aq_q8x2p(y1, 0x4140, dd, vsub);
                            bf[h][3] = aq_q8x2p(y1, 0x4342, dd, vsub);
                        }
#pragma unroll
                        for (int i = 0; i < MT; i++) {
                            unsigned a[4];
                            asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                                         : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
                                         : "r"(paddr + 2 * (16 * i * LDP + 16 * ks)));
#pragma unroll
                            for (int j = 0; j < 4; j++) mma16816(acc[i][j], a[0], a[1], a[2], a[3], bf[0][j], bf[1][j]);
                        }
                    }
                }
            } else {
                // ---- online softmax of tile jt - 1 (causal mask: row token t sees positions <= n + t); lane l4 of the
                // row's quad holds positions l4 + 4 k
                if (jt >= 1 && jt <= ntl && 8 * (warp - 12) < R) {
                    const int r = 8 * (warp - 12) + (lane >> 2), l4 = lane & 3; // rows >= R (zero S) are not stored
                    const int jp = jt - 1, b = jp & 1;
                    const int tb = c0 + jp * TP, np_ = min(TP, c1 - tb);
                    const int lim = n + (r % T);
                    const float * srow = sS + b * 32 * LDS + r * LDS;
                    float sv[8];
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        const int q = l4 + 4 * k;
                        sv[k] = srow[q];
                        if (q >= np_ || tb + q > lim) sv[k] = -INFINITY;
                    }
                    float m = sv[0];
#pragma unroll
                    for (int k = 1; k < 8; k++) m = fmaxf(m, sv[k]);
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 2));
                    m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, 1));
                    const float mold = sM(r);
                    const float mnew = fmaxf(mold, m);
                    float p[8], al = 1.f;
#pragma unroll
                    for (int k = 0; k < 8; k++) p[k] = 0.f;
                    if (mnew != -INFINITY) {
#pragma unroll
                        for (int k = 0; k < 8; k++) p[k] = expf(sv[k] - mnew);
                        al = expf(mold - mnew);
                    }
                    // attn_core_t's wsum tree: position bit 4 (k, k + 4), bit 3 (k, k + 2), bit 2 (k, k + 1), then lanes
                    const float t0 = p[0] + p[4], t1 = p[1] + p[5], t2 = p[2] + p[6], t3 = p[3] + p[7];
                    const float u0 = t0 + t2, u1 = t1 + t3;
                    float ps = u0 + u1;
                    ps += __shfl_xor_sync(0xffffffffu, ps, 2);
                    ps += __shfl_xor_sync(0xffffffffu, ps, 1);
                    if (r < R) {
                        __half * prow = sP + b * 32 * LDP + r * LDP;
#pragma unroll
                        for (int k = 0; k < 8; k++) prow[l4 + 4 * k] = __float2half_rn(p[k]);
                        if (l4 == 0) {
                            sL(r) = sL(r) * al + ps;
                            sM(r) = mnew;
                            sAl(b, r) = al;
                        }
                    }
                }
            }
            // ---- copies: K(jt + 2) into the stage of K(jt - 1), V(jt + 1) into the stage of V(jt - 3) (both read in
            // interval jt - 1), issued by warps 4-15 (the S warps carry the longest MMA chains: -5% per tile at T >= 3);
            // then wait for K(jt + 1) and V(jt)
            if (warp >= 4) {
                if constexpr (MIX) {
                    // K(jt + 3) into the stage of K(jt - 1) (unless issued at the top of the interval, MK_AQ_EARLY); then the
                    // low-bit tiles that landed at the end of interval jt - 1 are expanded to bytes for interval jt + 1: K(jt + 1)
                    // (S warps) and V(jt - 1) (P V warps), up to 3 rows per warp per call; all TP rows (rows >= np hold stale
                    // but finite bytes under zero scales: masked S, P = 0). Their stages are not touched by any copy in flight.
                    if constexpr (!MK_AQ_EARLY) {
                        if (jt + 1 < ntl) issue(tid - 128, kring + ((jt + 3) % NSK) * AQ_STG, jt + 3, vring + ((jt + 1) % NSV) * AQ_STG, jt + 1, NT - 128);
                    }
                    // row split of each tile: MK_XP_SM = 0: rows round-robin over the 12 warps; else rows 0 .. MK_XP_SM - 1 go to
                    // the (lightly loaded) softmax warps 12-15, the rest to the P V warps 4-11
                    int r0, rs, r1;
                    if constexpr (MK_XP_SM == 0) { r0 = warp - 4; rs = NW - 4; r1 = TP; }
                    else if (warp >= 12) { r0 = warp - 12; rs = 4; r1 = MK_XP_SM; }
                    else { r0 = MK_XP_SM + warp - 4; rs = 8; r1 = TP; }
                    if (jt + 1 < ntl) {
                        const int kb_ = kv_tbits(0, slot, c0 + (jt + 1) * TP, wsr);
                        if (kb_ != 8) kv_tile_expand<false>(kring + ((jt + 1) % NSK) * AQ_STG, kb_, r1, r0, rs, KVQ_ROW);
                    }
                    if (jt >= 1 && jt - 1 < ntl) {
                        const int vb_ = kv_tbits(1, slot, c0 + (jt - 1) * TP, wsr);
                        if (vb_ != 8) kv_tile_expand<false>(vring + ((jt - 1) % NSV) * AQ_STG, vb_, r1, r0, rs, KVQ_ROW);
                    }
                } else {
                    if (jt + 1 < ntl) issue(tid - 128, kring + ((jt + 2) % NSK) * AQ_STG, jt + 2, vring + ((jt + 1) % NSV) * AQ_STG, jt + 1, NT - 128);
                }
            }
            asm volatile("cp.async.commit_group;" ::: "memory");
            asm volatile("cp.async.wait_group 1;" ::: "memory");
            __syncthreads();
        }
        if (warp >= 4 && warp < 12) {
            const int wn = warp - 4;
#pragma unroll
            for (int i = 0; i < MT; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int r0 = 16 * i + g8, d0 = 32 * wn + 8 * tig + j; // columns 2 tig, 2 tig + 1: dims d0, d0 + 4
                    if (r0 < R) {
                        float * o = cB.apart + ((size_t) item * AR + ho * T + r0) * 256 + d0;
                        o[0] = acc[i][j][0];
                        o[4] = acc[i][j][1];
                    }
                    if (r0 + 8 < R) {
                        float * o = cB.apart + ((size_t) item * AR + ho * T + r0 + 8) * 256 + d0;
                        o[0] = acc[i][j][2];
                        o[4] = acc[i][j][3];
                    }
                }
        }
        if (tid < R) cB.aml[item * AR + ho * T + tid] = make_float2(sM(tid), sL(tid));
        __syncthreads();
    }
}

template <int T>
__device__ __noinline__ void attn_core_q8(const LayerW & W, int slot, int n) {
    attn_core_q8_t<T>(W, slot, n);
}

#if MK_DA_Q4
template <int T>
__device__ __noinline__ void attn_core_q8m(const LayerW & W, int slot, int n) {
    attn_core_q8_t<T, true>(W, slot, n);
}
#endif

// q8 cache: the pipelined kernel when a chunk has more than one tile (L_ctx > ATT_BUCKET; MK_DA_ROUTE), attn_core_t for
// one-tile chunks (the pipeline fill does not pay there) and with MK_DATTN=0. Every choice gives the same partials.
template <int T>
__device__ __noinline__ void attn_core(const LayerW & W, int slot, int n, uint8_t * /*sm*/) {
    if constexpr (T > 5) { // MK_MAXT > 5: all-q8 slots only (checked by the host), the pipelined kernel at every context
        attn_core_q8<T>(W, slot, n);
        return;
    } else
    if (cB.kvq8) {
#if MK_DA_ROUTE || MK_DA_Q4
        // one call site per kernel: attn_core_t is force-inlined, a second site would duplicate it
        const bool q8 = cB.kvb[0][slot] == 8 && cB.kvb[1][slot] == 8;
        if (cB.dattn && (!MK_DA_ROUTE || n + T > ATT_BUCKET) && (MK_DA_Q4 || q8)) {
#if MK_DA_Q4
            if (!q8) attn_core_q8m<T>(W, slot, n);
            else
#endif
            attn_core_q8<T>(W, slot, n);
        } else {
            attn_core_t<T, true>(W, slot, n);
        }
#else
        if (cB.dattn && cB.kvb[0][slot] == 8 && cB.kvb[1][slot] == 8) attn_core_q8<T>(W, slot, n);
        else attn_core_t<T, true>(W, slot, n);
#endif
    } else {
        attn_core_t<T, false>(W, slot, n);
    }
}

// combine split-K partials, apply sigmoid output gate, quantise for o_proj
template <int T>
__device__ __noinline__ void attn_combine(int n, int ws, int slot) { // ws: the slot's window start (attn_core's partition)
    const int L_ctx = n + T;
    int nch, CS;
    attn_dims_w(L_ctx, ws, nch, CS);
    const int tid = threadIdx.x, d = tid & 255, wh = (tid & 255) >> 5, hf = tid >> 8;
    // V stored rotated (MK_KV "r" on V): the head's output is rotated back (WHT, its own inverse) before the gate.
    // Both halves of the block run the same number of items (NH T is even), so the barriers below match.
    const bool rotv = cB.kvq8 && cB.kvrv[slot];
    float * so = reinterpret_cast<float *>(smem) + hf * 256; // scratch (the attention phase's shared memory is free)
    for (int it = blockIdx.x * 2 + hf; it < NH * T; it += 2 * gridDim.x) {
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
        float y = num / den;
        if (rotv) {
            so[d] = y;
            __syncthreads();
            if (wh == 0) {
                float v[8];
#pragma unroll
                for (int k = 0; k < 8; k++) v[k] = so[8 * (tid & 31) + k];
                fwht256(v);
#pragma unroll
                for (int k = 0; k < 8; k++) so[8 * (tid & 31) + k] = v[k];
            }
            __syncthreads();
            y = so[d];
        }
        const float gate = __ldcg(cB.qbuf + t * QG + h * 512 + 256 + d);
        const float o = y / (1.0f + expf(-gate));
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

// Merging sorted top-32 lists (the sampled lm_head's per-warp and per-block candidate lists).  Inserting lists one
// after the other with topk_insert keeps, of all entries, the best 32 under the order (value descending, then
// insertion order): an entry goes behind equal values already there and does not displace an equal 32nd.  So the
// insertion order can be replaced by an explicit sequence key s (list index * 32 + rank in the list) and the lists
// merged in any order with a comparison on (value, s): the result is identical, entry for entry, to the sequential
// merge.  Unfilled slots are (-inf, -1) and rank below every real entry (whose logits are > -inf: topk_insert never
// takes -inf or NaN); they are interchangeable, so their keys do not matter.
__device__ __forceinline__ bool tk_before(float av, int as, float bv, int bs) { return av > bv || (av == bv && as < bs); }

// (v, id, s): this warp's sorted list (lane = rank); merged with the sorted list (bv, bid, bs) into the best 32 of
// both, sorted: reversing b and keeping the better of each pair gives a bitonic sequence holding the best 32, which
// a 5-step bitonic merge sorts.
__device__ __forceinline__ void tk_merge(float & v, int & id, int & s, float bv, int bid, int bs) {
    const int lane = threadIdx.x & 31;
    {
        const float rv = __shfl_sync(0xffffffffu, bv, 31 - lane);
        const int ri = __shfl_sync(0xffffffffu, bid, 31 - lane);
        const int rs = __shfl_sync(0xffffffffu, bs, 31 - lane);
        if (tk_before(rv, rs, v, s)) { v = rv; id = ri; s = rs; }
    }
#pragma unroll
    for (int k = 16; k >= 1; k >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, k);
        const int oi = __shfl_xor_sync(0xffffffffu, id, k);
        const int os = __shfl_xor_sync(0xffffffffu, s, k);
        if (((lane & k) == 0) == tk_before(ov, os, v, s)) { v = ov; id = oi; s = os; } // low lane keeps the better
    }
}

// shared-memory lists [TL][16][32] (values tv, ids ti, sequence keys ts) -> list 0 of each row holds the best 32 of
// its 16 lists (pairwise tree, the merges of a level spread over all warps). Ends with a barrier.
template <int TL>
__device__ __noinline__ void tk_tree16(float * tv, int * ti, int * ts) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
#pragma unroll 1
    for (int d = 1; d < 16; d *= 2) {
        const int per = 8 / d; // merges per row at this level
#pragma unroll 1
        for (int q = warp; q < TL * per; q += NW) {
            const int t = q / per, l = (q % per) * 2 * d;
            const int a = (t * 16 + l) * 32 + lane, b = a + d * 32;
            float v = tv[a];
            int id = ti[a], s = ts[a];
            tk_merge(v, id, s, tv[b], ti[b], ts[b]);
            tv[a] = v;
            ti[a] = id;
            ts[a] = s;
        }
        __syncthreads();
    }
}
// warp w of the last block: block lists w, w + NW, ... of row t (cB.blk_best) merged into shared-memory list (t, w)
// (6 loads in flight per batch; out of line: own registers)
__device__ __noinline__ void tk_grid_chain(int t, float * tv, int * ti, int * ts) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, G = gridDim.x;
    float v = -INFINITY;
    int id = -1, s = 0;
#pragma unroll 1
    for (int k0 = warp; k0 < G; k0 += 6 * NW) {
        float2 e[6];
#pragma unroll
        for (int i = 0; i < 6; i++) {
            const int k = k0 + i * NW;
            e[i] = k < G ? __ldcg(&cB.blk_best[((size_t) k * MAXT + t) * TOPK_MAX + lane]) : make_float2(-INFINITY, __int_as_float(-1));
        }
#pragma unroll
        for (int i = 0; i < 6; i++) {
            const int k = k0 + i * NW;
            if (k < G) tk_merge(v, id, s, e[i].x, __float_as_int(e[i].y), k * 32 + lane);
        }
    }
    const int a = (t * 16 + warp) * 32 + lane;
    tv[a] = v;
    ti[a] = id;
    ts[a] = s;
}
// sampled lm_head: the final candidate rows [MAXT][32] (value, id) in shared memory, above the top-k merge lists,
constexpr int SM_CAND = MAXT * 16 * 32 * 12 > 32768 ? MAXT * 16 * 32 * 12 : 32768;
constexpr int SM_DQ = SM_CAND + MAXT * 32 * 8; // and the drafts' proposals q' staged for sample_targets
constexpr int SM_LSE = SM_DQ + MAXT * 32 * 8;   // the draft lm_head's per-block (max, sum of exp), [gridDim.x]
static_assert(MAXT * 16 * 32 * 12 <= SM_CAND && SM_LSE + 128 * 8 <= SM_MISC, "top-k merge shared memory");

// lm_head over TL rows (qn rows t0..t0+TL-1) with greedy argmax; returns true in the single
// block that finished last, which holds the reduced argmax in s_tok.  TOPK: additionally keeps the
// TOPK_MAX best logits of each row in cand (default cB.cand; for sampling).  FULL: additionally stores
// every logit of each row in cB.logits (scoring).  LSE (TL == 1): also the log-sum-exp of the row's
// true logits; the last block leaves misc[LM_LSE] = lse and misc[LM_BEST] = the best true logit.
// Draft rows: s_tok and the TOPK ids are token ids (mapped through cW.draft_ids).
// PEN: presence / frequency penalty (Launch::pres / freq) on the true logits, the occurrences of a token = its count in
// cB.pcnt plus its occurrences in ctl->tok[1 .. nx0 + t] (the drafts in row t's context; nx0: the drafts before row 0).
// Every value is then the true logit minus the penalty (correctly rounded, no contraction: the same in any T).
// misc() slots ([LM_WLSE, +2 * NW): per-warp lse), above the argmax lists [16, 16 + 2 NW MAXT) and s_last (LM_LAST)
constexpr int LM_WLSE = MAXT > 5 ? 16 + 2 * NW * MAXT : 208, LM_LSE = LM_WLSE + 32, LM_BEST = LM_LSE + 1;
constexpr int LM_LAST = MAXT > 5 ? LM_BEST + 1 : 200;
static_assert(LM_LAST < 400 && (MAXT == 5 || LM_WLSE >= 16 + 2 * NW * MAXT), "lm_head misc slots (s_tok at 400)");
__device__ __forceinline__ float pen_logit(float a, float sc, int c, float pres, float freq) {
    return __fsub_rn(__fsub_rn(__fmul_rn(a, sc), __fmul_rn(freq, (float) c)), c > 0 ? pres : 0.f);
}
template <int TL, bool TOPK = false, bool FULL = false, bool LSE = false, bool PEN = false>
__device__ __noinline__ bool lm_head(int t0, bool draft, uint8_t * /*sm*/, int * s_tok, float2 * cand = nullptr,
                                     float pres = 0.f, float freq = 0.f, int nx0 = 0) {
    static_assert(!LSE || TL == 1, "LSE: one row");
    static_assert(!(PEN && FULL), "no penalty when scoring");
    uint8_t * const sm = smem;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const SX sx = make_sx(sm, U_D);
    float * m = misc(sm);
    // the draft head reads lm_head rows through cW.draft_ids (row r of head_d is lm_head row draft_ids[r])
    const int * ids = draft ? cW.draft_ids : nullptr;
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
    int xt[MAXT - 1]; // PEN: the drafts tok[1..MAXT - 1] (only the first nx0 + t count for row t)
    if constexpr (PEN) {
#pragma unroll
        for (int i = 0; i < MAXT - 1; i++) xt[i] = ((volatile Ctl *) cB.ctl)->tok[1 + i];
    }
    for (int r = 2 * gwarp(); r < head.N; r += 2 * nwarps()) {
        float a[2][TL];
        zero_acc<TL, 2>(a);
        int p0, p1; // token ids of the rows
        {
            const uint8_t * rp[2];
            const __half * dp[2]; // Q6_K: the rows' scale (meta) streams
            p0 = ids ? __ldg(ids + r) : r;
            p1 = ids ? __ldg(ids + r + 1) : r + 1;
            rp[0] = head.w + (size_t) p0 * head.rb;
            rp[1] = head.w + (size_t) p1 * head.rb;
            dp[0] = head.d + (size_t) p0 * head.nd;
            dp[1] = head.d + (size_t) p1 * head.nd;
            rows_t<TL, 5, 2, T_Q6K>(rp, dp, sx, a);
        }
        wsum_acc<TL, 2>(a);
        if constexpr (PEN) { // a[k][t] <- the penalised true logit
            const int c0 = __ldcg(cB.pcnt + p0), c1 = __ldcg(cB.pcnt + p1);
#pragma unroll
            for (int t = 0; t < TL; t++) {
                int e0 = c0, e1 = c1;
#pragma unroll
                for (int i = 0; i < MAXT - 1; i++)
                    if (i < nx0 + t) {
                        e0 += xt[i] == p0;
                        e1 += xt[i] == p1;
                    }
                a[0][t] = pen_logit(a[0][t], m[t], e0, pres, freq);
                a[1][t] = pen_logit(a[1][t], m[t], e1, pres, freq);
            }
        }
        // (below, PEN ? a : a * m = the true logits: PEN's a already are)
        if constexpr (LSE) {
            const float v0 = PEN ? a[0][0] : a[0][0] * m[0], v1 = PEN ? a[1][0] : a[1][0] * m[0];
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
                topk_insert(bv[t], bi[t], PEN ? a[0][t] : a[0][t] * m[t], r);
                topk_insert(bv[t], bi[t], PEN ? a[1][t] : a[1][t] * m[t], r + 1);
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
    // TOPK: lists [TL][16][32] in shared memory for the merges (the activations are no longer needed)
    float * const tkv = reinterpret_cast<float *>(sm);
    int * const tki = reinterpret_cast<int *>(sm) + TL * 512;
    int * const tks = reinterpret_cast<int *>(sm) + 2 * TL * 512;
    if constexpr (TOPK) {
        // merge the warps' lists into the block's (the order of a sequential merge over warps 0..15: tk_merge)
        __syncthreads();
#pragma unroll
        for (int t = 0; t < TL; t++) {
            const int a = (t * 16 + warp) * 32 + lane;
            tkv[a] = bv[t];
            tki[a] = bi[t];
            tks[a] = warp * 32 + lane;
        }
        __syncthreads();
        tk_tree16<TL>(tkv, tki, tks);
        for (int q = tid; q < TL * 32; q += NT) {
            const int t = q >> 5, a = t * 512 + (q & 31);
            cB.blk_best[((size_t) blockIdx.x * MAXT + t) * TOPK_MAX + (q & 31)] = make_float2(tkv[a], __int_as_float(tki[a]));
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
    int * s_last = reinterpret_cast<int *>(m + LM_LAST);
    if (tid == 0) s_last[0] = (atomicAdd(cB.cnt, 1u) == gridDim.x - 1);
    __syncthreads();
    if (!s_last[0]) return false;
    __threadfence();
    if constexpr (LSE) { // the blocks' (max, sum of exp) -> shared memory in one parallel load; the sum stays in block order
        float2 * const sl = reinterpret_cast<float2 *>(sm + SM_LSE);
        for (int k = tid; k < (int) gridDim.x; k += NT) sl[k] = __ldcg(&cB.blk_lse[k]);
        __syncthreads();
        if (tid == NT - 1) {
            float M = -INFINITY, S = 0.f;
            for (int k = 0; k < (int) gridDim.x; k++) M = fmaxf(M, sl[k].x);
            for (int k = 0; k < (int) gridDim.x; k++) {
                const float2 e = sl[k];
                S += e.y * __expf(e.x - M);
            }
            m[LM_LSE] = M + __logf(S);
        }
    }
    if constexpr (TOPK) {
        // merge the blocks' lists (the order of a sequential merge over blocks 0..G-1): warp w merges lists
        // w, w + 16, ... of each row, then a tree over the 16 warps' lists
#pragma unroll 1
        for (int t = 0; t < TL; t++) tk_grid_chain(t, tkv, tki, tks);
        // main head: the drafts' q' (previous launch) for sample_targets, staged while the merge runs
        if (!draft)
            for (int q = tid; q < (TL - 1) * TOPK_MAX; q += NT) reinterpret_cast<float2 *>(sm + SM_DQ)[q] = __ldcg(cB.dq + q);
        __syncthreads();
        tk_tree16<TL>(tkv, tki, tks);
        if (warp < TL) {
            const float v = tkv[warp * 512 + lane];
            int id = tki[warp * 512 + lane];
            if (draft && cW.draft_ids && id >= 0) id = cW.draft_ids[id];
            (cand ? cand : cB.cand)[warp * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
            reinterpret_cast<float2 *>(sm + SM_CAND)[warp * TOPK_MAX + lane] = make_float2(v, __int_as_float(id));
            const int best = __shfl_sync(0xffffffffu, id, 0);
            if (lane == 0) {
                s_tok[warp] = best;
                if (LSE) m[LM_BEST] = v;
            }
        }
    } else if (warp < TL) {
        // the blocks' best of row `warp`: "larger value, then smaller id" is a total order (NaN never wins), so a warp
        // reduction finds the same entry as the sequential scan over blocks
        float b = -INFINITY;
        int ix = 0;
        for (int k = lane; k < (int) gridDim.x; k += 32) {
            const float2 e = __ldcg(&cB.blk_best[k * MAXT + warp]);
            const int id = __float_as_int(e.y);
            if (e.x > b || (e.x == b && id < ix)) { b = e.x; ix = id; }
        }
#pragma unroll
        for (int o = 16; o >= 1; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, b, o);
            const int oi = __shfl_xor_sync(0xffffffffu, ix, o);
            if (ob > b || (ob == b && oi < ix)) { b = ob; ix = oi; }
        }
        if (lane == 0) {
            s_tok[warp] = (draft && cW.draft_ids) ? cW.draft_ids[ix] : ix;
            if (LSE) m[LM_BEST] = PEN ? b : b * m[0];
        }
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
// main_control unchanged.  cand: the rows' candidates (cB.cand, here the shared-memory copy lm_head leaves at
// SM_CAND), dq: the drafts' proposals (cB.dq, staged at SM_DQ): single-thread code, so every load is latency.
__device__ void sample_targets(const Launch & P, int T, int tl, int * tg, const float2 * cand, const float2 * dq) {
    const Ctl * c = cB.ctl;
    const unsigned long long n = (unsigned long long) c->n;
    float p[TOPK_MAX];
    for (int i = 0; i < tl; i++) {
        const float2 * cd = cand + i * TOPK_MAX;
        const int K = cand_probs(cd, P, p);
        if (P.mode == MODE_DECODE && i < T - 1 && c->dsmp) {
            const int d = c->tok[i + 1];
            const float2 * q = dq + i * TOPK_MAX;
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
#if MK_WV
#include "wv.cuh"
#endif

__device__ void main_control(int mode, int last, int T, const int * tgt, bool pen) {
    Ctl * c = cB.ctl;
    const int n = c->n;
    int a, bonus = -1;
    // pen: count the emitted tokens (the response so far) for the presence / frequency penalty
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
    for (int p = m; p <= P && R < MAXT + 3; p++, R++) {
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
    if (pen)
        for (int i = 0; i < c->n_out; i++) {
            const int v = c->out[i];
            if (cB.pcnt[v] < 0xffff) cB.pcnt[v]++;
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
#if MK_WV
        // wv phases where at least one matrix of the phase is in the wv layout (the others: dp4a, per tensor at every T).
        // T >= 6 (MK_MAXT > 5): always the wv phase functions (their dp4a part runs the hybrid types, the only ones not in
        // the wv layout there), and every main-layer ffn_down is in the wv layout (the loader checks it).
        constexpr bool WIDE = T > 5;
        if (W.attn) {
            if constexpr (WIDE) attn_proj_wv<T>(W, L / 4, n);
            else if (W.q.fm || W.k.fm || W.v.fm) attn_proj_wv<T>(W, L / 4, n);
            else attn_proj<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_PROJ);
            attn_core<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_CORE);
            attn_combine<T>(n, 0, L / 4);
            gsync(epoch, PH_ATTN_COMB);
            if (W.o.fm) resid_wv<T, false, VD>(W.o, cB.qo, W.post_norm, 0);
            else resid_dp<T, 6>(W.o, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_O);
        } else {
            if constexpr (WIDE) dn_proj_wv<T>(W, dl, n, dn_c);
            else if (W.qkv.fm || W.gate.fm) dn_proj_wv<T>(W, dl, n, dn_c);
            else dn_proj<T>(W, dl, n, dn_c, sm);
            gsync(epoch, PH_DN_PROJ);
            dn_core<T>(W, dl, n, dn_c, n_commit, sm);
            gsync(epoch, PH_DN_CORE);
            if (W.sout.fm) resid_wv<T, false, VD>(W.sout, cB.qo, W.post_norm, 0);
            else resid_dp<T, 6>(W.sout, cB.qo, cB.x, true, W.post_norm, sm);
            gsync(epoch, PH_RESID_SOUT);
            dl++;
        }
        if constexpr (WIDE) ffn_gu_wv<T>(W);
        else if (W.fg.fm || W.fu.fm) ffn_gu_wv<T>(W);
        else ffn_gu<T>(W, sm);
        gsync(epoch, PH_FFN_GU);
        if constexpr (WIDE) resid_wv<T, true, FF>(W.fd, cB.qa, wnext, 1);
        else if (W.fd.fm) resid_wv<T, true, FF>(W.fd, cB.qa, wnext, 1);
        else resid_any<T, 17>(W.fd, cB.qa, cB.x, true, wnext, sm);
        gsync(epoch, PH_FFN_DOWN);
#else
        if (W.attn) {
            attn_proj<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_PROJ);
            attn_core<T>(W, L / 4, n, sm);
            gsync(epoch, PH_ATTN_CORE);
            attn_combine<T>(n, 0, L / 4);
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
#endif
    }
    write_h_rows<T>(cB.x, cB.qn, cW.onorm, cB.h_hist, n, true, sm);

    int * s_tok = reinterpret_cast<int *>(misc(sm) + 400);
    bool am_last;
    int tl = 0;
    const bool sampling = P.temp > 0.0f && !P.score;
    const bool pen = (P.pres != 0.f || P.freq != 0.f) && !P.score; // (the first token of a response has nothing to penalise)
    // The wide pass (MK_MAXT > 5, T = MAXT): decode launches of P.T = 6 .. MAXT rows run as one MAXT-row pass; the rows
    // P.T .. MAXT - 1 are padding (stale ctl tokens: their K / V, conv, delta-net ring and h rows land at future
    // positions, rewritten before any real row reads them; real rows never attend to them, and the host keeps
    // positions n + 1 .. n + MAXT in one attention partition). Only rows < P.T reach the verify / control.
    // (Prefill and scoring launches stay at T <= 5: no FULL lm_head here.)
#if MK_MAXT > 5
    const int Tr = T == MAXT ? P.T : T;
#else
    constexpr int Tr = T;
#endif
    if (P.mode == MODE_DECODE || P.score) {
        if (sampling) am_last = pen ? lm_head<T, true, false, false, true>(0, false, sm, s_tok, nullptr, P.pres, P.freq, 0)
                                    : lm_head<T, true>(0, false, sm, s_tok);
        else if (P.score == 2) {
            if constexpr (T <= 5) am_last = lm_head<T, false, true>(0, false, sm, s_tok);
            else am_last = false;
        } else am_last = pen ? lm_head<T, false, false, false, true>(0, false, sm, s_tok, nullptr, P.pres, P.freq, 0)
                             : lm_head<T>(0, false, sm, s_tok);
        tl = Tr;
    } else if (P.last) {
        am_last = sampling ? lm_head<1, true>(T - 1, false, sm, s_tok) : lm_head<1>(T - 1, false, sm, s_tok);
        tl = 1;
    } else {
        am_last = blockIdx.x == 0;
    }
    if (am_last && threadIdx.x == 0) {
        int tg[MAXT];
        for (int i = 0; i < tl; i++) tg[i] = s_tok[i];
        if (sampling) // lm_head left the candidates at SM_CAND and the drafts' q' at SM_DQ
            sample_targets(P, Tr, tl, tg, reinterpret_cast<const float2 *>(sm + SM_CAND), reinterpret_cast<const float2 *>(sm + SM_DQ));
        for (int i = 0; i < tl; i++) cB.ctl->tgt[i] = tg[i];
        main_control(P.mode, P.last, Tr, P.score ? s_tok + T - 1 : tg, pen);
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

    // M0: concat(enorm(embed(tok)), hnorm(h_prev)) -> qe.  M0_NB blocks per row: each computes the row's two norms
    // (the same sums in the same order in every block) and quantises its share of the 2 U_D units (was: one block per
    // row doing all 320 units, ~20 us while the grid waited)
    constexpr int M0_NB = 10;
    if (blockIdx.x < R * M0_NB) {
        const int r = blockIdx.x / M0_NB, s = blockIdx.x - r * M0_NB;
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
        for (int j = s * NW + warp; j < 2 * U_D; j += M0_NB * NW) {
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
    attn_combine<R>(pos0, mtp_ws(N_ATT_SLOTS - 1, pos0), N_ATT_SLOTS - 1);
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
        // presence / frequency penalty: draft k + 1 has the committed tokens and drafts 1..k in its context
        const bool pen = P.pres != 0.f || P.freq != 0.f;
        const bool lastb = P.spec_smp ? (pen ? lm_head<1, true, false, true, true>(R - 1, true, sm, s_tok, tl, P.pres, P.freq, k)
                                             : lm_head<1, true, false, true>(R - 1, true, sm, s_tok, tl))
                                      : (pen ? lm_head<1, false, false, true, true>(R - 1, true, sm, s_tok, nullptr, P.pres, P.freq, k)
                                             : lm_head<1, false, false, true>(R - 1, true, sm, s_tok));
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
            // (draft_sample reads lm_head's shared-memory copy of tl: the same values, without the L2 round trips)
            c->tok[1 + k] = P.spec_smp ? draft_sample(P, reinterpret_cast<const float2 *>(sm + SM_CAND), cB.dq + k * TOPK_MAX,
                                                      (unsigned long long) c->mtp_P + 1 + k)
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
#if MK_MAXT > 5
// T = 6 .. MAXT: one MAXT-row pass (padding rows, see main_forward), out of line: a separate function keeps mk_cycle (and
// ptxas's memory for it) at the size of the MAXT 5 kernel
__device__ __noinline__ void run_wide(const Launch & P, unsigned & epoch) { main_forward<MAXT>(P, smem, epoch); }
#endif

template <int T>
__device__ void bench_phase(int which, int layer, uint8_t * /*sm*/, unsigned & epoch) {
    uint8_t * const sm = smem;
    for (int it = 0; it < 4; it++) {
        const int L = layer + 4 * it; // different layers: weights are not L2-resident, code is
        const LayerW & W = cW.L[L];
#if MK_WV
        if (which == 0) {
            if (W.fg.fm || W.fu.fm) ffn_gu_wv<T>(W);
            else ffn_gu<T>(W, sm);
        } else if (which == 1) {
            if (W.fd.fm) resid_wv<T, true, FF>(W.fd, cB.qa, W.attn_norm, 1);
            else resid_any<T, 17>(W.fd, cB.qa, cB.x, true, W.attn_norm, sm);
        } else if (which == 2) {
            if (!W.attn) {
                if (W.qkv.fm || W.gate.fm) dn_proj_wv<T>(W, 0, 0, 0);
                else dn_proj<T>(W, 0, 0, 0, sm);
            }
        }
#else
        if (which == 0) ffn_gu<T>(W, sm);
        else if (which == 1) {
            resid_any<T, 17>(W.fd, cB.qa, cB.x, true, W.attn_norm, sm);
        } else if (which == 2) {
            if (!W.attn) dn_proj<T>(W, 0, 0, 0, sm);
        }
#endif
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
#if MK_MAXT > 5
        default: run_wide(P, epoch); break; // T = 6 .. MAXT
#endif
    }
    if (P.ndraft == 0 || P.score) return;
    volatile Ctl * vc = cB.ctl;
    const int Rtot = vc->mtp_R;
    const bool lg = vc->mtp_logits;
    int done = 0;
    while (done < Rtot) {
        const int R = min(5, Rtot - done); // (the MTP block's dp4a ffn_down holds at most 5 rows)
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

// Q3_K's 12 bytes of packed 6-bit scales -> sc - 32 (ggml's unpacking)
void q3k_scales(const uint8_t * packed, int8_t (&sc)[16]) {
    uint32_t aux[4];
    memcpy(aux, packed, 12);
    const uint32_t m1 = 0x03030303u, m2 = 0x0f0f0f0fu, tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & m2) | (((tmp >> 4) & m1) << 4);
    aux[3] = ((aux[1] >> 4) & m2) | (((tmp >> 6) & m1) << 4);
    aux[0] = (aux[0] & m2) | (((tmp >> 0) & m1) << 4);
    aux[1] = (aux[1] & m2) | (((tmp >> 2) & m1) << 4);
    for (int i = 0; i < 16; i++) sc[i] = (int8_t) ((int) reinterpret_cast<const uint8_t *>(aux)[i] - 32);
}

// GGUF Q3_K row (110 B blocks: hmask[32] | qs[64] | scales[12] | fp16 d) -> the native layout of rows_q3k: values (96 B
// per block) and the Q6_K meta stream (18 B per block in pairs). y: the row as ggml dequantizes it, for the self-check
// (the native values, dequantized as ggml does, must reproduce it bit for bit). nb must be even.
bool repack_q3k(const uint8_t * src, uint8_t * q, uint8_t * meta, int nb, const float * y) {
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = src + b * 110, *hm = blk, *qs = blk + 32;
        uint8_t u3[256];
        for (int e = 0; e < 256; e++) {
            const int n = e >> 7, j = (e >> 5) & 3, l = e & 31;
            u3[e] = (uint8_t) (((qs[32 * n + l] >> (2 * j)) & 3) | (((hm[l] >> (4 * n + j)) & 1) << 2));
        }
        uint8_t * o = q + (size_t) b * 96;
        memset(o, 0, 96);
        for (int w = 0; w < 8; w++) { // unit 2p + hh
            const int p = w >> 1, hh = w & 1;
            for (int i = 0; i < 16; i++) {
                const int ea = 64 * p + 16 * hh + i, eb = ea + 32, m = i & 3, j = i >> 2;
                o[8 * w + m] |= (uint8_t) ((u3[ea] & 3) << (2 * j));
                o[8 * w + 4 + m] |= (uint8_t) ((u3[eb] & 3) << (2 * j));
                o[64 + 4 * w + m] |= (uint8_t) (((u3[ea] >> 2) << j) | ((u3[eb] >> 2) << (4 + j)));
            }
        }
        int8_t sc[16];
        q3k_scales(blk + 96, sc);
        uint16_t dbits;
        memcpy(&dbits, blk + 108, 2);
        uint8_t * mp = meta + (size_t) (b & ~1) * 18;
        memcpy(mp + 16 * (b & 1), sc, 16);
        memcpy(mp + 32 + 2 * (b & 1), &dbits, 2);
        // self-check: decode the native bytes as rows_q3k does, dequantize as ggml (dl = d (sc - 32), dl q)
        const float d = ggml_fp16_to_fp32(dbits);
        for (int e = 0; e < 256; e++) {
            const int p = e >> 6, r = e & 63, isb = r >> 5, w = 2 * p + ((r >> 4) & 1), i = r & 15, m = i & 3, j = i >> 2;
            const int lo = (o[8 * w + 4 * isb + m] >> (2 * j)) & 3, hb = (o[64 + 4 * w + m] >> (4 * isb + j)) & 1;
            const int8_t s = (int8_t) mp[16 * (b & 1) + 4 * p + 2 * isb + ((r >> 4) & 1)];
            const float v = (d * (float) s) * (float) (lo + 4 * hb - 4);
            if (memcmp(&v, &y[b * 256 + e], 4) != 0) return false;
        }
    }
    return true;
}

// ggml's iq3s_grid, recovered with ggml's own IQ3_S dequantization (d = 1, scale 1, no signs: index -> 4 magnitudes)
std::vector<uint32_t> iq3s_grid_host() {
    std::vector<uint32_t> g(512);
    const auto * tr = ggml_get_type_traits(GGML_TYPE_IQ3_S);
    uint8_t blk[110] = {};
    const uint16_t one = 0x3c00;
    float y[256];
    for (int idx = 0; idx < 512; idx++) {
        memset(blk, 0, sizeof blk);
        memcpy(blk, &one, 2);
        blk[2] = (uint8_t) (idx & 0xff); // qs[0]: index 0 of sub-block 0
        blk[66] = (uint8_t) (idx >> 8);  // qh[0] bit 0
        tr->to_float(blk, y, 256);
        uint32_t v = 0;
        for (int m = 0; m < 4; m++) {
            const int gm = (int) y[m];
            if ((float) gm != y[m] || gm < 1 || gm > 15) throw std::runtime_error("iq3s_grid: unexpected IQ3_S dequantization");
            v |= (uint32_t) gm << (8 * m);
        }
        g[idx] = v;
    }
    return g;
}

// GGUF IQ3_S row (110 B blocks: fp16 d | qs[64] | qh[8] | signs[32] | scales[4]) -> the native layout of rows_iq3s:
// values (96 B per block) and meta (16 B per block). Self-check against y as repack_q3k.
bool repack_iq3s(const uint8_t * src, uint8_t * q, uint8_t * meta, int nb, const float * y, const std::vector<uint32_t> & grid) {
    for (int b = 0; b < nb; b++) {
        const uint8_t *blk = src + b * 110, *qs = blk + 2, *qh = blk + 66, *signs = blk + 74, *scales = blk + 106;
        uint8_t * o = q + (size_t) b * 96;
        uint8_t * mt = meta + (size_t) b * 16;
        memset(o, 0, 96);
        memset(mt, 0, 16);
        for (int w = 0; w < 8; w++) { // unit 2p + hh
            const int p = w >> 1, hh = w & 1;
            for (int isb = 0; isb < 2; isb++) {
                const int ib = 2 * p + isb; // the 32-weight sub-block of A (isb 0) or B (isb 1)
                for (int c = 0; c < 4; c++) {
                    const int k = 4 * hh + c; // grid index k of the sub-block (its weights 4k .. 4k + 3)
                    o[8 * w + 4 * isb + c] = qs[8 * ib + k];
                    mt[w] |= (uint8_t) (((qh[ib] >> k) & 1) << (4 * isb + c));
                }
                for (int i = 0; i < 16; i++) {
                    const int off = 16 * hh + i; // weight of the sub-block
                    const int sgn = (signs[4 * ib + off / 8] >> (off % 8)) & 1;
                    o[64 + 4 * w + (i & 3)] |= (uint8_t) (sgn << (4 * isb + (i >> 2)));
                }
            }
        }
        memcpy(mt + 8, scales, 4);
        memcpy(mt + 12, blk, 2);
        uint16_t dbits;
        memcpy(&dbits, blk, 2);
        const float d = ggml_fp16_to_fp32(dbits);
        for (int e = 0; e < 256; e++) {
            const int p = e >> 6, r = e & 63, isb = r >> 5, w = 2 * p + ((r >> 4) & 1), i = r & 15, m = i & 3, c = i >> 2;
            const int idx = o[8 * w + 4 * isb + c] | (((mt[w] >> (4 * isb + c)) & 1) << 8);
            const int g = (grid[idx] >> (8 * m)) & 0xff;
            const int sgn = (o[64 + 4 * w + m] >> (4 * isb + c)) & 1;
            const float db = d * (float) (1 + 2 * ((mt[8 + p] >> (4 * isb)) & 0xf));
            const float v = db * (float) g * (sgn ? -1.f : 1.f);
            if (memcmp(&v, &y[b * 256 + e], 4) != 0) return false;
        }
    }
    return true;
}

#if MK_WV
#include "wv_host.cuh"
#endif

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
#if MK_MAXT > 5
        // generalised gate (any depth): stop after draft k >= 1 below 0.4, drop the deepest draft (k = n_draft - 1, the
        // T = n_draft + 1 verify row) below drop_last; at n_draft 4 the same gates as the MAXT 5 default
        float gate_drop[MAXT - 1] = {}, gate_stop[MAXT - 1] = {0.f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f};
        float drop_last = 0.4f;
        // context cap: from position dctx on, at most 4 drafts (T <= 5: at long context the wide pass's second
        // head-group pass over K / V costs more than drafts 5..7 gain)
        int dctx = INT_MAX;
#else
        float gate_drop[4] = {0.f, 0.f, 0.f, 0.4f}, gate_stop[4] = {0.f, 0.4f, 0.4f, 0.4f}; // on-device early stop
#endif
        int gate_chain = 1;
        int h1 = 1; // sampling: sampled drafts (Launch::spec_smp)
    } pol;
    // the policy's draft depth at position pos: n_draft, or (MK_MAXT > 5) at most 4 from Policy::dctx on
#if MK_MAXT > 5
    int depth_top(int pos) const { return pos >= pol.dctx ? std::min(n_draft_max, 4) : n_draft_max; }
#else
    int depth_top(int) const { return n_draft_max; }
#endif
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
    // DeltaNet checkpoints: the state after position pos - 1 (recurrent state, conv ring, h ring, ctl), which lets a
    // caller rewind to pos (KV rows before it are never rewritten) and prefill from there. A pool of MK_CK_SLOTS
    // (default 8) checkpoints of 0.17 GB each in pinned host memory (MK_CK_DEVICE=1: one, in device memory); when it
    // is full, ck_assign evicts by coverage. Saving never waits on the host and adds no VRAM:
    // - Inside a prefill (pf_run, Snap): at position P of a chunk, each DeltaNet layer's scan is split at P and its
    //   first part writes the state at P into a staging slot in hp.xm (idle until the chunk's MTP part), next to the
    //   conv ring at P (pk_ck_conv); a device-to-host copy on ck_sv moves the staging slot into the checkpoint while
    //   the next layers run. The h ring at P goes straight into the mapped host buffer (pk_ck_hist). Stream 0 waits
    //   (on the device) only before reusing a staging slot and before the MTP part. The scan is split only at points
    //   of its 32-token grid, which changes no arithmetic: saving a checkpoint never changes what the prefill computes.
    //   A position off the grid (the exact ck_at of a chat turn) is saved by a side scan from the grid point before
    //   it (the state a prefill ending there computes, as the earlier split into two pf_runs did).
    // - At the end of a prompt (ck_save_end, F9): a device-to-device snapshot into the idle batched-prefill scratch
    //   on stream 0 (~0.4 ms), then the device-to-host copy on ck_sv, which runs under the first decode launch.
    // - Restore: the host-to-device copy runs on ck_st in CK_NG groups of DeltaNet layers, issued once the next
    //   prefill has uploaded its tokens (ck_issue_restore: nothing on stream 0 then queues behind it on the copy
    //   engine); stream 0 waits for a group's event only before the first kernel that reads that group's state
    //   (pf_run: dn_prep of the group's first layer). Same bytes, same order of use: bit-identical.
    // If the scratch is too small for the end snapshot (MK_PF_CAP below ~640), the end save copies straight from the
    // state and stream 0 waits for it before the state is next modified; below ~340 there are too few staging slots
    // too, and a checkpoint inside a prompt splits the prefill into two pf_runs around it (the earlier scheme).
    static constexpr size_t CK_SL = (size_t) NVH * 128 * 128, CK_CL = (size_t) CONV_RING * QKV; // floats per DeltaNet layer
    static constexpr size_t CK_NS = 48 * CK_SL, CK_NC = 48 * CK_CL, CK_NH = (size_t) H_RING * D;
    static constexpr int CK_NG = 5;
    static constexpr int ck_gb[CK_NG + 1] = {0, 1, 4, 12, 24, 48}; // restore group g: DeltaNet layers [ck_gb[g], ck_gb[g + 1])
    struct CkSlot {
        void * mem = nullptr;                                       // pinned host block (device memory: MK_CK_DEVICE)
        float *dstate = nullptr, *conv = nullptr, *hhist = nullptr; // in it
        float * hdev = nullptr;                                     // hhist as kernels address it
        Ctl ctl{};
        int pos = -1;                                               // -1: free
        cudaEvent_t ev = nullptr;                                   // the copies into this slot (ck_sv) are done
        bool pending = false;                                       // ... may still be running
    };
    struct Snap { int pos, slot; };
    std::vector<CkSlot> ck;
    bool ck_dev = false;
    double ck_ms = 0;             // last save / restore enqueue time (MK_MEMLOG)
    cudaStream_t ck_st = nullptr; // restores
    cudaStream_t ck_sv = nullptr; // saves
    cudaEvent_t ck_ev_src = nullptr, ck_ev_sv = nullptr, ck_ev_scr = nullptr, ck_ev_grp[CK_NG] = {};
    bool ck_scr_busy = false; // a save copy on ck_sv may still read the prefill scratch (ck_ev_scr)
    int ck_direct = -1;       // slot of an end save that reads the live state directly: stream 0 waits before modifying it
    int ck_rs_slot = -1;      // slot of a restore whose copies are not issued yet (ck_issue_restore)
    int ck_rs_src = -1;       // slot of the last restore (its copies may be running until its prefill waited for them)
    int ck_rs_next = CK_NG;   // first restore group stream 0 has not waited for (CK_NG: no restore pending)
    struct CkChunk { char * dev; char * scr; size_t off, n; };
    std::vector<CkChunk> ck_plan; // end save: state -> prefill scratch (stream 0) -> slot byte offset off (ck_sv)
    // staging slots of in-prefill saves in hp.xm: one DeltaNet layer's state + conv ring each
    static constexpr size_t STG_F = (CK_SL + CK_CL + 63) & ~(size_t) 63;
    static constexpr int STG_MAX = 16;
    int stg_n = 0, stg_next = 0, stg_last = -1;
    cudaEvent_t stg_ev[STG_MAX] = {};
    bool stg_busy[STG_MAX] = {};

    // Allocated at construction: cudaHostAlloc of 0.17 GB takes 0.1-0.5 s under WSL2, which used to land in the
    // first prompt's time. A pinned slot that fails to allocate ends the pool there (the first: one device slot).
    void ck_alloc() {
        const char * cd = getenv("MK_CK_DEVICE");
        ck_dev = cd && cd[0] == '1';
        int nslots = 8;
        if (const char * e = getenv("MK_CK_SLOTS")) nslots = std::max(1, std::min(64, atoi(e)));
        const size_t nf = CK_NS + CK_NC + CK_NH;
        for (int i = 0; i < (ck_dev ? 1 : nslots); i++) {
            CkSlot s;
            if (!ck_dev && cudaHostAlloc(&s.mem, nf * 4, cudaHostAllocMapped) != cudaSuccess) {
                cudaGetLastError();
                s.mem = nullptr;
                if (i > 0) {
                    fprintf(stderr, "[mk] warning: no pinned host memory for DeltaNet checkpoint %d: keeping %d\n", i + 1, i);
                    break;
                }
                ck_dev = true;
            }
            if (ck_dev) s.mem = dalloc<float>(nf, false);
            s.dstate = static_cast<float *>(s.mem);
            s.conv = s.dstate + CK_NS;
            s.hhist = s.conv + CK_NC;
            if (ck_dev) s.hdev = s.hhist;
            else CK(cudaHostGetDevicePointer(reinterpret_cast<void **>(&s.hdev), s.hhist, 0));
            CK(cudaEventCreateWithFlags(&s.ev, cudaEventDisableTiming));
            ck.push_back(s);
        }
        CK(cudaStreamCreateWithFlags(&ck_st, cudaStreamNonBlocking));
        CK(cudaStreamCreateWithFlags(&ck_sv, cudaStreamNonBlocking));
        for (cudaEvent_t * e : {&ck_ev_src, &ck_ev_sv, &ck_ev_scr}) CK(cudaEventCreateWithFlags(e, cudaEventDisableTiming));
        for (auto & e : ck_ev_grp) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        for (auto & e : stg_ev) CK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        mem_mark(ck_dev ? "DeltaNet checkpoint" : "DeltaNet checkpoints (host)");
    }
    void ck_free() {
        if (ck_st) cudaStreamSynchronize(ck_st);
        if (ck_sv) cudaStreamSynchronize(ck_sv);
        for (auto & s : ck) {
            if (s.ev) cudaEventDestroy(s.ev);
            if (s.mem && !ck_dev) cudaFreeHost(s.mem);
        }
        for (cudaEvent_t e : {ck_ev_src, ck_ev_sv, ck_ev_scr})
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : ck_ev_grp)
            if (e) cudaEventDestroy(e);
        for (cudaEvent_t e : stg_ev)
            if (e) cudaEventDestroy(e);
        if (ck_st) cudaStreamDestroy(ck_st);
        if (ck_sv) cudaStreamDestroy(ck_sv);
    }
    // end-save snapshot plan over the prefill scratch buffers (after pf_init; empty when they are too small) and
    // the staging slots in hp.xm
    void ck_make_plan() {
        ck_plan.clear();
        stg_n = (int) std::min<size_t>(STG_MAX, (size_t) hp.cap * D / STG_F);
        if (ck_dev) return; // (device checkpoint: the end save copies on stream 0)
        const size_t cap = hp.cap;
        const std::pair<void *, size_t> scr[] = {
            {hp.y0, cap * FF * 4}, {hp.y1, cap * FF * 4}, {hp.xh2, cap * FF * 2}, {hp.y2, cap * VD * 4}, {hp.x, cap * D * 4},
            {hp.xm, cap * D * 4},  {hp.dq, cap * QKV * 2}, {hp.xh, cap * 2 * D * 2}, {hp.qh, cap * NH * HD * 2}};
        const std::tuple<void *, size_t, size_t> reg[] = { // state, its byte offset in a slot, bytes
            {hb.dstate, 0, CK_NS * 4}, {hb.conv, CK_NS * 4, CK_NC * 4}, {hb.h_hist, (CK_NS + CK_NC) * 4, CK_NH * 4}};
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
                                   std::get<1>(r) + done, n});
                done += n;
                so += n;
            }
        }
    }
    // stream 0: before overwriting the prefill scratch, wait for an end save's copy still reading it
    void ck_scr_guard() {
        if (!ck_scr_busy) return;
        CK(cudaStreamWaitEvent(0, ck_ev_scr, 0));
        ck_scr_busy = false;
    }
    // stream 0: before modifying the live state, wait for an end save that reads it directly (no snapshot)
    void ck_guard_state() {
        if (ck_direct < 0) return;
        CK(cudaStreamWaitEvent(0, ck[ck_direct].ev, 0));
        ck_direct = -1;
    }
    // Issues a due restore's host-to-device copies on ck_st (5 groups, an event after each). Deferred from ck_restore
    // to just after the next prefill's token upload: a host-to-device copy on stream 0 (ctl, token ids, draft
    // vocabulary rows) queued behind the 0.17 GB on the copy engine would stall the host for the whole restore.
    void ck_issue_restore() {
        if (ck_rs_slot < 0) return;
        const CkSlot & s = ck[ck_rs_slot];
        ck_rs_src = ck_rs_slot;
        ck_rs_slot = -1;
        // stream 0's earlier work on the state (and on the slot's h ring) is complete before ck_st overwrites the
        // state, and so are the slot's save copies and a direct save still reading the state (all normally long done)
        CK(cudaEventRecord(ck_ev_src, 0));
        CK(cudaStreamWaitEvent(ck_st, ck_ev_src, 0));
        if (s.pending) CK(cudaStreamWaitEvent(ck_st, s.ev, 0));
        if (ck_direct >= 0) {
            CK(cudaStreamWaitEvent(ck_st, ck[ck_direct].ev, 0));
            ck_direct = -1;
        }
        const cudaMemcpyKind k = ck_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;
        CK(cudaMemcpyAsync(hb.h_hist, s.hhist, CK_NH * 4, k, ck_st)); // group 0
        for (int g = 0; g < CK_NG; g++) {
            const size_t l0 = ck_gb[g], nl = ck_gb[g + 1] - ck_gb[g];
            CK(cudaMemcpyAsync(hb.conv + l0 * CK_CL, s.conv + l0 * CK_CL, nl * CK_CL * 4, k, ck_st));
            CK(cudaMemcpyAsync(hb.dstate + l0 * CK_SL, s.dstate + l0 * CK_SL, nl * CK_SL * 4, k, ck_st));
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
        if (ck_rs_next >= CK_NG) ck_rs_src = -1; // (stream 0 waited for all of it)
    }
    // checkpoint at the end of a prompt into slot si (prefill read ctl back after its last launch: hctl is the device's)
    void ck_save_end(int si) {
        ck_wait_restore();
        CkSlot & s = ck[si];
        const auto t0 = std::chrono::steady_clock::now();
        if (ck_dev) { // device checkpoint: stream-ordered device copies
            CK(cudaMemcpyAsync(s.dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(s.conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToDevice, 0));
            CK(cudaMemcpyAsync(s.hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToDevice, 0));
        } else {
            ck_scr_guard(); // a previous end save may still read the scratch (normally long done)
            for (const auto & c : ck_plan) CK(cudaMemcpyAsync(c.scr, c.dev, c.n, cudaMemcpyDeviceToDevice, 0));
            CK(cudaEventRecord(ck_ev_sv, 0));
            CK(cudaStreamWaitEvent(ck_sv, ck_ev_sv, 0));
            char * base = static_cast<char *>(s.mem);
            if (!ck_plan.empty()) {
                for (const auto & c : ck_plan) CK(cudaMemcpyAsync(base + c.off, c.scr, c.n, cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaEventRecord(ck_ev_scr, ck_sv));
                ck_scr_busy = true;
            } else {
                CK(cudaMemcpyAsync(s.dstate, hb.dstate, CK_NS * 4, cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaMemcpyAsync(s.conv, hb.conv, CK_NC * 4, cudaMemcpyDeviceToHost, ck_sv));
                CK(cudaMemcpyAsync(s.hhist, hb.h_hist, CK_NH * 4, cudaMemcpyDeviceToHost, ck_sv));
                ck_direct = si;
            }
            CK(cudaEventRecord(s.ev, ck_sv));
            s.pending = true;
        }
        s.ctl = hctl;
        s.pos = hctl.dn_c;
        ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (memlog)
            fprintf(stderr, "[mk-mem] checkpoint save at %d enqueued in %.2f ms (%s)\n", s.pos, ck_ms,
                    ck_dev ? "device" : ck_plan.empty() ? "host, direct" : "host, via the prefill scratch");
    }
    void ck_restore(int si) {
        ck_wait_restore(); // (a previous restore not yet waited for: normally none)
        const auto t0 = std::chrono::steady_clock::now();
        ck_rs_slot = si; // issued by the next pf_run after its token upload, or by whatever touches the state first
        const CkSlot & s = ck[si];
        hctl = s.ctl;
        hctl.mtp_valid = std::min(hctl.mtp_valid, s.pos);
        CK(cudaMemcpy(hb.ctl, &hctl, sizeof(Ctl), cudaMemcpyHostToDevice));
        seq.resize(s.pos);
        for (auto & o : ck) // later checkpoints belong to the abandoned continuation
            if (o.pos > s.pos) o.pos = -1;
        ring_rewind(s.pos);
        ck_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (memlog) fprintf(stderr, "[mk-mem] checkpoint restore to %d enqueued in %.2f ms\n", s.pos, ck_ms);
    }
    // MK_KV_WIN: the sequence goes back to position P (checkpoint restore, truncate). The abandoned positions P .. kv_top - 1
    // (kv_top: every launch's rows, draft rows included) occupied the ring slots of the window rows P - kvrc .. kv_top - kvrc - 1,
    // so exactly those rows are refilled from the main buffer (they stay at its precision until they age out of the window;
    // a fresh prefill of the same text has them at q8). Positions >= P are rewritten before they are read.
    int kv_top = 0;
    void ring_rewind(int P) {
        if (hb.kvwin > 0) {
            const int p0 = std::max(0, P - hb.kvrc), p1 = std::min(P, kv_top - hb.kvrc);
            if (p1 > p0) {
                k_ring_refill<<<std::min(1024, (p1 - p0) * NKV * N_ATT_SLOTS * 2 / 8 + 1), 256>>>(p0, p1);
                CK(cudaGetLastError());
            }
            if (memlog) fprintf(stderr, "[mk-mem] KV window ring: rewind to %d (rows written to %d), refilled %d rows\n", P, kv_top, std::max(0, p1 - p0));
        }
        kv_top = std::min(kv_top, P);
    }
    int ck_find(int pos) const {
        for (int i = 0; i < (int) ck.size(); i++)
            if (ck[i].pos == pos && pos >= 0) return i;
        return -1;
    }
    // Slots for checkpoints at the new positions `want` (ascending): free ones, else evicting by coverage. While
    // there are more checkpoints (existing + new) than slots, the one dropped is the one (not `keep`; the slot of a
    // restore whose copies may still run only if nothing else is left) whose removal leaves the smallest gap relative
    // to its distance from the end E: (next - prev) / (E - prev). Old checkpoints thin out geometrically, recent ones
    // stay dense. Returns the slot of each new position (-1: dropped); evicted slots become free.
    std::vector<int> ck_assign(const std::vector<int> & want, int keep, int E) {
        struct C { int pos, slot, wi; }; // slot >= 0: existing checkpoint; wi >= 0: new (index into want)
        std::vector<C> c;
        for (int i = 0; i < (int) ck.size(); i++)
            if (ck[i].pos >= 0) c.push_back({ck[i].pos, i, -1});
        for (int i = 0; i < (int) want.size(); i++) c.push_back({want[i], -1, i});
        std::stable_sort(c.begin(), c.end(), [](const C & a, const C & b) { return a.pos < b.pos; });
        while (c.size() > ck.size()) {
            int best = -1;
            double bs = 1e300;
            for (int pass = 0; pass < 2 && best < 0; pass++)
                for (int i = 0; i < (int) c.size(); i++) {
                    if (c[i].wi >= 0 && c[i].pos == keep) continue;
                    if (!pass && c[i].slot >= 0 && (c[i].slot == ck_rs_src || c[i].slot == ck_rs_slot)) continue;
                    const double prev = i ? c[i - 1].pos : 0, next = i + 1 < (int) c.size() ? c[i + 1].pos : E;
                    const double sc = (next - prev) / std::max(1.0, E - prev);
                    if (sc < bs) bs = sc, best = i;
                }
            if (best < 0) break;
            c.erase(c.begin() + best);
        }
        std::vector<char> used(ck.size(), 0);
        for (const auto & x : c)
            if (x.slot >= 0) used[x.slot] = 1;
        for (int i = 0; i < (int) ck.size(); i++)
            if (!used[i]) ck[i].pos = -1;
        std::vector<int> out(want.size(), -1);
        int f = 0;
        for (const auto & x : c)
            if (x.wi >= 0) {
                while (f < (int) ck.size() && used[f]) f++;
                if (f == (int) ck.size()) break;
                used[f] = 1;
                out[x.wi] = f;
            }
        return out;
    }
    // staging slot for one DeltaNet layer's piece of an in-prefill checkpoint (stream 0 waits for its previous copy)
    float * stg_state(int s) { return hp.xm + (size_t) s * STG_F; }
    float * stg_conv(int s) { return stg_state(s) + CK_SL; }
    int stg_take() {
        const int s = stg_next;
        stg_next = (stg_next + 1) % stg_n;
        if (stg_busy[s]) {
            CK(cudaStreamWaitEvent(0, stg_ev[s], 0));
            stg_busy[s] = false;
        }
        return s;
    }
    // copies DeltaNet layer dl's staged pieces (staging slots sg[j] -> checkpoint slots ss[j]) on ck_sv
    void stg_flush(int dl, int sn, const int * sg, const int * ss) {
        CK(cudaEventRecord(ck_ev_sv, 0));
        CK(cudaStreamWaitEvent(ck_sv, ck_ev_sv, 0));
        const cudaMemcpyKind k = ck_dev ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost;
        for (int j = 0; j < sn; j++) {
            CkSlot & s = ck[ss[j]];
            if (ss[j] == ck_rs_src) CK(cudaStreamWaitEvent(ck_sv, ck_ev_grp[CK_NG - 1], 0)); // (still being restored from)
            CK(cudaMemcpyAsync(s.dstate + (size_t) dl * CK_SL, stg_state(sg[j]), CK_SL * 4, k, ck_sv));
            CK(cudaMemcpyAsync(s.conv + (size_t) dl * CK_CL, stg_conv(sg[j]), CK_CL * 4, k, ck_sv));
            CK(cudaEventRecord(stg_ev[sg[j]], ck_sv));
            stg_busy[sg[j]] = true;
            stg_last = sg[j];
        }
    }
    // stream 0, before hp.xm is used again (a chunk's MTP part): every staged copy has finished (ck_sv is in order)
    void stg_drain() {
        if (stg_last < 0) return;
        CK(cudaStreamWaitEvent(0, stg_ev[stg_last], 0));
        for (bool & b : stg_busy) b = false;
        stg_last = -1;
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
            if (e != cudaErrorNotReady) mk_cuda_fail(e, (std::string("(the GPU work of ") + what + ")").c_str(), __FILE__, __LINE__);
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
    // MK_MTP_Q4: the MTP draft block's (blk.64 + nextn.eh_proj) matrices stored wider than Q4_K (Q5_K, Q6_K, Q8_0;
    // UD-Q4_K_XL: all of them, Q4_K_M: eh, v, ffn_down) are re-quantised to Q4_K at load (ggml_quantize_chunk, no
    // imatrix). Lossy, but only the drafts can change: they are verified by the exact main model, so the output (greedy
    // ids, the sampled distribution) is unchanged and only acceptance moves. Each MTP pass streams ~0.1 GB less.
    // 0 off, 1 on (default), ab: both copies on the device and the policy key mtpq=0|1 picks one (A/B tests).
    int mtp_q4 = 1;
    bool req_q4 = false; // want(): re-quantise the current tensor to Q4_K (set while the MTP block loads)
    int n_req = 0;
    double req_s = 0;
    struct MtpW { LayerW L; Mat eh; };
    MtpW mtp_set[2]{}; // MK_MTP_Q4=ab: [0] as stored in the file, [1] re-quantised
    void set_mtp_q4(int on) {
        if (mtp_q4 != 2) throw std::runtime_error("policy mtpq needs MK_MTP_Q4=ab");
        hw.L[NL] = mtp_set[on ? 1 : 0].L;
        hw.eh = mtp_set[on ? 1 : 0].eh;
        CK(cudaMemcpyToSymbol(cW, &hw, sizeof(ModelW)));
    }

    // MK_REENC3=1: Q3_K and IQ3_S re-encoded to Q6_K at load (the path before native Q3_K / IQ3_S; A/B tests)
    const bool reenc3 = getenv("MK_REENC3") && atoi(getenv("MK_REENC3")) != 0;
    int n_nat = 0; // Q3_K / IQ3_S tensors repacked to their native layouts
    double nat_s = 0;
    bool iq3s_grid_up = false;

    Src want(const std::string & name, int to = -1) {
        const auto & t = file.get(name);
        Src s;
        s.type = t.type;
        s.ne = t.ne;
        s.data = t.data;
        s.nbytes = t.nbytes;
        const bool to_f32 = to == T_F32 && t.type != gguf::F32;
        const bool to_q4 = !to_f32 && req_q4 && (t.type == gguf::Q5_K || t.type == gguf::Q6_K || t.type == gguf::Q8_0);
        const bool to_q6 = !to_f32 && !to_q4 && reenc3 && (t.type == gguf::Q3_K || t.type == gguf::IQ3_S);
        const bool to_q8 = !to_f32 && !to_q4 && t.type == gguf::IQ4_NL && getenv("MK_IQ4NL_Q8"); // native IQ4_NL unless MK_IQ4NL_Q8
        if (!to_f32 && !to_q4 && !to_q6 && !to_q8) return s; // anything the kernels cannot read is refused by up_mat / need()
        const ggml_type from = (ggml_type) t.type;
        const ggml_type dst = to_f32 ? GGML_TYPE_F32 : to_q4 ? GGML_TYPE_Q4_K : to_q6 ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
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
            if (to_q4) { // lossy by design (see mtp_q4)
                ggml_quantize_chunk(GGML_TYPE_Q4_K, f.data(), out, 0, 1, K, nullptr);
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
        const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (to_q4) {
            n_req++;
            req_s += dt;
        } else {
            n_conv++;
            conv_s += dt;
        }
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
#if MK_WV
            if (wv_lay) {
                std::vector<uint8_t> w(t.data, t.data + t.nbytes), meta;
                wv_relay(m, w, meta);
                CK(cudaMemcpy(d, w.data(), w.size(), cudaMemcpyHostToDevice));
            } else
#endif
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
#if MK_MAXT > 5
            if (wv_lay && wv_fd) wv_relay(m, q, meta); // main-layer ffn_down: tile-major values and meta pairs
#endif
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
#if MK_WV
            if (wv_lay) {
                std::vector<uint8_t> meta;
                wv_relay(m, q, meta);
            }
#endif
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            m.w = dq;
            wbytes += q.size();
        } else if (t.type == gguf::Q3_K || t.type == gguf::IQ3_S) {
            // native (rows_q3k / rows_iq3s, prefill TQ3K / TIQ3S): values 96 B per block + meta (Q3_K: the Q6_K meta
            // stream, 18 B; IQ3_S: 16 B), checked bit for bit against ggml's dequantization of every row
            const bool q3 = t.type == gguf::Q3_K;
            m.type = q3 ? T_Q3K : T_IQ3S;
            if (nb & 1) throw std::runtime_error(name + ": Q3_K / IQ3_S rows need an even number of 256-blocks");
            const int mb = q3 ? Q6_META : IQ3S_META;
            m.rb = nb * Q3_BLK;
            m.nd = nb * mb / 2; // halves
            const auto t0 = std::chrono::steady_clock::now();
            std::vector<uint32_t> grid;
            if (!q3) {
                grid = iq3s_grid_host();
                if (!iq3s_grid_up) {
                    CK(cudaMemcpyToSymbol(g_iq3s_grid, grid.data(), grid.size() * 4));
                    iq3s_grid_up = true;
                }
            }
            std::vector<uint8_t> q((size_t) N * m.rb), meta((size_t) N * nb * mb);
            const auto * tr = ggml_get_type_traits((ggml_type) t.type);
            std::atomic<bool> exact{true};
            parallel_rows(N, [&](int64_t r) {
                thread_local std::vector<float> f;
                f.resize((size_t) K);
                const uint8_t * src = t.data + (size_t) r * nb * 110;
                tr->to_float(src, f.data(), K);
                uint8_t * qo = q.data() + (size_t) r * m.rb;
                uint8_t * mo = meta.data() + (size_t) r * nb * mb;
                if (!(q3 ? repack_q3k(src, qo, mo, nb, f.data()) : repack_iq3s(src, qo, mo, nb, f.data(), grid))) exact = false;
            });
            if (!exact) throw std::runtime_error(name + ": native Q3_K / IQ3_S repack is not exact");
#if MK_MAXT > 5
            if (wv_lay && wv_fd) wv_relay(m, q, meta); // main-layer ffn_down: tile-major values and meta
#endif
            uint8_t * dq = dalloc<uint8_t>(q.size(), false);
            CK(cudaMemcpy(dq, q.data(), q.size(), cudaMemcpyHostToDevice));
            uint8_t * dm = dalloc<uint8_t>(meta.size(), false);
            CK(cudaMemcpy(dm, meta.data(), meta.size(), cudaMemcpyHostToDevice));
            m.w = dq;
            m.d = reinterpret_cast<const __half *>(dm);
            wbytes += q.size() + meta.size();
            n_nat++;
            nat_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        } else if (t.type == gguf::IQ4_NL) {
            m.type = T_IQ4NL;
            m.rb = nb * 144;
            std::vector<uint8_t> q((size_t) N * m.rb);
            parallel_rows(N, [&](int64_t r) { repack_iq4nl(t.data + (size_t) r * nb * 144, q.data() + (size_t) r * m.rb, nb); });
#if MK_WV
            if (wv_lay) {
                std::vector<uint8_t> meta;
                wv_relay(m, q, meta);
            }
#endif
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
#if MK_WV
            if (wv_lay) {
                std::vector<uint8_t> meta(ds.size() * 2);
                memcpy(meta.data(), ds.data(), meta.size());
                wv_relay(m, q, meta);
                memcpy(ds.data(), meta.data(), meta.size());
            }
#endif
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

#if MK_WV
    // the matrices of the main layers (set while they load) of the wv types are re-laid-out on the host before the upload
    // (MK_MAXT > 5: wv_fd while a main layer's ffn_down loads: its Q6_K / Q3_K / IQ3_S too)
    bool wv_lay = false, wv_fd = false;
    int wv_n = 0;
    double wv_s = 0;
    void wv_relay(Mat & m, std::vector<uint8_t> & w, std::vector<uint8_t> & meta) {
        const auto t0 = std::chrono::steady_clock::now();
        wv_relayout(m.type, m.N, m.K, m.rb, w, meta);
        m.fm = 1;
        wv_n++;
        wv_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
#endif

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
        auto needq = [&](const std::string & name, int got) {
            need(name, got, {T_Q4K, T_Q5K, T_Q6K, T_Q8_0, T_IQ4X, T_IQ4NL, T_Q3K, T_IQ3S});
        };
        if (const char * e = getenv("MK_MTP_Q4")) {
            if (!strcmp(e, "ab")) mtp_q4 = 2;
            else if (!strcmp(e, "0") || !strcmp(e, "1")) mtp_q4 = e[0] - '0';
            else throw std::runtime_error("MK_MTP_Q4: 0, 1 or ab");
        }
        for (int i = 0; i < NLA; i++) {
            LayerW & L = hw.L[i];
            const bool mtp = i == NL;
            req_q4 = mtp && mtp_q4 != 0;
#if MK_WV
            wv_lay = !mtp;
#endif
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
#if MK_MAXT > 5
            wv_fd = true;
#endif
            L.fd = up_mat(blk(i, "ffn_down.weight"), FF, D);
#if MK_MAXT > 5
            wv_fd = false;
            if (!mtp && !L.fd.fm) throw std::runtime_error(blk(i, "ffn_down.weight") + ": MK_MAXT > 5 needs every ffn_down in the wv layout");
#endif
            needq(blk(i, "ffn_gate.weight"), L.fg.type);
            needq(blk(i, "ffn_up.weight"), L.fu.type);
            needq(blk(i, "ffn_down.weight"), L.fd.type);
        }
        mem_mark("weights: 64 layers + MTP");
#if MK_WV
        wv_lay = false;
        fprintf(stderr, "[mk] MK_WV: %d main-pass matrices in the wv layout (re-laid-out in %.1f s; %s), MAXT %d\n", wv_n, wv_s,
                MAXT > 5 ? "ffn_down: every type; elsewhere Q6_K / Q3_K / IQ3_S stay on dp4a" : "Q6_K / Q3_K / IQ3_S stay on dp4a", MAXT);
#endif
        req_q4 = mtp_q4 != 0;
        hw.eh = up_mat(blk(NL, "nextn.eh_proj.weight"), 2 * D, D);
        req_q4 = false;
        needq(blk(NL, "nextn.eh_proj.weight"), hw.eh.type);
        if (mtp_q4 == 2) { // A/B: the file's MTP matrices as well (the norms are shared)
            mtp_set[1] = {hw.L[NL], hw.eh};
            MtpW & a = mtp_set[0];
            a.L = hw.L[NL];
            a.L.q = up_mat(blk(NL, "attn_q.weight"), D, QG);
            a.L.k = up_mat(blk(NL, "attn_k.weight"), D, KVD);
            a.L.v = up_mat(blk(NL, "attn_v.weight"), D, KVD);
            a.L.o = up_mat(blk(NL, "attn_output.weight"), VD, D);
            a.L.fg = up_mat(blk(NL, "ffn_gate.weight"), D, FF);
            a.L.fu = up_mat(blk(NL, "ffn_up.weight"), D, FF);
            a.L.fd = up_mat(blk(NL, "ffn_down.weight"), FF, D);
            a.eh = up_mat(blk(NL, "nextn.eh_proj.weight"), 2 * D, D);
            mem_mark("MTP block, file types (MK_MTP_Q4=ab)");
        }
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
            fprintf(stderr, "[mk] %d tensors re-encoded in %.1f s (exact: Q3_K / IQ3_S -> Q6_K with MK_REENC3, IQ4_NL -> Q8_0 with MK_IQ4NL_Q8; ssm_alpha / beta -> F32)\n",
                    n_conv, conv_s);
        if (n_nat)
            fprintf(stderr, "[mk] %d Q3_K / IQ3_S tensors repacked to their native layouts in %.1f s (exact, checked per row)\n", n_nat, nat_s);
        if (n_req)
            fprintf(stderr, "[mk] MTP block: %d matrices re-quantised to Q4_K in %.1f s (drafts only; MK_MTP_Q4=0: as stored%s)\n",
                    n_req, req_s, mtp_q4 == 2 ? "; ab: both copies, policy mtpq=0|1" : "");

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
            hb.dattn = !(getenv("MK_DATTN") && atoi(getenv("MK_DATTN")) == 0);
            // MK_MTP_WIN=N: the MTP block's decode attention sees the last N positions only (its prefill attention and
            // the KV layout are unchanged). Drafts are verified, so the output is exact; acceptance can change.
            hb.mtp_win = getenv("MK_MTP_WIN") ? atoi(getenv("MK_MTP_WIN")) : 0;
            if (hb.mtp_win > 0) fprintf(stderr, "[mk] MTP decode attention window %d positions (MK_MTP_WIN)\n", hb.mtp_win);
            // MK_KV (q8 cache): per attention slot and K / V, q8_0 (default), q4 (kv_store_q4) or q5 / q6 / q7
            // (kv_store_qn), e.g. "V*=6r,K16=4r": comma-separated <K|V|*><slot>=<4..8>[r], slot 0..15 = attention layer
            // 4 slot + 3, 16 = MTP block, * or empty = all slots; r: stored Hadamard-rotated (K: the queries get the same
            // rotation; V: the head's attention output is rotated back)
            const size_t srows = (size_t) NKV * ctx; // rows of one slot
            for (int s = 0; s < N_ATT_SLOTS; s++) {
                hb.kvb[0][s] = hb.kvb[1][s] = 8;
                hb.kvr[s] = hb.kvrv[s] = 0;
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
                    if (md.empty() || md.size() > 2 || md[0] < '4' || md[0] > '8' || (md.size() == 2 && md[1] != 'r'))
                        throw std::runtime_error("MK_KV: expected 4..8 or 4r..8r in " + tok);
                    const bool rot = md.size() == 2;
                    const bool all = sl.empty() || sl == "*";
                    const int slot = all ? -1 : std::stoi(sl);
                    if (!all && (slot < 0 || slot >= N_ATT_SLOTS)) throw std::runtime_error("MK_KV: bad slot in " + tok);
                    for (int w = 0; w < 2; w++)
                        if (tok[0] == '*' || tok[0] == "KV"[w])
                            for (int s = 0; s < N_ATT_SLOTS; s++)
                                if (all || s == slot) {
                                    hb.kvb[w][s] = md[0] - '0';
                                    (w == 0 ? hb.kvr : hb.kvrv)[s] = rot;
                                }
                }
            }
            size_t vbytes[2] = {0, 0};
            int n4 = 0;
            for (int w = 0; w < 2; w++)
                for (int s = 0; s < N_ATT_SLOTS; s++) {
                    vbytes[w] += srows * HD * hb.kvb[w][s] / 8;
                    n4 += hb.kvb[w][s] != 8 || (w == 0 ? hb.kvr[s] : hb.kvrv[s]);
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
            // MK_KV_WIN=W: every low-bit side keeps its last >= W positions as q8 in a ring (kv_ws / kv_src; the ring holds
            // (ceil(W / ATT_BUCKET) + 1) ATT_BUCKET rows per kv head so a bucket-aligned window always fits)
            hb.kvwin = 0;
            hb.kvrc = 0;
            for (int w = 0; w < 2; w++)
                for (int s = 0; s < N_ATT_SLOTS; s++) {
                    hb.kvrp[w][s] = nullptr;
                    hb.kvrs[w][s] = nullptr;
                }
            size_t ring_bytes = 0;
            if (want_kv && getenv("MK_KV_WIN") && atoi(getenv("MK_KV_WIN")) > 0) {
                hb.kvwin = atoi(getenv("MK_KV_WIN"));
                hb.kvrc = ((hb.kvwin + ATT_BUCKET - 1) / ATT_BUCKET + 1) * ATT_BUCKET;
                // MK_KV_WIN_ALL=1 (test knob): rings for the q8 sides too. Their ring rows are the same q8 rows as the main
                // buffer (and k_ring_refill re-stores a q8 row exactly), so output must stay byte-identical to no window:
                // exercises the ring write / read / refill paths with an exact expectation.
                const bool win_all = getenv("MK_KV_WIN_ALL") && atoi(getenv("MK_KV_WIN_ALL")) != 0;
                int nr = 0;
                for (int w = 0; w < 2; w++)
                    for (int s = 0; s < N_ATT_SLOTS; s++)
                        if (hb.kvb[w][s] != 8 || win_all) {
                            hb.kvrp[w][s] = dalloc<int8_t>((size_t) NKV * hb.kvrc * HD);
                            hb.kvrs[w][s] = dalloc<__half>((size_t) NKV * hb.kvrc * (HD / 32));
                            ring_bytes += (size_t) NKV * hb.kvrc * (HD + HD / 16);
                            nr++;
                        }
                fprintf(stderr, "[mk] KV recent window: the last %d..%d positions of %d low-bit sides stay q8 (MK_KV_WIN=%d; rings of %d rows per head, %.1f MiB)\n",
                        hb.kvrc - ATT_BUCKET + 1, hb.kvrc, nr, hb.kvwin, hb.kvrc, ring_bytes / 1048576.0);
            }
            const double gb = q8 ? (vbytes[0] + vbytes[1] + 2.0 * rows * (HD / 32) * 2 + ring_bytes) / 1e9 : rows * HD * 4.0 / 1e9;
            fprintf(stderr, "[mk] KV cache %s, %.2f GB for %d positions\n", q8 ? (n4 ? "q8_0 + q4 (MK_KV)" : "q8_0") : "fp16", gb, ctx);
            if (n4) {
                std::string k, v, kr, vr;
                for (int s = 0; s < N_ATT_SLOTS; s++) {
                    k += (char) ('0' + hb.kvb[0][s]);
                    v += (char) ('0' + hb.kvb[1][s]);
                    kr += hb.kvr[s] ? 'r' : '.';
                    vr += hb.kvrv[s] ? 'r' : '.';
                }
                fprintf(stderr, "[mk] MK_KV bits per slot (layers 3, 7, .., 63, MTP): K %s  V %s; rotated: K %s  V %s\n", k.c_str(), v.c_str(), kr.c_str(), vr.c_str());
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
        hb.pcnt = nullptr;   // 0.5 MB for the presence / frequency penalty, allocated by the first set_penalty
        hb.bar = dalloc<unsigned>(4);
        hb.cnt = dalloc<unsigned>(4);
        hb.ctl = dalloc<Ctl>(1);
        hb.ctx = ctx;
#if MK_WV
        hb.wvpart = dalloc<float>((size_t) WV_S * MAXT * D);
        hb.wvcnt = dalloc<unsigned>(2 * (D / 32));
#endif
        profiling = getenv("MK_PROFILE") != nullptr;
        hb.prof = profiling ? dalloc<unsigned long long>(4096 + 1024 * 128) : nullptr;
        mem_mark("logits, sampling, ctl");
        if (const char * spec = getenv("MK_KV_EMU_SPEC")) kv_emu_parse(spec);
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
            ck_wait_restore();
        } catch (const std::exception & e) {
            fprintf(stderr, "[mk] teardown: %s\n", e.what());
        }
        ck_free(); // (checkpoint copies may still read or write device buffers)
        for (void * p : allocs) use_pool ? cudaFreeAsync(p, 0) : cudaFree(p);
        cudaStreamSynchronize(0);
        if (embd_host) cudaFreeHost(embd_host);
    }

    void launch(int mode, int T, int last, int score = 0) {
        kv_top = std::max(kv_top, hctl.n + T + ndraft); // rows hctl.n .. + T - 1, the MTP's draft rows after them (ring_rewind)
        Launch L{mode, T, last, ndraft, score, smp_temp, smp_top_p, smp_min_p, smp_top_k, smp_seed};
        for (int k = 0; k < MAXT - 1; k++) {
            L.gate_drop[k] = pol.gate_drop[k];
            L.gate_stop[k] = pol.gate_stop[k];
        }
#if MK_MAXT > 5
        const int top = depth_top(hctl.n); // drop_last: the deepest draft of this position's policy depth
        if (top >= 1) L.gate_drop[top - 1] = std::max(L.gate_drop[top - 1], pol.drop_last);
#endif
        L.gate_chain = pol.gate_chain;
        L.spec_smp = smp_temp > 0.f && pol.h1;
        if (!score && (pen_pres != 0.f || pen_freq != 0.f)) {
            L.pres = pen_pres;
            L.freq = pen_freq;
            pcnt_dirty = true;
        }
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
        mmh_attr<pf::TQ3K, false>();
        mmh_attr<pf::TIQ3S, false>();
        mmh_attr<pf::TQ4K, true>();
        mmh_attr<pf::TQ5K, true>();
        mmh_attr<pf::TIQ4, true>();
#if MK_WV
        mmh_attr<pf::TQ4K, false, 1>();
        mmh_attr<pf::TQ5K, false, 1>();
        mmh_attr<pf::TQ80, false, 1>();
        mmh_attr<pf::TIQ4, false, 1>();
        mmh_attr<pf::TIQ4NL, false, 1>();
        mmh_attr<pf::TQ4K, true, 1>();
        mmh_attr<pf::TQ5K, true, 1>();
        mmh_attr<pf::TIQ4, true, 1>();
#if MK_MAXT > 5
        mmh_attr<pf::TQ6K, false, 1>();
        mmh_attr<pf::TQ3K, false, 1>();
        mmh_attr<pf::TIQ3S, false, 1>();
#endif
#endif
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
        CK(cudaFuncSetAttribute(pk_attn8<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
        CK(cudaFuncSetAttribute(pk_attn8<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
        CK(cudaFuncSetAttribute(pk_attn8<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
        CK(cudaFuncSetAttribute(pk_attn8<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
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
            CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ8, pk_attn8<false, false>, A2_NT, A8_SMEM));
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
            // pk_attn8 reads K in any MK_KV format (low-bit tiles are expanded to int8 in its staging) and any V format
            // (a8_cv_v converts). Rotated K too: pk_attn_prep rotates Q (fp32 fwht256) before its fp16 store to qh, so the
            // int8 Q fragments are rotated as well; pk_attn_out un-rotates a rotated V's output for every kernel. MK_ATTN_S8=0
            // keeps pk_attn2 (fp16 Q) for comparison.
            const bool s8 = attn_pv16;
            const bool kl = hb.kvb[0][slot] != 8; // low-bit K: the expanding instantiation (q8 K: the plain kernel)
            if (s8 && attn_s8 == 1) {
                if (kl) pk_attn8<false, true><<<grid, A2_NT, A8_SMEM>>>(T, n, slot, nfull, S, hp.y1);
                else pk_attn8<false, false><<<grid, A2_NT, A8_SMEM>>>(T, n, slot, nfull, S, hp.y1);
            } else if (s8 && attn_s8 == 2) {
                if (kl) pk_attn8<true, true><<<grid, A2_NT, A8_SMEM>>>(T, n, slot, nfull, S, hp.y1);
                else pk_attn8<true, false><<<grid, A2_NT, A8_SMEM>>>(T, n, slot, nfull, S, hp.y1);
            }
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
    template <int TYPE, bool GLU, int LAY = 0> void mmh_attr() {
        CK(cudaFuncSetAttribute(pf::mmh<TYPE, GLU, 32, LAY>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<TYPE, 32>()));
        CK(cudaFuncSetAttribute(pf::mmh<TYPE, GLU, 64, LAY>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<TYPE, 64>()));
        CK(cudaFuncSetAttribute(pf::mmh<TYPE, GLU, 128, LAY>, cudaFuncAttributeMaxDynamicSharedMemorySize, pf::hm_smem<TYPE, 128>()));
    }
    // nrt row tiles (CTAs along N) x token tiles of pf_nb tokens; LAY 1: the matrix is in the wv layout (MK_WV)
    template <int TYPE, bool GLU, int LAY = 0>
    void mmh_launch(int nrt, const pf::WMat & W, const __half * X, int ntok, const pf::Out & O, const uint8_t * W2 = nullptr) {
        switch (pf_nb(ntok, nrt)) {
            case 32: pf::mmh<TYPE, GLU, 32, LAY><<<dim3(nrt, (ntok + 31) / 32), 256, pf::hm_smem<TYPE, 32>()>>>(W, X, ntok, O, W2); break;
            case 64: pf::mmh<TYPE, GLU, 64, LAY><<<dim3(nrt, (ntok + 63) / 64), 256, pf::hm_smem<TYPE, 64>()>>>(W, X, ntok, O, W2); break;
            default: pf::mmh<TYPE, GLU, 128, LAY><<<dim3(nrt, (ntok + 127) / 128), 256, pf::hm_smem<TYPE, 128>()>>>(W, X, ntok, O, W2); break;
        }
    }

    // y[t][row] (+)= W x_t for the first ntok fp16 rows of xh (token offset t0)
    void gemm(const Mat & m, int ntok, float * y, int ldy, int mode, int t0 = 0, const __half * xin = nullptr) {
        const pf::WMat W{m.w, m.d, m.type, m.K, m.N, m.rb, m.nd};
        const __half * X = (xin ? xin : hp.xh) + (size_t) t0 * m.K;
        const pf::Out O{y, ldy, mode, nullptr};
        const int nrt = m.N / pf::BM;
#if MK_WV
        if (m.fm) {
            switch (m.type) {
                case T_Q4K: mmh_launch<pf::TQ4K, false, 1>(nrt, W, X, ntok, O); return;
                case T_Q5K: mmh_launch<pf::TQ5K, false, 1>(nrt, W, X, ntok, O); return;
                case T_Q8_0: mmh_launch<pf::TQ80, false, 1>(nrt, W, X, ntok, O); return;
                case T_IQ4X: mmh_launch<pf::TIQ4, false, 1>(nrt, W, X, ntok, O); return;
                case T_IQ4NL: mmh_launch<pf::TIQ4NL, false, 1>(nrt, W, X, ntok, O); return;
#if MK_MAXT > 5
                case T_Q6K: mmh_launch<pf::TQ6K, false, 1>(nrt, W, X, ntok, O); return;
                case T_Q3K: mmh_launch<pf::TQ3K, false, 1>(nrt, W, X, ntok, O); return;
                case T_IQ3S: mmh_launch<pf::TIQ3S, false, 1>(nrt, W, X, ntok, O); return;
#endif
                default: throw std::runtime_error("gemm: unsupported weight type in the wv layout");
            }
        }
#endif
        switch (m.type) {
            case T_Q4K: mmh_launch<pf::TQ4K, false>(nrt, W, X, ntok, O); break;
            case T_Q5K: mmh_launch<pf::TQ5K, false>(nrt, W, X, ntok, O); break;
            case T_Q6K: mmh_launch<pf::TQ6K, false>(nrt, W, X, ntok, O); break;
            case T_Q8_0: mmh_launch<pf::TQ80, false>(nrt, W, X, ntok, O); break;
            case T_IQ4X: mmh_launch<pf::TIQ4, false>(nrt, W, X, ntok, O); break;
            case T_IQ4NL: mmh_launch<pf::TIQ4NL, false>(nrt, W, X, ntok, O); break;
            case T_Q3K: mmh_launch<pf::TQ3K, false>(nrt, W, X, ntok, O); break;
            case T_IQ3S: mmh_launch<pf::TIQ3S, false>(nrt, W, X, ntok, O); break;
            default: throw std::runtime_error("gemm: unsupported weight type");
        }
    }

    // xh2[t] = silu(Wg x_t) * (Wu x_t) for the first ntok rows of xh: one fused GEMM when gate and
    // up share a type with in-row scales, else two GEMMs into y0 / y1 and pk_glu
    static bool glu_fused(const Mat & g, const Mat & u) {
#if MK_WV
        if (g.fm != u.fm) return false;
#endif
        return g.type == u.type && g.rb == u.rb && (g.type == T_Q4K || g.type == T_Q5K || g.type == T_IQ4X);
    }
    void gemm_glu(const Mat & g, const Mat & u, int ntok) {
        const pf::WMat W{g.w, g.d, g.type, g.K, g.N, g.rb, g.nd};
        const pf::Out O{nullptr, g.N, pf::OUT_STORE, hp.xh2};
        const int nrt = g.N / (pf::BM / 2);
#if MK_WV
        if (glu_fused(g, u) && g.fm) {
            switch (g.type) {
                case T_Q4K: mmh_launch<pf::TQ4K, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_Q5K: mmh_launch<pf::TQ5K, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
                case T_IQ4X: mmh_launch<pf::TIQ4, true, 1>(nrt, W, hp.xh, ntok, O, u.w); return;
                default: break;
            }
        }
#endif
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
    // snaps: checkpoints to save on the way (positions n0 < P <= n0 + toks.size(), ascending, slots from ck_assign;
    // at most stg_n per chunk: ck_positions); complete when pf_run returns.
    void pf_run(const std::vector<int> & toks, int n0, std::vector<int> * pred, const MKEngine::LogitSink * sink = nullptr,
                int first = 0, const std::vector<Snap> * snaps = nullptr) {
        pf_init();
        kv_top = std::max(kv_top, n0 + (int) toks.size()); // (ring_rewind)
        ck_scr_guard(); // the scratch may still be the source of an end save's copy
        ck_guard_state();
        const int N = (int) toks.size(), cap = hp.cap;
        for (int c0 = 0; c0 < N; c0 += cap) {
#if MK_KV_EMU
            kv_demote_to(n0 + c0 - hb.kvemu_win); // recent-window emulation: rows older than the window before this chunk
#endif
            const int T = std::min(cap, N - c0), n = n0 + c0;
            int sn = 0, sp[STG_MAX], ss[STG_MAX]; // this chunk's checkpoints (n < P <= n + T) and their slots
            if (snaps && !pred && !sink)
                for (const Snap & q : *snaps)
                    if (q.pos > n && q.pos <= n + T && sn < stg_n) sp[sn] = q.pos, ss[sn++] = q.slot;
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
                    pf_time("attn_out", [&] { pk_attn_out<<<T, 256>>>(L / 4); });
                    pf_time("gemm_attn_o", [&] { gemm(W.o, T, hp.x, D, pf::OUT_ADD); });
                } else {
                    pf_time("gemm_dn_in", [&] { gemm(W.qkv, T, hp.y0, QKV, pf::OUT_STORE); gemm(W.gate, T, hp.y1, VD, pf::OUT_STORE); });
                    pf_time("dn_ab", [&] { pk_ab<<<dim3((T + 63) / 64, AB_SPLITS), 256>>>(W.ab, W.abq, W.abd, W.attn_norm, T); });
                    ck_wait_restore(dl); // this layer's conv ring / state may still be arriving from the checkpoint
                    int sg[STG_MAX], ta = -1; // staging slots of this layer's checkpoint pieces (+ a side scan's input)
                    for (int j = 0; j < sn; j++) sg[j] = stg_take();
                    for (int j = 0; j < sn; j++)
                        if ((sp[j] - n) % 32 && sp[j] - n > 32) ta = stg_take();
                    pf_time("dn_prep", [&] {
                        pk_dn_prep<<<T, 256>>>(T, n, dl, W.conv, W.dt, W.A);
                        for (int j = 0; j < sn; j++) pk_ck_conv<<<CONV_RING, 256>>>(n, sp[j], dl, stg_conv(sg[j]));
                        pk_conv_ring<<<std::min(T, CONV_RING), 256>>>(T, n, dl);
                    });
                    pf_time("dn_scan", [&] {
                        // The scan is split only at points of its 32-token grid, which changes no arithmetic: the state
                        // at an on-grid checkpoint goes to its staging slot on the way. An off-grid one (the exact
                        // ck_at) is computed on the side: a scan of the rows from the grid point before it, from the
                        // state there, into its staging slot (= what a prefill ending at it computes); the main scan
                        // then rewrites those rows' y2.
                        float * live = hb.dstate + (size_t) dl * CK_SL;
                        const float * in = live;
                        int t = 0;
                        for (int j = 0; j < sn; j++) {
                            const int te = sp[j] - n, tg = te / 32 * 32;
                            if (te == tg) {
                                pk_dn_scan<<<dim3(NVH, 2), 256>>>(tg - t, t, in, stg_state(sg[j]));
                                in = stg_state(sg[j]);
                                t = tg;
                                continue;
                            }
                            if (tg > t) {
                                pk_dn_scan<<<dim3(NVH, 2), 256>>>(tg - t, t, in, stg_state(ta));
                                in = stg_state(ta);
                                t = tg;
                            }
                            pk_dn_scan<<<dim3(NVH, 2), 256>>>(te - tg, tg, in, stg_state(sg[j]));
                        }
                        pk_dn_scan<<<dim3(NVH, 2), 256>>>(T - t, t, in, live);
                    });
                    if (sn) stg_flush(dl, sn, sg, ss);
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
            stg_drain(); // (the MTP part writes hp.xm)
            for (int j = 0; j < sn; j++) { // h ring at each checkpoint: positions before the chunk, from the ring now
                if (ck[ss[j]].pending) CK(cudaStreamWaitEvent(0, ck[ss[j]].ev, 0)); // (an earlier save into the slot)
                if (sp[j] - H_RING < n) pk_ck_hist<<<n - (sp[j] - H_RING), 256>>>(sp[j] - H_RING, n, 1, ck[ss[j]].hdev);
            }
            pf_time("mtp", [&] {
                pk_final_h<<<T, 256>>>(T, n);
                for (int j = 0; j < sn; j++) { // ... and positions of the chunk, from its h rows
                    const int lo = std::max(n, sp[j] - H_RING);
                    pk_ck_hist<<<sp[j] - lo, 256>>>(lo, n, 0, ck[ss[j]].hdev);
                }
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
        if (snaps && !pred && !sink)
            for (const Snap & q : *snaps) { // (its last piece was copied before the last chunk's MTP part)
                CK(cudaEventRecord(ck[q.slot].ev, ck_sv));
                ck[q.slot].pending = true;
            }
        CK(cudaGetLastError());
        wait_device("the prefill tail", std::max(wd_limit, 120.0));
        if (memlog && pf_runs_seen++ == 0) mem_mark("first prefill run");
        pf_report();
    }

    // Checkpoint positions for a batched part n0 .. hend - 1 (pf_run chunks of hp.cap from n0): the requested P with
    // n0 < P <= hend, `exact` as it is (saved by a side scan), every other one moved down onto the DeltaNet scan's
    // 32-token grid from its chunk's start; ascending, without duplicates or existing checkpoints, at most stg_n per
    // chunk (one less with `exact`, whose side scan may need a staging slot; dropping the earliest).
    std::vector<int> ck_positions(const std::vector<int> & cks, int exact, int n0, int hend) {
        const int cap = hp.cap;
        auto chunk = [&](int P) { return (P - n0 - 1) / cap; }; // the chunk P belongs to (n < P <= n + T)
        std::vector<int> v;
        for (int P : cks) {
            if (P <= n0 || P > hend) continue;
            if (P != exact) {
                const int cs = n0 + chunk(P) * cap;
                P = cs + (P - cs) / 32 * 32;
                if (P <= n0) continue;
            }
            if (ck_find(P) < 0) v.push_back(P);
        }
        std::sort(v.begin(), v.end());
        v.erase(std::unique(v.begin(), v.end()), v.end());
        for (size_t i = 0; i < v.size();) { // per chunk: at most stg_n staging slots
            size_t j = i;
            while (j < v.size() && chunk(v[j]) == chunk(v[i])) j++;
            const bool ex = std::find(v.begin() + i, v.begin() + j, exact) != v.begin() + j;
            int extra = (int) (j - i) + (ex ? 1 : 0) - stg_n;
            for (size_t k = i; k < j && extra > 0;)
                if (v[k] != exact) v.erase(v.begin() + k), j--, extra--;
                else k++;
            i = j;
        }
        return v;
    }

    // cB.logits ([MAXT][VOCAB] fp32, 5 MB): only scoring launches (Launch::score == 2) write it
    void logits_alloc() {
        if (hb.logits) return;
        hb.logits = dalloc<float>((size_t) MAXT * VOCAB);
        CK(cudaMemcpyToSymbol(cB, &hb.logits, sizeof(hb.logits), offsetof(Bufs, logits)));
        mem_mark("scoring logits");
    }

    // presence / frequency penalty (Launch::pres / freq): the response's token counts (cB.pcnt), zeroed by the next
    // prefill once a penalised launch has counted into them
    float pen_pres = 0.f, pen_freq = 0.f;
    bool pcnt_dirty = false;
    // MK_KV_EMU_SPEC / set_kv_emu (MK_KV_EMU builds): comma-separated <K|V|*><slot>=<bits>[flags] and sink=N. slot
    // 0..15 (attention layer 4 slot + 3), 16 (MTP) or * / empty (all). Flags (kv_store_emu): r Hadamard rotation,
    // s random signs + rotation, a q4_0-style scale, l Lloyd-Max codebook with rms scale (bits 2..6), n one scale per
    // 256-value row (else per 32). "8" without flags = plain q8. sink=N: positions < N stay q8.
    void kv_emu_parse(const std::string & s) {
#if MK_KV_EMU
        for (int w = 0; w < 2; w++)
            for (int q = 0; q < N_ATT_SLOTS; q++) hb.kvemu[w][q] = 0;
        hb.kvemu_sink = 0;
        hb.kvemu_win = 0;
        demoted = 0;
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) j = s.size();
            const std::string tok = s.substr(i, j - i);
            i = j + 1;
            if (tok.empty()) continue;
            if (tok.compare(0, 5, "sink=") == 0) {
                hb.kvemu_sink = std::stoi(tok.substr(5));
                continue;
            }
            if (tok.compare(0, 4, "win=") == 0) {
                hb.kvemu_win = std::stoi(tok.substr(4));
                continue;
            }
            const size_t e = tok.find('=');
            if (e == std::string::npos || e == 0 || !strchr("KV*", tok[0])) throw std::runtime_error("MK_KV_EMU_SPEC: bad item " + tok);
            const std::string sl = tok.substr(1, e - 1), md = tok.substr(e + 1);
            int bits = 0, mode = 0;
            size_t k = 0;
            while (k < md.size() && isdigit((unsigned char) md[k])) bits = 10 * bits + (md[k++] - '0');
            for (; k < md.size(); k++) {
                if (md[k] == 'r') mode |= 16;
                else if (md[k] == 'a') mode |= 32;
                else if (md[k] == 'l') mode |= 64;
                else if (md[k] == 'n') mode |= 128;
                else if (md[k] == 's') mode |= 256 | 16;
                else if (md[k] == 't') mode |= 2048 | 256 | 16;
                else if (md[k] == 'T') mode |= 4096 | 256 | 16;
                else throw std::runtime_error("MK_KV_EMU_SPEC: bad mode " + md);
            }
            if (bits < 2 || bits > 8) throw std::runtime_error("MK_KV_EMU_SPEC: bits must be 2..8 in " + tok);
            if ((mode & 64) && (bits > 6 || (mode & 32))) throw std::runtime_error("MK_KV_EMU_SPEC: l needs 2..6 bits and no a: " + tok);
            if ((mode & (2048 | 4096)) && (mode & (32 | 64 | 128))) throw std::runtime_error("MK_KV_EMU_SPEC: t / T take no a, l, n: " + tok);
            mode |= bits;
            const bool all = sl.empty() || sl == "*";
            const int slot = all ? -1 : std::stoi(sl);
            if (!all && (slot < 0 || slot >= N_ATT_SLOTS)) throw std::runtime_error("MK_KV_EMU_SPEC: bad slot in " + tok);
            for (int w = 0; w < 2; w++)
                if (tok[0] == '*' || tok[0] == "KV"[w])
                    for (int q = 0; q < N_ATT_SLOTS; q++)
                        if (all || q == slot) hb.kvemu[w][q] = mode == 8 ? 0 : mode;
        }
        fprintf(stderr, "[mk] MK_KV_EMU_SPEC %s: K", s.c_str());
        for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[0][q]);
        fprintf(stderr, " | V");
        for (int q = 0; q < N_ATT_SLOTS; q++) fprintf(stderr, " %d", hb.kvemu[1][q]);
        fprintf(stderr, " | sink %d win %d\n", hb.kvemu_sink, hb.kvemu_win);
        if (!hb.kvq8) throw std::runtime_error("MK_KV_EMU_SPEC needs the q8 KV cache (--kv q8)");
        for (int w = 0; w < 2; w++)
            for (int q = 0; q < N_ATT_SLOTS; q++)
                if (hb.kvb[w][q] != 8 || hb.kvr[q] || hb.kvrv[q]) throw std::runtime_error("MK_KV_EMU_SPEC needs an all-q8 cache (no MK_KV)");
#else
        (void) s;
        throw std::runtime_error("MK_KV_EMU_SPEC needs a build with -DMK_KV_EMU=1");
#endif
    }
    void set_kv_emu(const std::string & spec) { // between sequences (no kernel in flight): new spec, cB re-uploaded
        kv_emu_parse(spec);
#if MK_KV_EMU
        CK(cudaDeviceSynchronize());
        CK(cudaMemcpyToSymbol(cB, &hb, sizeof(Bufs)));
#endif
    }
    int demoted = 0; // recent-window emulation: positions < demoted have been demoted to their slot's mode
    void kv_demote_to(int upto) { // demote everything older than `upto` (positions demoted .. upto - 1)
#if MK_KV_EMU
        if (hb.kvemu_win <= 0 || upto <= demoted) return;
        CK(cudaDeviceSynchronize());
        k_kv_demote<<<256, 256>>>(demoted, upto);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        demoted = upto;
#else
        (void) upto;
#endif
    }
    void set_penalty(float pres, float freq) {
        if (!std::isfinite(pres) || !std::isfinite(freq)) throw std::runtime_error("penalties must be finite");
        if ((pres != 0.f || freq != 0.f) && !hb.pcnt) {
            hb.pcnt = dalloc<unsigned short>(VOCAB);
            CK(cudaMemcpyToSymbol(cB, &hb.pcnt, sizeof(hb.pcnt), offsetof(Bufs, pcnt)));
            mem_mark("penalty counts");
        }
        pen_pres = pres;
        pen_freq = freq;
    }
    void pcnt_clear() {
        if (!pcnt_dirty) return;
        CK(cudaMemsetAsync(hb.pcnt, 0, (size_t) VOCAB * sizeof(unsigned short)));
        pcnt_dirty = false;
    }

    void reset() {
        ck_rs_slot = -1;   // (a restore not issued yet is moot)
        ck_wait_restore(); // (an issued one still writes the state)
        ck_guard_state();
        kv_top = 0; // (MK_KV_WIN: every position is rewritten before it is read; the ring needs nothing)
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
#if MK_MAXT > 5
    if (n_draft > 4) { // the wide pass (T >= 6) runs the all-q8 decode attention only (attn_core<T > 5>)
        bool q8 = p_->hb.kvq8;
        for (int s = 0; s < N_ATT_SLOTS; s++) q8 = q8 && p_->hb.kvb[0][s] == 8 && p_->hb.kvb[1][s] == 8;
        if (!q8) {
            delete p_;
            p_ = nullptr;
            throw std::runtime_error("n_draft > 4 needs the all-q8 KV cache (--kv q8, no low-bit MK_KV)");
        }
    }
#endif
    reset();
    if (const char * pol = getenv("MK_POLICY")) set_policy(pol);
#if MK_SPEC_LOG
    if (const char * f = getenv("MK_SPEC_LOG_FILE")) p_->slog = fopen(f, "a");
#endif
}

// "key=value,...": depth=0|1|2 (see Impl::Policy), dmin=N; on-device early stop (Launch::gate_*): drop=F,
// stop=F (every draft), dropK=F / stopK=F (draft K = 1..4 only), chain=0|1; h1=0|1 (sampled drafts when
// sampling); mtpq=0|1 (MK_MTP_Q4=ab only: the MTP block as stored / re-quantised to Q4_K; kept until changed);
// MK_MAXT > 5: dropK / stopK up to K = 7, droplast=F (the deepest draft), dctx=N (from position N on at most 4 drafts);
// "" = defaults
void MKEngine::set_policy(const std::string & spec) {
    Impl::Policy pol;
    float dk[MAXT - 1], sk[MAXT - 1]; // per-draft overrides
    for (int j = 0; j < MAXT - 1; j++) dk[j] = sk[j] = -1.f;
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
                 k[4] < '1' + MAXT - 1) // dropK / stopK: draft K alone (e.g. drop4: the T = 5 verify row)
            (k[0] == 'd' ? dk : sk)[k[4] - '1'] = std::stof(v);
#if MK_MAXT > 5
        else if (k == "droplast") pol.drop_last = std::stof(v); // the deepest draft (k = n_draft - 1)
        else if (k == "dctx") pol.dctx = std::stoi(v); // from position dctx on: at most 4 drafts
#endif
        else if (k == "chain") pol.gate_chain = std::stoi(v);
        else if (k == "h1") pol.h1 = std::stoi(v);
        else if (k == "mtpq") p_->set_mtp_q4(std::stoi(v)); // MK_MTP_Q4=ab only; takes effect from the next prefill
        else throw std::runtime_error("policy: unknown key " + k);
    }
    for (int j = 0; j < MAXT - 1; j++) {
        if (dk[j] >= 0.f) pol.gate_drop[j] = dk[j];
        if (sk[j] >= 0.f) pol.gate_stop[j] = sk[j];
    }
    p_->pol = pol;
}

MKEngine::~MKEngine() { delete p_; }

void MKEngine::bench_phases() { p_->bench(); }

void MKEngine::set_penalty(float presence, float frequency) { p_->set_penalty(presence, frequency); }

void MKEngine::set_kv_emu(const std::string & spec) { p_->set_kv_emu(spec); }

void MKEngine::reset() {
    p_->pcnt_clear();
    p_->reset();
    p_->demoted = 0;
    p_->seq.clear();
    for (auto & k : p_->ck) k.pos = -1;
    p_->dv_reset();
    stats = MKStats();
}

int MKEngine::position() const { return p_->hctl.n; }

int MKEngine::prefill(const std::vector<int> & tokens, int ck_at) {
    return prefill(tokens, ck_at >= 0 ? std::vector<int>{ck_at} : std::vector<int>{}, ck_at);
}

int MKEngine::prefill(const std::vector<int> & tokens, const std::vector<int> & cks, int ck_exact) {
    if (tokens.empty()) throw std::runtime_error("empty prompt");
    p_->pcnt_clear(); // a new response: the penalties count its generated tokens only
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
    bool end_save = true; // a checkpoint at the end of the prompt, unless ck_exact is inside the batched part
    const char * mode = getenv("MK_PREFILL");
    const bool batched = !(mode && std::string(mode) == "mk");
    if (batched && feed.size() >= 2) {
        // all but the last token in batched kernels; the last one goes through the megakernel,
        // which produces the first output token and the MTP drafts
        const std::vector<int> head(feed.begin(), feed.end() - 1);
        const int hend = n0 + (int) head.size(); // the batched part: positions n0 .. hend - 1
        end_save = !(ck_exact > n0 && ck_exact < hend);
        if (p_->stg_n < 2) { // too few staging slots (MK_PF_CAP below ~340): two pf_runs around ck_exact, other positions ignored
            if (!end_save) {
                p_->pf_run(std::vector<int>(head.begin(), head.begin() + (ck_exact - n0)), n0, nullptr);
                c.dn_c = ck_exact;
                c.mtp_valid = c.dn_c;
                CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
                const int si = p_->ck_assign({ck_exact}, ck_exact, (int) seq.size())[0];
                if (si >= 0) p_->ck_save_end(si);
                p_->pf_run(std::vector<int>(head.begin() + (ck_exact - n0), head.end()), ck_exact, nullptr);
            } else {
                p_->pf_run(head, n0, nullptr);
            }
        } else {
            const std::vector<int> want = p_->ck_positions(cks, ck_exact, n0, hend);
            const std::vector<int> slot = p_->ck_assign(want, ck_exact, (int) seq.size());
            std::vector<Impl::Snap> snaps;
            for (size_t k = 0; k < want.size(); k++)
                if (slot[k] >= 0) snaps.push_back({want[k], slot[k]});
            const auto t0 = std::chrono::steady_clock::now();
            p_->pf_run(head, n0, nullptr, nullptr, 0, &snaps);
            for (const auto & q : snaps) { // (the state after position q.pos - 1)
                Impl::CkSlot & k = p_->ck[q.slot];
                k.ctl = c;
                k.ctl.dn_c = k.ctl.mtp_valid = q.pos;
                k.pos = q.pos;
            }
            if (p_->memlog && !snaps.empty()) {
                fprintf(stderr, "[mk-mem] prefill %d..%d with %zu checkpoints (", n0, hend, snaps.size());
                for (const auto & q : snaps) fprintf(stderr, " %d", q.pos);
                fprintf(stderr, " ) in %.1f ms\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
            }
        }
        i = head.size();
        c.dn_c = hend;
        c.mtp_valid = c.dn_c;
    }
    while (i < feed.size()) {
        const int T = att_cap_T(c.dn_c, (int) std::min<size_t>(MAXT_PF, feed.size() - i));
        const bool last = i + T == feed.size();
        c.n = c.dn_c;
        for (int k = 0; k < T; k++) c.tok[k] = feed[i + k];
        CK(cudaMemcpy(p_->hb.ctl, &c, sizeof(Ctl), cudaMemcpyHostToDevice));
        p_->ndraft = p_->depth_top(c.n);
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
    if (end_save && p_->ck_find(c.dn_c) < 0) {
        const int si = p_->ck_assign({c.dn_c}, c.dn_c, c.dn_c)[0];
        if (si >= 0) p_->ck_save_end(si);
    }
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
std::vector<int> MKEngine::checkpoints() const {
    std::vector<int> v;
    for (const auto & k : p_->ck)
        if (k.pos >= 0) v.push_back(k.pos);
    std::sort(v.begin(), v.end());
    return v;
}
int MKEngine::checkpoint_pos() const {
    const std::vector<int> v = checkpoints();
    return v.empty() ? -1 : v.back();
}
bool MKEngine::rewind_to(int pos) {
    const int si = p_->ck_find(pos);
    if (si < 0 || pos > (int) p_->seq.size()) return false;
    p_->ck_restore(si);
    return true;
}
bool MKEngine::rewind_to_checkpoint() { return rewind_to(checkpoint_pos()); }

void MKEngine::truncate(int len) {
    std::vector<int> & seq = p_->seq;
    if (len < p_->hctl.dn_c || len > (int) seq.size())
        throw std::runtime_error("truncate: can only drop tokens emitted after the committed position");
    p_->ring_rewind(len); // MK_KV_WIN: the dropped positions' ring slots held window rows
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
        const int T = att_cap_T((int) i, (int) std::min<size_t>(MAXT_PF, tokens.size() - i));
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
        const int T = att_cap_T(i, std::min(MAXT_PF, N - i));
#if MK_KV_EMU
        p_->kv_demote_to(i - p_->hb.kvemu_win); // recent-window emulation
#endif
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
#if MK_MAXT > 5
    // T >= 6 runs as one MAXT-row pass (padding rows): the whole n + 1 .. n + MAXT must lie in one partition, else at most 5
    int T = att_cap_T(c.n, 1 + p_->nd_avail);
    if (T > 5 && att_cap_T(c.n, MAXT) < MAXT) T = 5;
#else
    const int T = att_cap_T(c.n, 1 + p_->nd_avail);
#endif
    // Draft depth (an upper bound: the on-device gate may stop drafting earlier). Default (depth 0):
    // after a cycle that rejected a draft, draft one token less (but at least 3), since a deeper draft
    // is then unlikely to be accepted; the full depth again after a fully accepted cycle.
    const Impl::Policy & pol = p_->pol;
    const int top = p_->depth_top(c.n); // n_draft (MK_MAXT > 5: at most 4 from Policy::dctx on)
    int nd = top;
    if (pol.depth == 0) nd = (!p_->all_acc && top > 3) ? top - 1 : top;
    else if (pol.depth == 2) nd = std::min(top, std::max(std::min(pol.dmin, top), 1 + p_->last_acc));
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
