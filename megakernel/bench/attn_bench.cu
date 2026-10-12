// Correctness + throughput of the prefill attention kernels (prefill_attn.cuh) on a random q8 (or fp16) KV
// cache: the original pk_attn against pk_attn2 (exact, + tail split, + split and fp16 P.V) and pk_attn8 (int8
// Q K^T, q8 cache only; Q scales per 32-dim block / per row).
//   nvcc -O3 -arch=sm_86 -std=c++17 -Xptxas -dlcm=cg attn_bench.cu -o attn_bench
//   ./attn_bench <ctx> <reps> [f16] <T:n0> ...      e.g. ./attn_bench 101100 3 1024:30000 1024:100000
// Per config (T queries at positions n0 .., one layer): ms per launch (min of reps), achieved TFLOPS (causal
// FLOPs), and per variant the number of outputs that differ from pk_attn (0 for exact pk_attn2), max |diff| and the
// relative rms difference.  AB_VARS=<digits> selects variants (0 old, 1 new, 2 split, 3 split + PV16, 4 s8, 5 s8row;
// default all); TRUEREF=1 (q8) also reports each variant's relative rms error against pk_attn on the unquantized
// fp16 cache (the q8 cache's own error is the "old" line).
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#define CK(x)                                                                                    \
    do {                                                                                         \
        cudaError_t e_ = (x);                                                                    \
        if (e_ != cudaSuccess) {                                                                 \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(e_)); \
            exit(1);                                                                             \
        }                                                                                        \
    } while (0)

namespace {
constexpr int NH = 24, NKV = 4, HD = 256, VD = NH * HD;
struct Bufs {
    void *kc, *vc;
    __half *ksc, *vsc;
    int kvq8;
    int ctx;
};
struct PBufs {
    __half * qh;
    float * y2;
};
__constant__ Bufs cB;
__constant__ PBufs cP;

__device__ __forceinline__ void cp_async16(void * smem_dst, const void * gsrc) {
    const unsigned d = (unsigned) __cvta_generic_to_shared(smem_dst);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(d), "l"(gsrc) : "memory");
}
__device__ __forceinline__ void q8_to_f16(uint2 q, float d, __half2 (&h)[4]) {
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const unsigned x = k < 2 ? q.x : q.y;
        const int sh = 16 * (k & 1);
        h[k] = __floats2half2_rn(d * (float) (int8_t) (x >> sh), d * (float) (int8_t) (x >> (sh + 8)));
    }
}
__device__ __forceinline__ void mma16816(float (&c)[4], unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0,
                                         unsigned b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
namespace pf {
__device__ __forceinline__ void cpa_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N> __device__ __forceinline__ void cpa_wait() { asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory"); }
__device__ __forceinline__ void ldsm_x4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
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
} // namespace pf

#include "../prefill_attn.cuh"

int g_slots = 0;
void *g_kcf = nullptr, *g_vcf = nullptr;
float * g_part = nullptr;
size_t g_part_floats = 0;

// variant: 0 old, 1 new exact (no split), 2 new + split, 3 new + split + PV16
void launch(int variant, bool q8, int T, int n0) {
    const int slot = 0;
    if (variant == 0) {
        if (q8) pk_attn<true><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n0, slot);
        else pk_attn<false><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n0, slot);
        return;
    }
    const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV;
    int nfull = N, S = 1;
    if (variant >= 2) a2_plan(T, n0, g_slots, g_part_floats, nfull, S);
    const int grid = nfull + (N - nfull) * S;
    if (variant >= 4) {
        if (variant == 4) pk_attn8<false><<<grid, A2_NT, A8_SMEM>>>(T, n0, slot, nfull, S, g_part);
        else pk_attn8<true><<<grid, A2_NT, A8_SMEM>>>(T, n0, slot, nfull, S, g_part);
        if (N > nfull) pk_attn2_comb<<<N - nfull, 256>>>(T, nfull, S, g_part);
        return;
    }
    const bool pv16 = variant == 3;
    if (q8 && !pv16) pk_attn2<true, false><<<grid, A2_NT, A2_SMEM>>>(T, n0, slot, nfull, S, g_part);
    else if (q8) pk_attn2<true, true><<<grid, A2_NT, A2_SMEM>>>(T, n0, slot, nfull, S, g_part);
    else if (!pv16) pk_attn2<false, false><<<grid, A2_NT, A2_SMEM>>>(T, n0, slot, nfull, S, g_part);
    else pk_attn2<false, true><<<grid, A2_NT, A2_SMEM>>>(T, n0, slot, nfull, S, g_part);
    if (N > nfull) pk_attn2_comb<<<N - nfull, 256>>>(T, nfull, S, g_part);
}
} // namespace

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: attn_bench <ctx> <reps> [f16] <T:n0> ...\n");
        return 1;
    }
    const int ctx = atoi(argv[1]), reps = atoi(argv[2]);
    int ai = 3;
    bool q8 = true;
    if (!strcmp(argv[ai], "f16")) { q8 = false; ai++; }
    int maxT = 0;
    std::vector<std::pair<int, int>> cfg;
    for (; ai < argc; ai++) {
        int T, n0;
        if (sscanf(argv[ai], "%d:%d", &T, &n0) != 2 || n0 + T > ctx) { fprintf(stderr, "bad config %s\n", argv[ai]); return 1; }
        cfg.push_back({T, n0});
        maxT = std::max(maxT, T);
    }
    CK(cudaFuncSetAttribute(pk_attn<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
    CK(cudaFuncSetAttribute(pk_attn<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, AT_SMEM));
    CK(cudaFuncSetAttribute(pk_attn2<true, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(pk_attn2<true, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(pk_attn2<false, false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(pk_attn2<false, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A2_SMEM));
    CK(cudaFuncSetAttribute(pk_attn8<false>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    CK(cudaFuncSetAttribute(pk_attn8<true>, cudaFuncAttributeMaxDynamicSharedMemorySize, A8_SMEM));
    int nsm = 0, occ = 0;
    CK(cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0));
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, pk_attn2<true, false>, A2_NT, A2_SMEM));
    g_slots = nsm * occ;
    printf("SMs %d, pk_attn2 CTAs/SM %d, smem old %d new %d B, kv %s, ctx %d\n", nsm, occ, AT_SMEM, A2_SMEM, q8 ? "q8" : "f16", ctx);

    // random cache: K, V values ~ N(0, 1) (q8 with per-32 scales), Q ~ N(0, 1) * 3 / 16 (pre-scaled like pk_attn_prep)
    std::mt19937 rng(1234);
    std::normal_distribution<float> nd(0.f, 1.f);
    const size_t rows = (size_t) NKV * ctx;
    void *kc, *vc;
    __half *ksc = nullptr, *vsc = nullptr;
    {
        std::vector<int8_t> q(rows * HD);
        std::vector<__half> sc(rows * (HD / 32)), hv(rows * HD);
        for (int which = 0; which < 2; which++) {
            for (size_t r = 0; r < rows; r++)
                for (int b = 0; b < HD / 32; b++) {
                    float v[32], am = 0.f;
                    for (int k = 0; k < 32; k++) { v[k] = nd(rng); am = std::max(am, std::fabs(v[k])); }
                    for (int k = 0; k < 32; k++) {
                        q[r * HD + 32 * b + k] = (int8_t) std::lrint(v[k] * 127.f / am);
                        hv[r * HD + 32 * b + k] = __float2half(v[k]);
                    }
                    sc[r * (HD / 32) + b] = __float2half(am / 127.f);
                }
            void * d;
            if (q8 && getenv("TRUEREF")) { // unquantized fp16 copy for the true reference
                void * f;
                CK(cudaMalloc(&f, rows * HD * 2));
                CK(cudaMemcpy(f, hv.data(), rows * HD * 2, cudaMemcpyHostToDevice));
                (which ? g_vcf : g_kcf) = f;
            }
            if (q8) {
                CK(cudaMalloc(&d, rows * HD));
                CK(cudaMemcpy(d, q.data(), rows * HD, cudaMemcpyHostToDevice));
                __half * s;
                CK(cudaMalloc(&s, rows * (HD / 32) * 2));
                CK(cudaMemcpy(s, sc.data(), rows * (HD / 32) * 2, cudaMemcpyHostToDevice));
                (which ? vsc : ksc) = s;
            } else {
                CK(cudaMalloc(&d, rows * HD * 2));
                CK(cudaMemcpy(d, hv.data(), rows * HD * 2, cudaMemcpyHostToDevice));
            }
            (which ? vc : kc) = d;
        }
    }
    __half * qh;
    float *y2, *y2r;
    {
        std::vector<__half> hq((size_t) maxT * NH * HD);
        for (auto & x : hq) x = __float2half(nd(rng) * 3.f / 16.f);
        CK(cudaMalloc(&qh, hq.size() * 2));
        CK(cudaMemcpy(qh, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaMalloc(&y2, (size_t) maxT * VD * 4));
        CK(cudaMalloc(&y2r, (size_t) maxT * VD * 4));
    }
    g_part_floats = (size_t) maxT * 17408; // y1 of the engine: cap x FF floats
    CK(cudaMalloc(&g_part, g_part_floats * 4));
    Bufs b{kc, vc, ksc, vsc, q8 ? 1 : 0, ctx};
    PBufs p{qh, y2};
    CK(cudaMemcpyToSymbol(cB, &b, sizeof b));
    CK(cudaMemcpyToSymbol(cP, &p, sizeof p));

    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const char * names[6] = {"old", "new", "new+split", "new+split+pv16", "s8+split", "s8row+split"};
    for (auto [T, n0] : cfg) {
        const int nqt = (T + A2_Q - 1) / A2_Q, N = nqt * NKV;
        int nfull = N, S = 1;
        a2_plan(T, n0, g_slots, g_part_floats, nfull, S);
        // causal FLOPs: 4 * HD per (query row, visible key)
        double flop = 0;
        for (int t = 0; t < T; t++) flop += 4.0 * HD * NH * (double) (n0 + t + 1);
        printf("T %d n0 %d: items %d, plan nfull %d split S %d\n", T, n0, N, nfull, S);
        std::vector<float> ref((size_t) T * VD), out((size_t) T * VD), tref;
        if (g_kcf) { // true reference: original kernel on the unquantized fp16 cache
            Bufs bf{g_kcf, g_vcf, nullptr, nullptr, 0, ctx};
            CK(cudaMemcpyToSymbol(cB, &bf, sizeof bf));
            pk_attn<false><<<dim3((T + AT_Q - 1) / AT_Q, NKV), 192, AT_SMEM>>>(T, n0, 0);
            CK(cudaDeviceSynchronize());
            tref.resize((size_t) T * VD);
            CK(cudaMemcpy(tref.data(), y2, tref.size() * 4, cudaMemcpyDeviceToHost));
            CK(cudaMemcpyToSymbol(cB, &b, sizeof b));
        }
        const char * vsel = getenv("AB_VARS") ? getenv("AB_VARS") : "0123";
        for (int v = 0; v < 6; v++) {
            if (!strchr(vsel, '0' + v)) continue;
            if (v >= 4 && !q8) continue; // pk_attn8 reads the q8 cache
            CK(cudaMemset(y2, 0xff, (size_t) T * VD * 4)); // NaN
            launch(v, q8, T, n0);
            CK(cudaDeviceSynchronize());
            CK(cudaGetLastError());
            CK(cudaMemcpy(v == 0 ? ref.data() : out.data(), y2, (size_t) T * VD * 4, cudaMemcpyDeviceToHost));
            float best = 1e30f;
            for (int r = 0; r < reps; r++) {
                CK(cudaEventRecord(e0));
                launch(v, q8, T, n0);
                CK(cudaEventRecord(e1));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                best = std::min(best, ms);
            }
            printf("  %-15s %9.3f ms  %6.2f TFLOPS", names[v], best, flop / best / 1e9);
            if (!tref.empty()) {
                const std::vector<float> & o = v == 0 ? ref : out;
                double sd = 0, sr = 0;
                for (size_t i = 0; i < o.size(); i++) { sd += (double) (o[i] - tref[i]) * (o[i] - tref[i]); sr += (double) tref[i] * tref[i]; }
                printf("  vs-fp16-cache rel-rms %.4g", std::sqrt(sd / sr));
            }
            if (v > 0) {
                size_t nbits = 0, nnan = 0;
                double md = 0, mref = 0, sd = 0, sr = 0;
                for (size_t i = 0; i < out.size(); i++) {
                    if (std::isnan(out[i])) { nnan++; continue; }
                    if (memcmp(&out[i], &ref[i], 4)) nbits++;
                    md = std::max(md, (double) std::fabs(out[i] - ref[i]));
                    sd += (double) (out[i] - ref[i]) * (out[i] - ref[i]);
                    sr += (double) ref[i] * ref[i];
                    mref = std::max(mref, (double) std::fabs(ref[i]));
                }
                printf("  differing %zu / %zu, NaN %zu, max|d| %.3g (max|ref| %.3g) rel-rms %.3g", nbits, out.size(), nnan, md, mref, sr > 0 ? std::sqrt(sd / sr) : 0.0);
            }
            printf("\n");
        }
    }
    printf("DONE\n");
    return 0;
}
