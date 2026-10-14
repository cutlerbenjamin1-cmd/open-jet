// GPU clock keeper for CLI benchmarks.  The qwen38_mk CLI's model load is CPU-bound, so the GPU idles to a low
// performance state and the first ~1.8 s of a timed decode would run at ramping clocks (seen on an RTX 3090).  This
// keeps the GPU busy (spin kernels on every SM, ~50% duty) while the CLI loads; cli_bench.py --gpu-keep kills it when
// the CLI prints its "[mk] load" line, i.e. before any timed work starts, so nothing runs next to the measured part.
//
//   nvcc -O2 -arch=sm_86 -o gpu_keep gpu_keep.cu        (use the -arch of your GPU)
#include <cstdio>
#include <unistd.h>
__global__ void spin(unsigned long long ns) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while (t - t0 < ns);
}
int main() {
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, 0);
    for (;;) {
        spin<<<nsm * 2, 128>>>(4000000ull);  // 4 ms busy
        if (cudaDeviceSynchronize() != cudaSuccess) { fprintf(stderr, "gpu_keep: CUDA error\n"); return 1; }
        usleep(4000);                          // 4 ms idle
    }
}
