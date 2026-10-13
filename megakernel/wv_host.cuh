// wv host side: load-time re-layout of the main-pass matrices (MK_WV builds; see wv.cuh). Included by qwen38_mk.cu inside
// its anonymous namespace, after parallel_rows.
#pragma once

// ---- host: load-time re-layout (exact byte permutation of the device layout) --------------------------------------
// w: N rows x rb bytes in the device layout of `type` (after the loader's repacks); meta: Q8_0's scale stream (N rows
// x 2 nd bytes) or empty. N % 16 == 0.
void wv_relayout(int type, int N, int K, int rb, std::vector<uint8_t> & w, std::vector<uint8_t> & meta) {
    const int nb = K / 256;
    std::vector<uint8_t> o(w.size());
    if (wv_fm(type)) {
        const int BS = type == T_Q4K ? 144 : type == T_Q5K ? 176 : 136;
        parallel_rows(N / 16, [&](int64_t tile) {
            for (int rr = 0; rr < 16; rr++) {
                const int r = (int) tile * 16 + rr, g = rr & 7, half = rr >> 3;
                const uint8_t * row = w.data() + (size_t) r * rb;
                for (int sb = 0; sb < nb; sb++) {
                    uint8_t * u = o.data() + ((size_t) tile * nb + sb) * 16 * BS;
                    if (type == T_IQ4X) { // source: the loader's 272-B block pairs (hdr0 | hdr1 | qs0 | qs1)
                        const uint8_t * pr = row + (sb >> 1) * 272;
                        const int odd = sb & 1;
                        memcpy(u + 16 * g + 8 * half, pr + 8 * odd, 8);
                        for (int tq = 0; tq < 4; tq++)
                            for (int p = 0; p < 4; p++)
                                memcpy(u + 128 + 512 * p + 16 * (4 * g + tq) + 8 * half, pr + 16 + 128 * odd + 32 * p + 8 * tq, 8);
                    } else {
                        const uint8_t * blk = row + (size_t) sb * BS;
                        memcpy(u + 32 * g + 16 * half, blk, 16);
                        const int qs = type == T_Q4K ? 16 : 48, qo = type == T_Q4K ? 256 : 768;
                        for (int tq = 0; tq < 4; tq++) {
                            const int l = 4 * g + tq;
                            if (type == T_Q5K) memcpy(u + 256 + 16 * l + 8 * half, blk + 16 + 8 * tq, 8);
                            for (int p = 0; p < 4; p++) memcpy(u + qo + 512 * p + 16 * l + 8 * half, blk + qs + 32 * p + 8 * tq, 8);
                        }
                    }
                }
            }
        });
    } else { // tile-major whole blocks of BS bytes (per super-block), the meta stream likewise
        auto tm = [&](const uint8_t * src, uint8_t * dst, int rowb, int BS) {
            const int nbk = rowb / BS;
            parallel_rows(N / 16, [&](int64_t tile) {
                for (int rr = 0; rr < 16; rr++)
                    for (int i = 0; i < nbk; i++)
                        memcpy(dst + (((size_t) tile * nbk + i) * 16 + rr) * BS, src + ((size_t) tile * 16 + rr) * rowb + (size_t) i * BS, BS);
            });
        };
        tm(w.data(), o.data(), rb, type == T_Q8_0 ? 256 : 144);
        if (!meta.empty()) {
            std::vector<uint8_t> om(meta.size());
            tm(meta.data(), om.data(), (int) (meta.size() / N), 16);
            meta.swap(om);
        }
    }
    w.swap(o);
}
