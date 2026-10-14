// wv host side: load-time re-layout of the main-pass matrices (MK_WV builds; see wv.cuh). Included by qwen38_mk.cu inside
// its anonymous namespace, after parallel_rows.
#pragma once

// ---- host: load-time re-layout (exact byte permutation of the device layout) --------------------------------------
// w: N rows x rb bytes in the device layout of `type` (after the loader's repacks); meta: Q8_0's scale stream (N rows
// x 2 nd bytes) or empty. N % 16 == 0.
void wv_relayout(int type, int N, int K, int rb, std::vector<uint8_t> & w, std::vector<uint8_t> & meta) {
    const int nb = K / 256;
#if MK_MAXT > 5 && MK_WV_Q6F
    if (type == T_Q6K) { // FM Q6_K (wv.cuh "FM Q6_K"): units of Q6F_UNIT bytes, the meta stream inline (meta is emptied)
        const int mrow = nb * Q6_META;
        std::vector<uint8_t> f((size_t) N / 16 * nb * Q6F_UNIT);
        parallel_rows(N / 16, [&](int64_t tile) {
            for (int rr = 0; rr < 16; rr++) {
                const int r = (int) tile * 16 + rr, g = rr & 7, half = rr >> 3;
                for (int sb = 0; sb < nb; sb++) {
                    uint8_t * u = f.data() + ((size_t) tile * nb + sb) * Q6F_UNIT;
                    const uint8_t * blk = w.data() + (size_t) r * rb + (size_t) sb * Q6_BLK;
                    const uint8_t * mp = meta.data() + (size_t) r * mrow + (sb >> 1) * 2 * Q6_META;
                    memcpy(u + 32 * g + 16 * half, mp + 16 * (sb & 1), 16);
                    memcpy(u + Q6F_D + 4 * g + 2 * half, mp + 32 + 2 * (sb & 1), 2);
                    for (int tq = 0; tq < 4; tq++) {
                        const int l = 4 * g + tq;
                        for (int p = 0; p < 4; p++) {
                            memcpy(u + Q6F_LO + 512 * p + 16 * l + 8 * half, blk + 32 * p + 4 * tq, 4);
                            memcpy(u + Q6F_LO + 512 * p + 16 * l + 8 * half + 4, blk + 32 * p + 16 + 4 * tq, 4);
                            uint8_t * hd = u + Q6F_HI + 256 * p + 8 * l + 4 * half;
                            for (int mm = 0; mm < 4; mm++) { // high 2 bits of element 4 tq + mm of (sub-block 2p + part, chunk c)
                                auto hb = [&](int part, int c) { return (blk[128 + 8 * (2 * p + c) + 4 * part + mm] >> (2 * tq)) & 3; };
                                hd[mm] = (uint8_t) ((hb(0, 0) << 4) | (hb(1, 0) << 6) | hb(0, 1) | (hb(1, 1) << 2));
                            }
                        }
                    }
                }
            }
        });
        w.swap(f);
        meta.clear();
        return;
    }
#endif
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
    } else { // tile-major whole blocks of BS bytes (per super-block), the meta stream likewise (MB-byte entries)
        auto tm = [&](const uint8_t * src, uint8_t * dst, int rowb, int BS) {
            const int nbk = rowb / BS;
            parallel_rows(N / 16, [&](int64_t tile) {
                for (int rr = 0; rr < 16; rr++)
                    for (int i = 0; i < nbk; i++)
                        memcpy(dst + (((size_t) tile * nbk + i) * 16 + rr) * BS, src + ((size_t) tile * 16 + rr) * rowb + (size_t) i * BS, BS);
            });
        };
        // Q4_K / IQ4_NL 144, Q5_K 176, Q8_0 256 (+ 16-B scale entries); MK_MAXT > 5 ffn_down: Q6_K 192 / Q3_K 96 (+ 36-B meta
        // pairs), IQ3_S 96 (+ 16-B meta per block)
        const int BS = type == T_Q8_0 ? 256 : type == T_Q5K ? 176 : type == T_Q6K ? Q6_BLK : type == T_Q3K || type == T_IQ3S ? Q3_BLK : 144;
        const int MB = type == T_Q6K || type == T_Q3K ? 2 * Q6_META : 16;
        tm(w.data(), o.data(), rb, BS);
        if (!meta.empty()) {
            std::vector<uint8_t> om(meta.size());
            tm(meta.data(), om.data(), (int) (meta.size() / N), MB);
            meta.swap(om);
        }
    }
    w.swap(o);
}
