// CPU vision encoder over libmtmd (see mk_vision.h).
#include "mk_vision.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "gguf.h"
#include "llama.h"
#include "mtmd-helper.h"
#include "mtmd.h"

namespace {
thread_local std::string g_err;

bool g_verbose = false;
void log_filtered(ggml_log_level level, const char * text, void *) {
    if (g_verbose || level == GGML_LOG_LEVEL_ERROR || level == GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
} // namespace

struct mkv_ctx {
    llama_model * vocab = nullptr; // the text model, vocabulary only (mtmd reads its special tokens and RoPE type)
    mtmd_context * mt = nullptr;
    int n_embd = 0;
    int n_threads = 0;
};

struct mkv_image {
    mtmd_input_chunk * chunk = nullptr; // the image chunk of mtmd_tokenize (owned copy)
    int nx = 0, ny = 0, n = 0;
    std::vector<float> rows;
    double encode_ms = 0;
};

extern "C" {

struct mkv_params mkv_params_default(void) {
    mkv_params p;
    p.n_threads = 0;
    p.min_tokens = 0;
    p.max_tokens = 0;
    p.flash_attn = -1;
    p.verbose = 0;
    return p;
}

const char * mkv_last_error(void) { return g_err.c_str(); }

mkv_ctx * mkv_new(const char * mmproj, const char * text_model, struct mkv_params params) {
    g_err.clear();
    auto * c = new mkv_ctx;
    try {
        g_verbose = params.verbose != 0;
        mtmd_helper_log_set(log_filtered, nullptr);
        llama_log_set(log_filtered, nullptr);
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        c->vocab = llama_model_load_from_file(text_model, mp);
        llama_log_set(nullptr, nullptr); // (back to libllama's default for the engine's own logging)
        if (!c->vocab) throw std::runtime_error(std::string("cannot load the text model's vocabulary: ") + text_model);
        mtmd_context_params cp = mtmd_context_params_default();
        cp.use_gpu = false;
        cp.print_timings = params.verbose != 0;
        c->n_threads = params.n_threads > 0 ? params.n_threads : (int) std::max(1u, std::thread::hardware_concurrency());
        cp.n_threads = c->n_threads;
        cp.flash_attn_type = params.flash_attn < 0 ? LLAMA_FLASH_ATTN_TYPE_AUTO
                             : params.flash_attn  ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                                                  : LLAMA_FLASH_ATTN_TYPE_DISABLED;
        cp.warmup = false; // (the first encode pays the graph allocation, ~10 ms)
        cp.image_min_tokens = params.min_tokens > 0 ? params.min_tokens : -1;
        cp.image_max_tokens = params.max_tokens > 0 ? params.max_tokens : -1;
        c->mt = mtmd_init_from_file(mmproj, c->vocab, cp);
        if (!c->mt) throw std::runtime_error(std::string("cannot load the mmproj: ") + mmproj);
        if (!mtmd_support_vision(c->mt)) throw std::runtime_error("the mmproj has no vision encoder");
        if (!mtmd_decode_use_mrope(c->mt)) throw std::runtime_error("the mmproj does not use M-RoPE positions (not a Qwen-VL projector)");
        // the row width: the mmproj's projection size (mtmd sizes its output by it; a vocabulary-only text model carries
        // no hyperparameters to check it against)
        gguf_init_params gp = {true, nullptr};
        gguf_context * g = gguf_init_from_file(mmproj, gp);
        if (!g) throw std::runtime_error(std::string("cannot read the mmproj's metadata: ") + mmproj);
        const int64_t k = gguf_find_key(g, "clip.vision.projection_dim");
        c->n_embd = k >= 0 ? (int) gguf_get_val_u32(g, k) : 0;
        gguf_free(g);
        if (c->n_embd <= 0) throw std::runtime_error("the mmproj has no clip.vision.projection_dim");
        return c;
    } catch (const std::exception & e) {
        g_err = e.what();
        mkv_free(c);
        return nullptr;
    }
}

void mkv_free(mkv_ctx * c) {
    if (!c) return;
    if (c->mt) mtmd_free(c->mt);
    if (c->vocab) llama_model_free(c->vocab);
    delete c;
}

int mkv_n_embd(const mkv_ctx * c) { return c->n_embd; }
int mkv_n_threads(const mkv_ctx * c) { return c->n_threads; }

mkv_image * mkv_image_load(mkv_ctx * c, const unsigned char * data, size_t len) {
    g_err.clear();
    mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_buf(c->mt, data, len, false);
    if (!bw.bitmap) {
        g_err = "cannot decode the image";
        return nullptr;
    }
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char * marker = mtmd_default_marker();
    mtmd_input_text txt = {marker, strlen(marker), false, true};
    const mtmd_bitmap * bms[1] = {bw.bitmap};
    const int32_t r = mtmd_tokenize(c->mt, chunks, &txt, bms, 1);
    mtmd_bitmap_free(bw.bitmap);
    mkv_image * im = nullptr;
    if (r != 0) {
        g_err = "image preprocessing failed (mtmd_tokenize " + std::to_string(r) + ")";
    } else {
        for (size_t i = 0; i < mtmd_input_chunks_size(chunks); i++) {
            const mtmd_input_chunk * ch = mtmd_input_chunks_get(chunks, i);
            if (mtmd_input_chunk_get_type(ch) != MTMD_INPUT_CHUNK_TYPE_IMAGE) continue;
            const mtmd_image_tokens * it = mtmd_input_chunk_get_tokens_image(ch);
            const size_t n = mtmd_image_tokens_get_n_tokens(it);
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, n - 1);
            im = new mkv_image;
            im->chunk = mtmd_input_chunk_copy(ch);
            im->nx = (int) last.x + 1; // (pos_0 = 0: the last raster row's x / y are nx - 1 / ny - 1)
            im->ny = (int) last.y + 1;
            im->n = (int) n;
            if ((size_t) im->nx * im->ny != n) {
                g_err = "unexpected image token layout";
                mkv_image_free(im);
                im = nullptr;
            }
            break;
        }
        if (!im && g_err.empty()) g_err = "no image chunk";
    }
    mtmd_input_chunks_free(chunks);
    return im;
}

int mkv_image_grid(const mkv_image * im, int * nx, int * ny) {
    if (nx) *nx = im->nx;
    if (ny) *ny = im->ny;
    return im->n;
}

int mkv_image_encode(mkv_ctx * c, mkv_image * im) {
    g_err.clear();
    if (!im->rows.empty()) return 0;
    const double t0 = now_ms();
    if (mtmd_encode_chunk(c->mt, im->chunk) != 0) {
        g_err = "image encode failed";
        return -1;
    }
    const float * e = mtmd_get_output_embd(c->mt); // n rows of the projection size
    im->rows.assign(e, e + (size_t) im->n * c->n_embd);
    im->encode_ms = now_ms() - t0;
    return 0;
}

const float * mkv_image_rows(const mkv_image * im) { return im->rows.empty() ? nullptr : im->rows.data(); }
double mkv_image_encode_ms(const mkv_image * im) { return im->encode_ms; }

void mkv_image_free(mkv_image * im) {
    if (!im) return;
    if (im->chunk) mtmd_input_chunk_free(im->chunk);
    delete im;
}

} // extern "C"
