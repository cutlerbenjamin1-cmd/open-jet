// Qwen3.8-27B (llama.cpp arch "qwen35", shared with Qwen3.5/3.6) single-launch-per-cycle CUDA megakernel decoder.
#pragma once

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

// A failed (non-sticky) CUDA call inside the engine. The engine's sequence state is then undefined: reset() it
// before the next use (the C API does). Sticky errors (the CUDA context is lost) exit the process with code 4.
struct MKCudaError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// An image in a prompt (vision input, e.g. an mmproj encoder's output): positions pos .. pos + nx * ny - 1 take these
// input-embedding rows instead of their tokens' embeddings (the tokens there are placeholders, e.g. <|image_pad|>),
// with llama.cpp's M-RoPE positions for Qwen-VL: the grid's rows in raster order, nx merged patches per row.
struct MKImage {
    int pos;            // absolute sequence position of the first row
    int nx, ny;         // grid: columns, rows
    const float * embd; // [nx * ny][5120] fp32 (read during the prefill call)
};

struct MKStats {
    long cycles = 0;
    long drafted = 0;
    long accepted = 0;
    int last_T = 0, last_nd = 0, last_acc = 0; // the last step: tokens verified, drafts produced, accepted
};

class MKEngine {
  public:
    // n_draft: MTP draft tokens per cycle (0..4). ctx: KV-cache capacity in tokens.
    // draft_vocab: optional file of int32 token ids; the MTP draft head is restricted to them.
    // kv_mode: KV cache format, 0 fp16, 1 q8_0 (half the memory, ~1-2% slower), -1 auto (q8_0 only
    // when the fp16 cache for ctx positions does not fit in the free GPU memory).
    MKEngine(const std::string & gguf_path, int ctx, int n_draft, const std::string & draft_vocab = "", int kv_mode = -1);
    ~MKEngine();

    // Clears all sequence state (KV, recurrent state, positions).
    void reset();
    // Processes the prompt, or appends `tokens` to the sequence so far (prompt + emitted tokens,
    // e.g. the next user turn); returns the first generated token (greedy).
    // ck_at (absolute position inside the fed range): save the rewind checkpoint there instead of at the end of the
    // prompt, e.g. at the start of the generation prompt, which a follow-up turn's template may render differently
    // (thinking mode re-renders the previous turn as <think>\n\n</think>, one token before the end of the prompt).
    int prefill(const std::vector<int> & tokens, int ck_at = -1);
    // prefill() saving checkpoints at the absolute positions cks (inside this prompt) into the checkpoint pool:
    // MK_CK_SLOTS (default 8) checkpoints of 0.17 GB of pinned host memory each, evicted by coverage when full
    // (older ones thin out, recent ones stay dense). ck_exact (or -1) is kept at its exact position; the others move
    // down by up to 31 tokens onto the batched prefill's scan grid. Saving changes nothing the prefill computes.
    // Unless ck_exact is inside the batched part, a checkpoint at the end of the prompt is saved too (as prefill()).
    int prefill(const std::vector<int> & tokens, const std::vector<int> & cks, int ck_exact);
    // prefill(tokens, cks, ck_exact) with images inside the prompt (each ends before its last token; batched prefill
    // only). The sequence keeps their positions for its RoPE rows until a reset, rewind or truncate drops them; after a
    // rewind into an image, the next prefill must pass that image again (same pos and grid).
    int prefill(const std::vector<int> & tokens, const std::vector<int> & cks, int ck_exact, const std::vector<MKImage> & images);
    // Sampling: temp 0 = greedy (default); otherwise llama.cpp's order top_k (1..32), top_p, min_p,
    // temperature.  Speculative verification keeps the sampled distribution exact.
    void set_sampling(float temp, float top_p, int top_k, float min_p, unsigned long long seed);
    // Presence / frequency penalty (OpenAI parameters, vLLM's semantics): each generated token's logit is lowered by
    // frequency * c + presence * (c > 0), c = its occurrences among the tokens generated so far in this response
    // (since the last prefill; prompt tokens do not count). Greedy and sampled; speculative verification stays exact
    // (a verify row counts the drafts before it). 0, 0 (default): off.
    void set_penalty(float presence, float frequency);
    // Draft-depth policy, "key=value,..." (see qwen38_mk.cu); "" restores the default. The env var
    // MK_POLICY sets it at construction.
    void set_policy(const std::string & spec);
    // KV-cache quantisation emulation (builds with -DMK_KV_EMU=1; accuracy experiments): the MK_KV_EMU_SPEC syntax
    // (see qwen38_mk.cu), applied to the rows stored from now on. Throws on other builds.
    void set_kv_emu(const std::string & spec);
    // Drops emitted tokens beyond `len` sequence positions (e.g. those generated after an
    // end-of-turn token within the last cycle) before the next prefill.
    void truncate(int len);
    // The sequence so far (prompts and emitted tokens; position = index) and the number of its
    // positions folded into the recurrent state (truncate() can only go back to there).
    const std::vector<int> & sequence() const;
    int committed() const;
    // Each prefill saves a checkpoint at the end of its prompt (or at ck_at, and at cks); rewinding to one
    // drops everything after it (also tokens before the committed position) and the checkpoints after it, so a
    // prompt that diverges anywhere after the checkpoint can be re-fed from there.
    // checkpoints(): positions of the checkpoints (ascending); rewind_to(pos): restores the one at pos (false: none);
    // checkpoint_pos() / rewind_to_checkpoint(): the last one (-1 / false: none).
    std::vector<int> checkpoints() const;
    bool rewind_to(int pos);
    int checkpoint_pos() const;
    bool rewind_to_checkpoint();
    // Runs one verify+draft cycle; appends 1..n_draft+1 accepted tokens to `out`.
    int step(std::vector<int> & out);
    // Debug helper: copy of the last main-model logits row selection (argmax ids) is not kept;
    // instead expose the final normalised hidden row of the last processed position.
    std::vector<float> last_hidden();
    // Test helper (exact-reuse probes): 64-bit digests of the sequence state, layout in qwen38_mk.cu (state_digest).
    std::vector<unsigned long long> state_digest();
    // MK_EXACT=1 at construction: prefill results independent of the chunking (see qwen38_mk.cu `exact`).
    bool exact() const;

    // Draft depth for the following prefills and steps, 0 .. the n_draft given at construction (CLI batch mode: an engine
    // built for the deepest job runs each job at its own depth; the same launches as an engine built for that depth).
    void set_n_draft(int n);
    // Post-prefill state files (test runs: decode A-B-As at long context without re-prefilling). prefill(tokens) on an
    // empty sequence that first looks in dir for a state file of the same batched prefill (every prompt token but the
    // last, which then goes through the megakernel as in prefill()) and restores it instead of running the prefill; on a
    // miss it runs the prefill and saves the file. The key covers the binary (its file hash), the model file, the
    // context, the KV format and every MK_* setting that is not known to be decode-only, so a restore leaves the state
    // bit for bit as the prefill would. *how: 1 restored, 2 saved (0: not applicable, plain prefill).
    int prefill_state(const std::vector<int> & tokens, const std::string & dir, int * how = nullptr);

    int position() const;
    // Teacher-forced greedy predictions: pred[i] = argmax of the next-token distribution after
    // tokens[0..i] (resets the sequence).
    std::vector<int> score(const std::vector<int> & tokens);
    // Teacher-forced next-token logits (resets the sequence): sink(i0, n, logits) receives the logits
    // ([n][n_vocab()], valid during the call) after tokens[0..i] for i = i0 .. i0 + n - 1, in position
    // order, for all positions >= first.  Positions below mk_from are processed by the batched
    // prefill kernels, the others by the decode megakernel.
    using LogitSink = std::function<void(int i0, int n, const float * logits)>;
    void score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink);
    // score_logits with images (inside the batched part: below mk_from)
    void score_logits(const std::vector<int> & tokens, int mk_from, int first, const LogitSink & sink,
                      const std::vector<MKImage> & images);
    static int n_vocab();
    static int n_embd(); // width of an MKImage row
    void bench_phases();
    int n_draft() const { return n_draft_; }
    MKStats stats;

  private:
    struct Impl;
    Impl * p_;
    int n_draft_;
};
