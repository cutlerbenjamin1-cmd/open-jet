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
    // Drops emitted tokens beyond `len` sequence positions (e.g. those generated after an
    // end-of-turn token within the last cycle) before the next prefill.
    void truncate(int len);
    // The sequence so far (prompts and emitted tokens; position = index) and the number of its
    // positions folded into the recurrent state (truncate() can only go back to there).
    const std::vector<int> & sequence() const;
    int committed() const;
    // Each prefill saves a checkpoint at the end of its prompt; rewinding to it drops everything
    // after (also tokens before the committed position), so a prompt that diverges anywhere after
    // the checkpoint can be re-fed from there.  Returns false when there is no checkpoint.
    int checkpoint_pos() const;
    bool rewind_to_checkpoint();
    // Runs one verify+draft cycle; appends 1..n_draft+1 accepted tokens to `out`.
    int step(std::vector<int> & out);
    // Debug helper: copy of the last main-model logits row selection (argmax ids) is not kept;
    // instead expose the final normalised hidden row of the last processed position.
    std::vector<float> last_hidden();

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
    static int n_vocab();
    void bench_phases();
    int n_draft() const { return n_draft_; }
    MKStats stats;

  private:
    struct Impl;
    Impl * p_;
    int n_draft_;
};
