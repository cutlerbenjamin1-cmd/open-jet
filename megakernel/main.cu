// CLI for the Qwen3.8-27B megakernel: tokenises with libllama (vocab only), runs greedy
// speculative decoding and reports decode throughput.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>

#include "llama.h"
#include "logit_stats.h"
#include "qwen38_mk.h"

static void usage() {
    fprintf(stderr,
            "usage: qwen38_mk -m model.gguf [-p prompt | -f prompt_file] [-n n_predict] [-d n_draft]\n"
            "                 [-c ctx] [--draft-vocab ids.bin] [--raw] [--ids] [--quiet]\n"
            "                 [--turn2 text | --turn2-file f]   (a second user turn, appended to the first)\n"
            "                 [--prompt-ids f]   (prompt given as whitespace-separated token ids)\n"
            "                 [--think [xhigh|medium|low]]   (reasoning mode, the chat template's default effort is xhigh)\n"
            "                 [--temp t] [--top-p p] [--top-k k] [--min-p p] [--seed s]   (default: greedy)\n"
            "                 [--presence-penalty p] [--frequency-penalty f]   (on the generated tokens; default 0)\n"
            "                 [--kv f16|q8|auto]   (KV cache format; auto: q8_0 only when fp16 does not fit)\n"
            "                 [--kld ref [--ref-out f] [--score-path prefill|decode|mixed]]   (KL divergence, perplexity and\n"
            "                     top-token agreement against a reference file of tools/llama_ref or of --ref-out)\n"
            "                 [--state-dir dir]   (post-prefill state files: the first prompt's prefill is restored from\n"
            "                     dir when a file of the same prompt, binary, model and settings is there, else saved there)\n"
            "                 [--digest]   (print a hash of the sequence state after the first prefill: test runs)\n"
            "                 [--batch jobs.txt]   (batch mode: one job per line, its arguments as on this command line\n"
            "                     (the generation options above) plus -o <prefix>: stdout to <prefix>.out, stderr to\n"
            "                     <prefix>.err; the model, -c, --kv and --draft-vocab are this command line's; one model\n"
            "                     load for all jobs, the engine reset between them: each job's output = a separate run's)\n"
            "  The CLI keeps 1 DeltaNet checkpoint slot unless MK_CK_SLOTS is set (the server's pool is for reuse).\n");
    exit(1);
}

// The options of one generation job (the command line's, or one line of --batch on top of them)
struct Job {
    std::string prompt = "Write a Python function that checks whether a number is prime.";
    int n_predict = 256, n_draft = 2;
    bool raw = false, print_ids = false, quiet = false, digest = false;
    std::string turn2, prompt_ids_file, think, out, state_dir;
    float temp = 0.f, top_p = 1.f, min_p = 0.f, pres = 0.f, freq = 0.f;
    int top_k = 32;
    unsigned long long seed = 1;
};

static std::string read_file(const std::string & p) {
    std::ifstream f(p);
    if (!f) {
        fprintf(stderr, "cannot read %s\n", p.c_str());
        exit(1);
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}

// one job option at argv[i] (advancing i past its value); false: not a job option
static bool job_arg(Job & j, const std::vector<std::string> & av, size_t & i) {
    const std::string & a = av[i];
    auto next = [&]() -> std::string {
        if (i + 1 >= av.size()) usage();
        return av[++i];
    };
    if (a == "-p") j.prompt = next();
    else if (a == "-f") j.prompt = read_file(next());
    else if (a == "-n") j.n_predict = atoi(next().c_str());
    else if (a == "-d") j.n_draft = atoi(next().c_str());
    else if (a == "--raw") j.raw = true;
    else if (a == "--ids") j.print_ids = true;
    else if (a == "--quiet") j.quiet = true;
    else if (a == "--digest") j.digest = true;
    else if (a == "--turn2") j.turn2 = next();
    else if (a == "--turn2-file") j.turn2 = read_file(next());
    else if (a == "--prompt-ids") j.prompt_ids_file = next();
    else if (a == "--think") j.think = (i + 1 < av.size() && av[i + 1][0] != '-') ? next() : "xhigh";
    else if (a == "--temp") j.temp = atof(next().c_str());
    else if (a == "--top-p") j.top_p = atof(next().c_str());
    else if (a == "--top-k") j.top_k = atoi(next().c_str());
    else if (a == "--min-p") j.min_p = atof(next().c_str());
    else if (a == "--seed") j.seed = strtoull(next().c_str(), nullptr, 10);
    else if (a == "--presence-penalty") j.pres = atof(next().c_str());
    else if (a == "--frequency-penalty") j.freq = atof(next().c_str());
    else if (a == "--state-dir") j.state_dir = next();
    else if (a == "-o") j.out = next();
    else return false;
    return true;
}

// shell-like split of a batch line: whitespace separates, '...' is literal, "..." keeps backslash escapes of " and \.
static std::vector<std::string> split_line(const std::string & s, bool & ok) {
    std::vector<std::string> v;
    std::string cur;
    bool have = false;
    ok = true;
    for (size_t i = 0; i < s.size();) {
        const char c = s[i];
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (have) v.push_back(cur), cur.clear(), have = false;
            i++;
        } else if (c == '\'') {
            const size_t e = s.find('\'', i + 1);
            if (e == std::string::npos) return ok = false, v;
            cur += s.substr(i + 1, e - i - 1);
            have = true;
            i = e + 1;
        } else if (c == '"') {
            i++;
            for (; i < s.size() && s[i] != '"'; i++) {
                if (s[i] == '\\' && i + 1 < s.size() && (s[i + 1] == '"' || s[i + 1] == '\\')) i++;
                cur += s[i];
            }
            if (i >= s.size()) return ok = false, v;
            have = true;
            i++;
        } else if (c == '\\' && i + 1 < s.size()) {
            cur += s[i + 1];
            have = true;
            i += 2;
        } else {
            cur += c;
            have = true;
            i++;
        }
    }
    if (have) v.push_back(cur);
    return v;
}

static int run(int argc, char ** argv);

// engine errors (bad arguments, a prompt over the context, non-sticky CUDA errors) end the run with a message and
// exit code 1; sticky CUDA errors exit inside the engine with code 4, a watchdog timeout with code 3
int main(int argc, char ** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception & e) {
        fprintf(stderr, "[mk] error: %s\n", e.what());
        return 1;
    }
}

static int run(int argc, char ** argv) {
    std::string model, draft_vocab, batch_file;
    int ctx = 8192;
    bool bench_phases = false;
    std::string score_file; // file with whitespace-separated continuation token ids to teacher-force
    // teacher-forced comparison with a reference file; its sequences go through the batched prefill
    // kernels, the decode megakernel, or (mixed) the unscored prefix through the former, like a prompt,
    // and the scored positions through the latter (the first half and the rest when all are scored)
    std::string kld_file, ref_out, score_path = "prefill";
    std::string kld_specs; // --kld-specs: ';'-separated MK_KV_EMU_SPEC values, scored in turn (MK_KV_EMU builds)
    int kv_mode = -1;
    Job job0; // the command line's job (the defaults of every --batch line)
    const std::vector<std::string> av(argv, argv + argc);
    for (size_t i = 1; i < av.size(); i++) {
        if (job_arg(job0, av, i)) continue;
        const std::string & a = av[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= av.size()) usage();
            return av[++i];
        };
        if (a == "-m") model = next();
        else if (a == "-c") ctx = atoi(next().c_str());
        else if (a == "--draft-vocab") draft_vocab = next();
        else if (a == "--bench-phases") bench_phases = true;
        else if (a == "--score") score_file = next();
        else if (a == "--kld") kld_file = next();
        else if (a == "--ref-out") ref_out = next();
        else if (a == "--score-path") score_path = next();
        else if (a == "--kld-specs") kld_specs = next();
        else if (a == "--batch") batch_file = next();
        else if (a == "--kv") { const std::string k = next(); kv_mode = k == "f16" ? 0 : k == "q8" ? 1 : -1; }
        else usage();
    }
    if (model.empty() || (kld_file.empty() && !ref_out.empty())) usage();
    if (score_path != "prefill" && score_path != "decode" && score_path != "mixed") usage();
    // batch jobs: parsed up front (a bad line fails before the model load)
    std::vector<Job> jobs;
    if (!batch_file.empty()) {
        std::istringstream in(read_file(batch_file));
        std::string line;
        int ln = 0;
        while (std::getline(in, line)) {
            ln++;
            size_t f = line.find_first_not_of(" \t\r");
            if (f == std::string::npos || line[f] == '#') continue;
            bool ok;
            std::vector<std::string> v = split_line(line, ok);
            if (!ok) {
                fprintf(stderr, "%s:%d: unbalanced quotes\n", batch_file.c_str(), ln);
                return 1;
            }
            v.insert(v.begin(), "batch");
            Job j = job0;
            for (size_t i = 1; i < v.size(); i++)
                if (!job_arg(j, v, i)) {
                    fprintf(stderr, "%s:%d: '%s' is not a job option (the model, -c, --kv, --draft-vocab are global)\n",
                            batch_file.c_str(), ln, v[i].c_str());
                    return 1;
                }
            if (j.out.empty()) {
                fprintf(stderr, "%s:%d: a batch job needs -o <prefix>\n", batch_file.c_str(), ln);
                return 1;
            }
            jobs.push_back(j);
        }
        if (jobs.empty()) {
            fprintf(stderr, "%s: no jobs\n", batch_file.c_str());
            return 1;
        }
        if (!kld_file.empty() || !score_file.empty() || bench_phases) {
            fprintf(stderr, "--batch runs generation jobs only (no --kld / --score / --bench-phases)\n");
            return 1;
        }
    }
    setenv("MK_CK_SLOTS", "1", 0); // (the CLI restores no checkpoints; 8 pinned slots cost ~2 s per process under WSL2)

    // the tokenizer (libllama, vocab only) loads on a thread alongside the engine (MK_VOCAB_SERIAL=1: before it)
    llama_model * lm = nullptr;
    double vocab_s = 0;
    auto load_vocab = [&] {
        const auto t_vocab = std::chrono::steady_clock::now();
        llama_backend_init();
        llama_model_params mp = llama_model_default_params();
        mp.vocab_only = true;
        lm = llama_model_load_from_file(model.c_str(), mp);
        vocab_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_vocab).count();
    };
    struct VJoin {
        std::thread t;
        ~VJoin() {
            if (t.joinable()) t.join();
        }
    } vj;
    if (getenv("MK_VOCAB_SERIAL") && atoi(getenv("MK_VOCAB_SERIAL")) != 0) load_vocab();
    else vj.t = std::thread(load_vocab);
    const llama_vocab * vocab = nullptr;
    auto need_vocab = [&] {
        if (vocab) return;
        if (vj.t.joinable()) vj.t.join();
        if (!lm) {
            fprintf(stderr, "failed to load vocab\n");
            exit(1);
        }
        vocab = llama_model_get_vocab(lm);
        fprintf(stderr, "[mk] vocab loaded in %.2f s\n", vocab_s);
    };

    auto piece = [&](int id) {
        need_vocab();
        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, false);
        return std::string(buf, n > 0 ? n : 0);
    };
    auto tokenize = [&](const std::string & t) {
        need_vocab();
        std::vector<llama_token> v(t.size() + 16);
        const int k = llama_tokenize(vocab, t.c_str(), (int) t.size(), v.data(), (int) v.size(), false, true);
        if (k < 0) { fprintf(stderr, "tokenize failed\n"); exit(1); }
        return std::vector<int>(v.begin(), v.begin() + k);
    };

    if (!kld_file.empty()) {
        // --kld-specs (MK_KV_EMU builds): the whole reference is scored once per ';'-separated MK_KV_EMU_SPEC value with
        // one model load, each report preceded by "### KV_EMU <spec>"
        std::vector<std::string> specs;
        for (size_t i = 0; i <= kld_specs.size();) {
            size_t j = kld_specs.find(';', i);
            if (j == std::string::npos) j = kld_specs.size();
            specs.push_back(kld_specs.substr(i, j - i));
            i = j + 1;
        }
        size_t longest = 0;
        {
            ls::Session probe(kld_file, "", {}, MKEngine::n_vocab());
            for (const ls::Seq & q : probe.seqs()) longest = std::max(longest, q.tok.size());
        }
        MKEngine eng(model, std::max(ctx, (int) longest + 64), 0, "", kv_mode);
        for (const std::string & sp : specs) {
            if (!kld_specs.empty()) {
                eng.set_kv_emu(sp);
                printf("### KV_EMU %s\n", sp.c_str());
                fflush(stdout);
            }
            ls::Session ses(kld_file, ref_out, {}, MKEngine::n_vocab());
            for (size_t s = 0; s < ses.seqs().size(); s++) {
                const ls::Seq & q = ses.seqs()[s];
                const int N = (int) q.tok.size();
                const int mk_from = score_path == "decode" ? 0 : score_path == "prefill" ? N : q.first > 0 ? q.first : N / 2;
                eng.score_logits(q.tok, mk_from, q.first, [&](int i0, int n, const float * l) { ses.rows((int) s, i0, n, l); });
                fprintf(stderr, "\r%zu / %zu sequences", s + 1, ses.seqs().size());
            }
            fprintf(stderr, "\n");
            ses.report(piece);
            fflush(stdout);
        }
        return 0;
    }

    int nd_engine = job0.n_draft; // batch: the engine's depth is the deepest job's (each job sets its own)
    if (!jobs.empty()) {
        nd_engine = 0;
        for (const Job & j : jobs) nd_engine = std::max(nd_engine, j.n_draft);
    }
    MKEngine eng(model, ctx, nd_engine, draft_vocab, kv_mode);
    eng.set_sampling(job0.temp, job0.top_p, job0.top_k, job0.min_p, job0.seed);
    eng.set_penalty(job0.pres, job0.freq);
    if (bench_phases) {
        eng.bench_phases();
        return 0;
    }

    // chat template of the model (llama.cpp renders the same): with thinking, an optional reasoning-effort
    // system message and an open <think> block; without, an empty think block
    auto gen_prompt_of = [](const Job & J) {
        return J.think.empty() ? std::string("<|im_start|>assistant\n<think>\n\n</think>\n\n") : std::string("<|im_start|>assistant\n<think>\n");
    };
    auto prompt_of = [&](const Job & J) {
        auto reasoning_line = [&]() -> std::string {
            if (J.think == "xhigh")
                return "Reasoning effort is set to xhigh. Please think carefully through the task, validate key assumptions, "
                       "consider plausible alternatives, and prioritize correctness, consistency, and clarity in the final answer.";
            if (J.think == "low")
                return "Reasoning effort is set to low. Keep your thinking brief and focused, moving directly to the conclusion "
                       "without unnecessary elaboration.";
            if (!J.think.empty() && J.think != "medium") { fprintf(stderr, "--think: xhigh, medium or low\n"); exit(1); }
            return "";
        };
        const std::string sys = reasoning_line().empty() ? "" : "<|im_start|>system\n" + reasoning_line() + "<|im_end|>\n";
        const std::string text = J.raw ? J.prompt : sys + "<|im_start|>user\n" + J.prompt + "<|im_end|>\n" + gen_prompt_of(J);
        std::vector<int> ids = tokenize(text);
        if (!J.prompt_ids_file.empty()) {
            std::ifstream f(J.prompt_ids_file);
            ids.clear();
            int id;
            while (f >> id) ids.push_back(id);
        }
        return ids;
    };
    if (!score_file.empty()) {
        const std::vector<int> prompt_ids = prompt_of(job0);
        std::ifstream f(score_file);
        std::vector<int> cont;
        int id;
        while (f >> id) cont.push_back(id);
        std::vector<int> all = prompt_ids;
        all.insert(all.end(), cont.begin(), cont.end());
        std::vector<int> pred = eng.score(all);
        int agree = 0, total = 0, first_bad = -1;
        for (size_t i = prompt_ids.size() - 1; i + 1 < all.size(); i++) {
            total++;
            if (pred[i] == all[i + 1]) agree++;
            else if (first_bad < 0) first_bad = (int) (i + 1 - prompt_ids.size());
        }
        printf("SCORE agree %d / %d (%.2f%%), first disagreement at continuation index %d\n", agree, total,
               100.0 * agree / total, first_bad);
        return 0;
    }

    // one job: prompt (+ an optional second turn), output to stdout, stats to stderr
    auto run_job = [&](const Job & J) {
        eng.set_n_draft(J.n_draft);
        eng.set_sampling(J.temp, J.top_p, J.top_k, J.min_p, J.seed);
        eng.set_penalty(J.pres, J.freq);
        const std::string gen_prompt = gen_prompt_of(J);
        std::vector<int> prompt_ids = prompt_of(J);
        // one turn: prefill (or append) `in`, then decode up to n_predict tokens or end of turn
        bool first_turn = true;
        auto turn = [&](const std::vector<int> & in, const char * tag) {
            const MKStats s0 = eng.stats;
            auto t0 = std::chrono::steady_clock::now();
            int first;
            if (first_turn && !J.state_dir.empty()) {
                int how = 0;
                first = eng.prefill_state(in, J.state_dir, &how);
            } else {
                first = eng.prefill(in);
            }
            auto t1 = std::chrono::steady_clock::now();
            if (first_turn && J.digest) {
                const std::vector<unsigned long long> d = eng.state_digest();
                unsigned long long h = 0xcbf29ce484222325ULL;
                for (unsigned long long w : d) h = (h ^ w) * 0x100000001b3ULL, h ^= h >> 31;
                fprintf(stderr, "[mk] state digest after the prefill: %016llx (%zu words)\n", h, d.size());
            }
            first_turn = false;
            std::vector<int> out = {first};
            bool eog = llama_vocab_is_eog(vocab, first);
            while (!eog && (int) out.size() < J.n_predict) {
                size_t before = out.size();
                eng.step(out);
                for (size_t i = before; i < out.size(); i++) {
                    if (llama_vocab_is_eog(vocab, out[i])) {
                        out.resize(i + 1);
                        eog = true;
                        break;
                    }
                }
            }
            if ((int) out.size() > J.n_predict) out.resize(J.n_predict);
            auto t2 = std::chrono::steady_clock::now();
            if (!J.quiet) {
                std::string str;
                for (int id : out) str += piece(id);
                printf("%s\n", str.c_str());
            }
            if (J.print_ids) {
                printf("IN_%s:", tag);
                for (int id : in) printf(" %d", id);
                printf("\n%s:", tag);
                for (int id : out) printf(" %d", id);
                printf("\n");
            }
            const double pf = std::chrono::duration<double>(t1 - t0).count();
            const double dc = std::chrono::duration<double>(t2 - t1).count();
            const int n_dec = (int) out.size() - 1;
            const long cyc = eng.stats.cycles - s0.cycles, dr = eng.stats.drafted - s0.drafted, ac = eng.stats.accepted - s0.accepted;
            fprintf(stderr, "\n[mk] prompt %d tok in %.3f s (%.1f tok/s)\n", (int) in.size(), pf, in.size() / pf);
            fprintf(stderr, "[mk] decode %d tok in %.3f s = %.2f tok/s | cycles %ld, accepted %ld/%ld drafts (%.1f%%), %.2f tok/cycle\n",
                    n_dec, dc, n_dec / dc, cyc, ac, dr, dr ? 100.0 * ac / dr : 0.0, cyc ? (double) n_dec / cyc : 0.0);
            return std::make_pair(out, eog);
        };
        const int plen = (int) prompt_ids.size();
        auto [out1, eog1] = turn(prompt_ids, "IDS");
        if (!J.turn2.empty()) {
            // the engine may have run past the end of turn (or n_predict) within the last cycle
            eng.truncate(plen + (int) out1.size());
            const std::string t = (eog1 ? "\n" : "<|im_end|>\n") + std::string("<|im_start|>user\n") + J.turn2 + "<|im_end|>\n" + gen_prompt;
            turn(tokenize(t), "IDS2");
        }
    };

    if (jobs.empty()) {
        run_job(job0);
    } else {
        // each job's stdout / stderr go to <prefix>.out / .err (as a separate run's would, without the load lines)
        const auto tb = std::chrono::steady_clock::now();
        for (size_t k = 0; k < jobs.size(); k++) {
            const Job & J = jobs[k];
            fflush(stdout);
            fflush(stderr);
            const int o1 = dup(1), o2 = dup(2);
            const int f1 = open((J.out + ".out").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            const int f2 = open((J.out + ".err").c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (f1 < 0 || f2 < 0) {
                fprintf(stderr, "cannot write %s.out / .err\n", J.out.c_str());
                return 1;
            }
            dup2(f1, 1);
            dup2(f2, 2);
            close(f1);
            close(f2);
            const auto tj = std::chrono::steady_clock::now();
            bool failed = false;
            try {
                eng.reset();
                run_job(J);
            } catch (const std::exception & e) {
                fprintf(stderr, "[mk] error: %s\n", e.what());
                failed = true;
            }
            const double sj = std::chrono::duration<double>(std::chrono::steady_clock::now() - tj).count();
            fflush(stdout);
            fflush(stderr);
            dup2(o1, 1);
            dup2(o2, 2);
            close(o1);
            close(o2);
            fprintf(stderr, "[mk] batch job %zu / %zu (%s) %s in %.2f s\n", k + 1, jobs.size(), J.out.c_str(), failed ? "FAILED" : "done", sj);
            if (failed) return 1;
        }
        fprintf(stderr, "[mk] batch: %zu jobs in %.2f s after the load\n", jobs.size(),
                std::chrono::duration<double>(std::chrono::steady_clock::now() - tb).count());
    }
    llama_model_free(lm);
    return 0;
}
