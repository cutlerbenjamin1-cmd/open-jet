# Title

megakernel: opt-in low-bit KV cache (MK_KV per-slot q4..q7 formats, K / V rotation, MK_KV_WIN q8 recent window, presets D / D')

# Summary

The default KV cache stays q8_0. This PR adds an opt-in, near-lossless smaller cache: `MK_KV` sets the bits of K and V per
attention slot (q4 to q7, optionally Hadamard-rotated), and `MK_KV_WIN=2048` keeps the most recent 2048 positions in a q8
ring, which is where nearly all of the low-bit loss sits. The recommended preset is
`MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048` ("D'"): it takes 1,085 MiB less VRAM than q8 at 120k context (19,922 vs 21,007 MiB
whole engine), so the same VRAM holds ~162k tokens instead of 120k, and a 262,144-token context loads on a 24 GB card
(UD-Q4_K_XL peak 24,180 MiB). KLD against llama.cpp at 32k is 0.00239 for D' vs 0.00232 for the same build with q8 (+3%), and
the greedy ids equal q8's at 38k and 101.6k for the lighter preset C. The cost is speed: with D' decode is 1-3% slower per
cycle at 38k / 101.6k and long prefill ~4% slower (prefill -7% / -3% before the KBITS instantiation, which recovered 2.5% /
1.3%). Nothing changes unless `MK_KV` is set; with it unset the output is byte-identical to the previous build.

# What changed

## Environment variables (all opt-in, library reads them at engine construction)

| variable | default | effect |
|---|---|---|
| `MK_KV` | empty (all-q8) | per-slot K / V formats, see syntax below. A non-empty value selects the q8 cache in `--kv auto`; with `--kv f16` the load throws. |
| `MK_KV_WIN` | 0 (off) | `W`: every low-bit side keeps its most recent rows in a q8 ring of `(ceil(W / 640) + 1) x 640 + 32` rows per kv head (W = 2048: 3.3 MiB per side; 56 / 113 MiB for V / K+V). Needs a low-bit `MK_KV`. |
| `MK_KV_WIN_ALL` | 0 | test knob: rings for the q8 sides too (output must equal plain q8: exercises ring write / read / refill). |
| `MK_MTP_WIN` | 0 | measurement knob: the MTP block's decode attention sees the first tile and the last N positions (drafts only, output exact). Negative result, kept for measurement. |
| `MK_ATTN_S8` | 1 | (from PR 3) low-bit unrotated or rotated K stays on the int8 prefill attention; `=0` forces pk_attn2 (fp16) everywhere. |
| build: `-DMK_DA_Q4=1` (default 1), `-DMK_AQ_EARLY` (default 0), `-DMK_XP_SM` (default 0) | | compile-time switches of the low-bit decode attention; the last two are measured-neutral/negative knobs. |
| build: `-DMK_KV_EMU=1` | 0 | accuracy-experiment build: `MK_KV_EMU_SPEC` rounds cache slots to a lower-bit grid and stores them as q8 (`kv_store_emu`); `--kld-specs "a;b;c"` scores a reference once per spec with one model load. |

`MK_KV` syntax: comma-separated `<K|V|*><slot>=<bits>[r]`; bits 4..8 (8 = q8_0); slot 0..15 = the 16 attention layers
(layers 3, 7, ..., 63), 16 = the MTP block; an empty slot or `*` = all slots; `r` = stored Hadamard-rotated (K: the queries
get the same orthonormal 256-point rotation; V: the head's attention output is rotated back). `K*=6,V*=5,*16=8` therefore
means: all trunk K at 6 bits, all V at 5 bits, and the MTP slot (both K and V) back at q8.

## Storage formats (f138d36, 380d2dd)

q4 = q4_0 rounding (16 levels, fp16 scale per 32 values); q5 / q6 / q7 = the q4 nibble plus a 1 / 2 / 3-bit high plane
(`kv_store_qn`); the slots are packed in `kc` / `vc` in order (an all-q8 cache gives the old `[slot][head][ctx][HD]` addresses).
Per slot and K / V separately. `r` rotation of K (queries rotated in `pk_attn_prep` / the decode prologue) and of V (output
rotated back in `attn_combine` for decode and `pk_attn_out` for prefill).

## Recent-window q8 ring (9b05adb, f292074)

`kv_store` writes the row to the main buffer (its MK_KV format) and to the ring (q8, same rotation). A tile at position p0
is read from the ring iff p0 >= `kv_ws(L_ctx)`; the window start is stepped by `ATT_BUCKET` (640) like the attention
partition, so the chunk partition is unchanged and greedy ids stay independent of the draft depth. `ring_rewind(P)` refills
only the ring rows the abandoned continuation overwrote (also called from `MKEngine::truncate`), so a checkpoint restore does
not demote the whole window.

## Kernels

- Decode (680d412, abe5556, f64ff51, ceb7d44, 2ea89b6, eb133a5): `attn_core_q8m` (the "MIX" instantiation of the pipelined q8
  decode attention) reads the per-slot formats; a packed tile is expanded in place to the q8 layout one interval before it is
  read (`kv_rows_expand`: `q = n - 2^(bits-1)`, scale unchanged, so d*q is the same fp16 product as the per-fragment unpack:
  exact), K ring one stage deeper (AQ_NSKM = 4). Tile format / source is decided once per launch (`KvFmt`, `kv_ring_tile`,
  `kv_tile_src`), without the per-tile integer modulo.
- Prefill (f138d36, 251849d, 23905b4, 71ac710, 1548b89, 3d84349): pk_attn2 handles every format; pk_attn8 (PR 3) takes low-bit
  K by expanding the packed rows to int8 in its staging (`<QROW, KLOW>`, a separate instantiation so the q8 kernel keeps its
  registers), and rotated K runs it too (Q is rotated in fp32 before its fp16 store, so the int8 Q fragments are rotated like
  pk_attn2's). 251849d is a bug fix: pk_attn8 used the uniform `[slot][head][ctx][HD]` offset, so every q8 slot after a q4 slot read
  another slot's rows in prefill (b1000r mixed KLD 94.14% / 0.0355 instead of 98.2% / 0.0026).
- 3d84349 (see Commits, depends on PR 5): `pk_attn8<..., KB, VB>` compile-time K6 / V5 instantiation for D / D'.
- Low-bit slots at T > 5 (5245602): with `MK_MAXT=8` builds (PR 1), `-d 5..7` works with a low-bit `MK_KV` (the wide pass calls
  `attn_core_q8m`); the ring gets `kvrc = (Wb + 1) x 640 + 32` rows so the MTP block's draft rows past a partition boundary do not
  overwrite the window's oldest MTP-slot rows (affected only drafts; outputs and KLD unchanged). Before it, `n_draft > 4` with a
  low-bit `MK_KV` threw.
- Tooling (91b0551): KV emulation builds and batch KLD scoring (`--kld-specs`), used to choose the formats.

# The user story: which preset to use

Whole-engine VRAM from `MK_MEMLOG`: q8 at 120k 21,007 MiB; D' at 120k 19,922 MiB; D' at 162k 21,010 MiB
[notes/agent_kvq.md "kvd7 summary" / run3-run4, MAXT 8 build, `-d 7`]. The q8 cache is 36,992 B / token (17 slots x 4 kv heads x
(256 + 256 + 32) B; 4,233 MiB at 120k, of which the MTP slot is 249 MiB) [notes/agent_vram.md]. Presets, "MiB" = saved at 120k
net of the rings; ctx = what fits in today's 120k-q8 VRAM; KL columns are mean KL vs the llama.cpp reference (q8 baselines
mixed 0.00113, 32k 0.00219, 64k 0.00156; fp16-cache floors 0.00113 / 0.00226 / 0.00161); decode / prefill = cost vs q8 at 38k /
101.6k [notes/agent_kvq.md "kvq4 summary"; LEDGER "From agent_kvq4"; build 2ea89b6, `-d 4`, A-B-A]:

| preset | `MK_KV` (with `MK_KV_WIN=2048`) | MiB saved | ctx | KL mixed | KL 32k | KL 64k | decode ms/cycle | prefill |
|---|---|---|---|---|---|---|---|---|
| C | `V*=5` | 691 | 144k | = q8 | 0.00228 (+4%) | 0.00159 (+2%) | +1.2% / +3.7% | -1.3% / -2.7% |
| D | `K*=6,V*=5` | 1132 | 165k | = q8 | 0.00221 (+1%) | 0.00164 (+5%) | +2.9% / +5.0% | -3.3% / -6.7% |
| **D'** | `K*=6,V*=5,*16=8` | ~1085 | ~162k | = D | = D (see below) | n/m | see below | see below |
| E | `K*=6r,V*=4r` | 1381 | 180k | 0.00115 (+2%) | 0.00230 (+5%) | 0.00172 (+10%) | +2.4% / +4.9% | -2.0% / -4.2% |

(n/m = not measured separately; D' differs from D only in the MTP slot, whose K / V stay q8, so the main slots, greedy ids and
32k KLD equal D's.) C: greedy ids equal q8 at 38k and 101.6k. E is at the edge of "no visible loss" at 64k (+10%); with
`MK_ATTN_S8=0` E's 64k KL is 0.00162 (+4%) at -12% / -23% prefill.

Recommendation, from the maintainer's own pick (HANDOFF 10-13 20:15): D' whenever a low-bit cache is used. It costs +70 MB
over D for +1.2% / +2.9% tok/s at 101.6k / 38k (the MTP draft reads q8 K / V, so more drafts are accepted). C is the
conservative choice when you want identical greedy text; E when you want the most context. The launcher default remains q8_0
and 120k.

D' on 447a87c (A-B-A, `-DMK_WV=1 -DMK_MAXT=8`, `-d 7`) [notes/agent_kvq.md "kvd7", LEDGER "From agent_kvd7"]:

| | 101.6k | 38k |
|---|---|---|
| q8, `-d 7` | 181.2 tok/s (38.0 ms / cycle) | 178.3 tok/s |
| D', `-d 7` | 169.0 tok/s (38.7 ms / cycle) | 158.2 tok/s |
| D', `-d 4` | 133.8 tok/s | 137.2 tok/s |

D' `-d 7` vs `-d 4`: +26% / +15%. Per cycle D' is +1.7% / -0.8% vs q8 (101.6k / 38k); tok/s is -7% / -11% because D's output
drafts fewer tokens per cycle on these prompts (the notes recommend comparing ms per cycle when formats differ; at short
context D' equals q8 exactly). Prefill: D' -7% / -3% at 101.6k / 38k before the KBITS instantiation (kvq4's numbers for D);
3d84349 then measured D' prefill -2.5% at 101.6k and -1.3% at 38k better than without it (CLI A-B-A, 101.6k: 81.60 | 79.95 |
82.41 s, 38k: 23.54 | 23.21 | 23.50 s) [notes/agent_reuse.md; LEDGER "From agent_reuse2"]. D' prefill vs q8 after it: not
measured directly.

Context reach: D' loads at 262,144 tokens on an RTX 3090 (UD-Q4_K_XL, KV 7.13 GB, peak 24,180 MiB; Q4_K_M peak 23,828 MiB)
[notes/main/final/ctx_probe.out; single load check, not a quality or speed measurement at that length].

Quality vs llama.cpp on real tasks (preset D, the same main-slot formats as D'; megakernel A = wide verify, `-d 4`; llama.cpp
b11548 tuned q8_0 KV; same model, ctx 120k, greedy) [notes/main/qa/RESULTS.md]: 232 / 282 vs 229 / 282 items (GSM8K 50 / 50
both, MMLU-Pro 47 vs 45 of 70, ARC 47 / 47 of 50, MATH-500 30 vs 29 of 40, HumanEval 38 / 38 of 41, NIAH 32k-100k 9 / 9, LongBench
v2 2 / 2 of 12). 7 disagreements, 5 for the megakernel and 2 for llama.cpp: a tie within noise. Caveat in the notes: with 40-70
items per set, differences under ~5 points are noise; KLD is the precise measure.

At the sweep-3 operating point (A + D vs A, 120k ctx, `-d 4`) [notes/main/sweep3/RESULTS.md]: decode short 127.1 vs 127.3 tok/s,
38k 97.2 vs 103.7, 101.6k 98.9 vs 100.3; prefill 101.6k 83.5 vs 78.0 s, 38k 24.20 vs 23.47 s; VRAM idle 20.11 vs 21.24 GB.
(This predates the pk_attn8 low-bit K work and KBITS.)

# Correctness

- Default path byte-identical (greedy ids = `tinv`, KLD mixed) to the build before this series at every gate: 9f25c2f vs
  all-q8 for 380d2dd / 9b05adb / abe5556 / 2ea89b6; q8 on MAXT 8 d0 / 4 / 7 ids equal the wide-verify build, KLD mixed byte-identical
  [agent_kvq.md "kvq2 summary", "kvq3 summary", "kvq4 summary", "kvd7 summary"].
- Low-bit decode kernels reproduce attn_core_t's partials bit for bit: 0 differing partials in 364 configs (n 0..101,600, T 1..5,
  q8 / K4 / V4 / K4V4 / K4rV4 / K8r) [680d412]. In-place expansion is byte-identical to the per-fragment kernels (C and rotated D
  32k KLD, tinv ids) [abe5556].
- Ring: with every sequence inside the window (mixed set), KLD is byte-identical to all-q8; `MK_KV_WIN_ALL=1` gives KLD mixed + 32k
  and server outputs byte-identical to plain q8; greedy ids d0 == d4 on 4 prompts, d0 == d4 == d7 for D / D' / C / E at short
  context, 38k and 101.6k.
- Format study (emulation `MK_KV_EMU`, KL increase vs q8 on the mixed / 32k sets; agent_kvq.md "Scan g0-g3", kvq summaries): all-V
  6r +3.5% / +8%; all-K 7r +2.7% / +4%; K 6r +12% / +4%; V 5r +27% / +16%; with the last 2048 rows at q8, V*=5 is at the fp16
  floor on 32k (0.00220 vs q8 0.00219, fp16 0.00226). Lloyd-Max codebooks, per-row scales, random signs, TurboAngle were
  worse than the plain per-32 absmax grid once rotated; the 3-bit class costs +300% KL here. Unrotated K / V with the window
  is as good as rotated, so the recommended presets are unrotated.
- D' 32k KLD on the MAXT 8 build: 98.974% / 0.00239, equal to D's on that build; the same build with q8 is 0.00232 (B, sweep-3
  RESULTS). On the r5 build: q8 mixed 98.677 / 0.00119, 32k 99.096 / 0.00223, D' 32k 99.170 / 0.00238 (top-1 +0.2 pp, KL +7%)
  [HANDOFF]. Server 2-turn reuse test at `-d 7` runs (ring rewind of 11 rows).
- Gates for the kernels: Q4_K_M ids d0 == d4 == d7 for q8 and D.

# Commits

See commits.txt: 19 commits as assigned (f138d36 .. 3d84349), sum of per-commit numstat 7 files, +1601 / -603 (includes the
bench and README hunks). Changes to the assignment: **3d84349 should move to PR 5**: its `pk_attn8<QROW, KLOW, EXA, KB, VB>`
signature includes the `EXA` (MK_EXACT) parameter that PR 5 introduces (4ca650d, 43d67cd); it is the "KBITS" milestone the
reuse agent implemented (agent_reuse.md). It is listed in commits.txt with a marker and its numbers are in this description in
case it stays in PR 4 in a form with the EXA parameter stripped. Dependencies on other PRs: 680d412 on the pipelined q8
decode attention (PR 2); 5245602 on `MK_MAXT=8` / wide pass (PR 1); 23905b4 / 71ac710 / 1548b89 on pk_attn8 (PR 3).

# Notes for the reviewer

- Opt-in: `MK_KV` unset = no behaviour change (checked above). No flag in the CLI; env only. If the maintainers prefer
  a flag (`--kv-preset d`), the string is the same.
- Risks: (1) the low-bit decode path adds work on warps 4-15; the measured cost is 1.7-5% per cycle at 38k-101.6k and nothing
  latency-shaped is left to hide (ncu, agent_kvq.md "kvq4"). (2) Server: after a rewind to a non-cap-aligned checkpoint,
  "reuse != fresh" outputs diverge (texts of 912 vs 905 reasoning chars, first difference at char 848) with D + ring at `-d 4` and `-d 7`; plain q8,
  q8 with rings on every side, and D without the ring are clean (kvd7 run7 / run7b). Not shown to come from this code; likely the
  decode-vs-prefill non-bitwise class; open. `MK_EXACT` (PR 5) makes D' reuse bitwise. (3) `MK_MTP_WIN` costs 5-7% tok/s
  (acceptance falls because of the sinks); do not use.
- Known limits: low-bit presets need the q8 cache; E (rotated V at 4 bits) reaches +10% KL at 64k; per-slot allocation was
  tried and closed (the window dominates it); RoPE tables on the fly and MTP KV ring rejected.
- Follow-ups listed in the notes: W48 kernel tuning at `-d 7`, D' prefill KBITS (done: 3d84349), a rotated-K prefill form.
- sm75: not applicable as is (see PORT.md): the int8 prefill path needs `mma.m16n8k32` s8 and the decode pipeline cp.async.
