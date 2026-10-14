# README draft for PR 5 (prompt reuse)

Against upstream's current `megakernel/README.md` (origin/master 4d4db90). Three edits.

## Edit 1: replace the bullet "Successive requests of a conversation only prefill what is new ..." in `## Server`

Replace this bullet:

> - Successive requests of a conversation only prefill what is new: the prompt is matched against the
>   engine's sequence (prompts + emitted tokens), and when the re-rendered history diverges after the
>   end of the previous prompt (e.g. a reformatted tool call) the engine rewinds to a checkpoint
>   saved there. Anything else starts over (batched prefill, ~1500 tok/s).

with:

> - Successive requests of a conversation only prefill what is new: the prompt is matched against the
>   engine's sequence (prompts + emitted tokens). When the new prompt diverges inside what the engine
>   already processed (an edited earlier message, a changed system prompt, a thinking-mode follow-up that
>   re-renders the previous turn, context compaction) the server restores the last DeltaNet checkpoint at or
>   before the divergence and prefills from there; only a prompt with no usable checkpoint starts over
>   (batched prefill, ~1500 tok/s). See [Prompt reuse](#prompt-reuse).

## Edit 2: new subsection, after the last bullet of `## Server` ("`MK_PF_CAP=1024` halves prefill scratch ..."), before `## How it gets its speed`

> ### Prompt reuse
>
> The attention KV rows before a divergence stay valid, but the DeltaNet layers keep a recurrent state that
> cannot be rewound, so the engine saves its state (DeltaNet state, conv ring, h ring: ~0.17 GB) as a
> *checkpoint* and restores one when a prompt diverges. Checkpoints live in pinned host memory (no VRAM) in a pool
> of `MK_CK_SLOTS` (default 8, about 1.4 GB of pinned RAM in total; when the pool is full the one whose removal
> leaves the smallest gap is dropped, so old checkpoints thin out and recent ones stay dense). They are saved
> inside the normal prefill pass, at the places a chat client edits: the generation prompt of the request (the last
> `<|im_start|>`, which is where a thinking-mode follow-up re-renders the turn), the start of each message, and
> every `--ck-every` tokens. Saving does not change the prefill's arithmetic, and the copies overlap the next
> kernels: the save runs under the first decode launch, the restore is issued right after the prompt's tokens are
> uploaded and each group of DeltaNet layers waits only for its own part.
>
> | Setting | Default | Effect |
> |---|---|---|
> | `MK_CK_SLOTS` | 8 | checkpoints in the pool, ~0.17 GB pinned host RAM each; 1 keeps only the latest |
> | `MK_CK_DEVICE=1` | off | keep checkpoints in VRAM (device-to-device copies) instead of pinned host RAM (+0.16 GB VRAM) |
> | `mk_server --ck-every N` | 4096 | also checkpoint every N prompt tokens (snapped to the 32-token scan grid); 0 = off |
> | `mk_server --ck-msgs 0/1` | 1 | checkpoint at each message start |
> | `mk_server --exact-reuse` / `MK_EXACT=1` | off | bitwise-exact reuse, below |
> | `MK_EXACT_SEG` | 4096 | exact mode: key-segment length of the attention fold (positions) |
> | `MK_EXACT_SPLIT=0` | split on | exact mode: do not split attention items across CTAs (test knob) |
>
> With `MK_PF_CAP` below about 340 there are too few staging slots for checkpoints inside the prefill pass
> (below about 170 the prompt is prefilled in two passes around the checkpoint, as before).
>
> **Exact reuse.** By default a reused request is as good as a fresh one but not bit-identical: the chunk composition
> (DeltaNet scan grid, attention split plan) differs from a prefill from token 0, the previous reply's rows come from
> the decode kernels, and a long greedy thinking answer can drift after a few hundred characters. With
> `mk_server --exact-reuse` (`MK_EXACT=1`) the prefill attention is computed as an in-order fold of absolute key
> segments, so any work split gives the same bits; the `MK_KV_WIN` window split is taken per query tile; checkpoints sit
> on the absolute 32-token grid and carry the q8 window rings; and the server never appends to decoded rows: it
> restores the last checkpoint at or before the divergence and re-prefills the rest, including the previous reply. A
> reused request then answers bit for bit as the same request prefilled from token 0 (state-digest probe over every
> K / V row, ring row and recurrent state). The cost is the re-prefill of the previous reply on a verbatim follow-up (a
> 675-token reply at 11.6k context: 477 ms instead of 81 ms prompt time) and +1% to +2% on a fresh 98k or 38k prefill; edits
> cost the same as without it. The default numerics are unchanged when it is off.

## Edit 3: `## Limitations`, replace the bullet "Text only; one sequence and one checkpoint ..."

> - Text only; one sequence: a conversation is reused while it grows, or when it diverges at a point where the
>   engine holds a checkpoint (generation prompts, message starts, every 4096 tokens; see
>   [Prompt reuse](#prompt-reuse)). A change earlier than the oldest checkpoint, or after the pool has dropped it,
>   re-prefills from the last one before it. No slot save / restore (open-jet's low-memory shell swap falls back to
>   re-prompting).

Placement notes. The CLI has no checkpoint flags; `MK_CK_*` and `MK_EXACT*` are environment variables read by the
engine (server and CLI). If PR 4 (KV cache) merges first, the `MK_EXACT` row should mention `MK_KV_WIN`; if it
merges later, the exact-reuse paragraph's sentence about window rings is dropped until it does.
