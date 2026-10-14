# README.md changes for PR 4 (low-bit KV cache)

Written against origin/master (4d4db90) `megakernel/README.md`. Our fabce52 README documents none of `MK_KV` beyond one clause
("q4 `MK_KV` cache slots run the pipelined kernel too"), so this text is new. Numbers: see PR.md for sources and conditions.

## Edit 1: new section "KV cache"

Where: after "## Prefill" and before "## Turing (RTX 8000)".

> ## KV cache
>
> The cache is fp16 when it fits the requested context and q8_0 otherwise (`--kv auto|f16|q8`): per position 36,992 bytes
> for q8_0 (17 attention slots - 16 layers and the MTP block - x 4 kv heads x 256 + 256 + 32 scale bytes), 4.1 GiB at 120K tokens.
> An opt-in low-bit cache saves up to ~1.1 GiB of that at 120K with no visible quality change. It needs the q8 cache and is
> off unless `MK_KV` is set (a non-empty `MK_KV` selects q8 in `--kv auto`).
>
> `MK_KV` is a comma-separated list of `<K|V|*><slot>=<bits>[r]`: `bits` 4 to 8 (8 = q8_0; 4 to 7 = q4_0-style per-32 scale
> plus a 1 / 2 / 3-bit high plane for 5 / 6 / 7), `slot` 0 to 15 for the 16 attention layers, 16 for the MTP block, empty or `*`
> for all, and `r` stores the side Hadamard-rotated (K: the queries get the same rotation; V: the output is rotated back).
> `MK_KV_WIN=W` keeps the most recent W positions of every low-bit side in a q8 ring (W = 2048: 3.3 MiB per side): measured
> losses of the low-bit formats sit almost entirely in the last 0.5 to 2K positions.
>
> | preset | `MK_KV` + `MK_KV_WIN=2048` | VRAM saved at 120K | context in the same VRAM | KL vs q8 (mixed / 32K / 64K) |
> |---|---|---|---|---|
> | C | `V*=5` | 691 MiB | ~144K | = / +4% / +2% |
> | D | `K*=6,V*=5` | 1132 MiB | ~165K | = / +1% / +5% |
> | **D'** | `K*=6,V*=5,*16=8` | ~1085 MiB | ~162K | = D (the MTP slot stays q8: +70 MB, drafts from q8 K / V) |
> | E | `K*=6r,V*=4r` | 1381 MiB | ~180K | +2% / +5% / +10% |
>
> Use D' for a smaller cache; C if identical greedy text matters; E for the most context. Example:
> `MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048 python server/mk_server.py ...` (the variables are read by the library). Costs, D' vs q8 at 38K / 101.6K:
> decode -0.8% / +1.7% per cycle (`-d 7`), prefill about -3% / -7% (D, before the K6 / V5 kernel specialisation).
> The default stays q8_0. With `MK_MAXT=8` builds low-bit presets work at `-d 5` to `-d 7`; on the default build they need `-d` <= 4.
> `MK_KV_WIN_ALL=1` and `MK_KV_EMU` builds (`-DMK_KV_EMU=1`, `MK_KV_EMU_SPEC`, `--kld-specs`) exist for testing and for
> choosing formats.

(Note for the text above: confirm "sm_80 and later only" wording and the `-d` limits against the port: the `-d > 4` refusal is
the MAXT 5 build's guard.)

## Edit 2: CLI-flags paragraph

Where: "Build / run", paragraph starting "CLI flags:", after "`MK_EMBD_DEVICE=1` (keep `token_embd` in GPU memory),". Insert:

> `MK_KV` / `MK_KV_WIN` (low-bit KV cache, see KV cache),

## Edit 3: Prefill / How it gets its speed attention bullet

Where: "How it gets its speed", the "Attention" bullet. Append one sentence:

> Slots stored below 8 bits (`MK_KV`) are expanded to the q8 layout in shared memory one interval ahead of use, so the MMA path
> and the partials are the same as for a q8 cache.

## Edit 4: Limitations

Where: "Limitations", last bullet. Append:

> With a low-bit `MK_KV` and `MK_KV_WIN`, a conversation re-opened after a rewind to a non-cap-aligned checkpoint can generate
> a different continuation than a fresh prefill of the same text (the refilled window rows hold the low-bit values); the output
> is exact when `MK_EXACT` is on (see Prompt reuse).

(Drop the last clause if PR 5 is not in.)

## Edit 5: Files

Where: "## Files". If `bench/dattn_bench.cu` is listed, no change; otherwise add
"`bench/dattn_bench.cu` - decode attention microbench (`KQ4` / `VQ4` / `KROT` select KV formats)" (680d412 regenerates it).
