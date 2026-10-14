#!/usr/bin/env bash
# One decode run with the long-context prefill restored from a post-prefill state file (qwen38_mk --state-dir): the
# first run of a (binary, model, ctx, KV format, MK_* settings, prompt) prefills and saves the file, later runs restore it
# and measure decode only.  Needs a binary with --state-dir (the load-time / state-file work).
#
#   decode_state.sh OUTDIR LABEL BIN short|p0|38|64|101 DEPTH [prof]       env MODEL (required), BENCH_PROMPT_DIR
#
#   short  the 4 prompts of ../prompts.txt, 512 tokens each (no state files: their prefills are short)
#   p0     prompt 0, 256 tokens      38 / 64 / 101   pf38k / pf64k / pf101k.txt (make_prompts.py), 256 tokens, --raw
#   prof=1 MK_PROFILE=1.   MK_STATE_DIR: where the files go (default ~/mk_state; ~1.5 GB at 38k, ~3.9 GB at 101.6k, q8).
#   NOSTATE=1: plain prefill.  EXTRA_DIGEST=1: --digest.  Extra env (MK_KV, MK_KV_WIN, ...) is inherited and part of the state key.
O=$1; label=$2; B=$3; ctx=$4; D=$5; prof=${6:-0}
[ -x "$B" ] && [ -f "${MODEL:-}" ] && [ -n "$ctx" ] && [ -n "$D" ] || { sed -n '2,10p' "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
PD=${BENCH_PROMPT_DIR:-bench_data/prompts}
DV=${DV:-$(dirname "$B")/draft_vocab_32k.bin}
SD=${MK_STATE_DIR:-$HOME/mk_state}
mkdir -p "$O" "$SD"
envs=()
[ "$prof" = 1 ] && envs+=(MK_PROFILE=1)
st=(--state-dir "$SD")
dg=(); [ -n "${EXTRA_DIGEST:-}" ] && dg=(--digest)
[ -n "${NOSTATE:-}" ] && st=()
echo "== $label $(date +%T) $(nvidia-smi --query-gpu=clocks.sm,clocks.mem,temperature.gpu --format=csv,noheader 2>/dev/null) MK_KV=${MK_KV:-} WIN=${MK_KV_WIN:-} d$D"
case $ctx in
    short)
        mapfile -t P < <(grep -v '^\s*$' "$HERE/../prompts.txt")
        for i in 0 1 2 3; do
            env "${envs[@]}" MK_PF_CAP=1024 "$B" -m "$MODEL" -p "${P[$i]}" -n 512 -c 40000 -d "$D" --kv q8 --draft-vocab "$DV" --quiet --ids \
                > "$O/${label}_p$i.out" 2> "$O/${label}_p$i.err"
            grep -E '\[mk\] decode' "$O/${label}_p$i.err" | sed "s/^/p$i /"
        done
        exit 0 ;;
    p0) args=(-p "$(grep -v '^\s*$' "$HERE/../prompts.txt" | sed -n 1p)" -n 256 -c 40000); envs+=(MK_PF_CAP=1024); st=() ;;
    38) args=(-f "$PD/pf38k.txt" -n 256 -c 40000 --raw); envs+=(MK_PF_CAP=2048) ;;
    64) args=(-f "$PD/pf64k.txt" -n 256 -c 70000 --raw); envs+=(MK_PF_CAP=1024) ;;
    101) args=(-f "$PD/pf101k.txt" -n 256 -c 102400 --raw); envs+=(MK_PF_CAP=1024) ;;
    *) echo "unknown context '$ctx'"; exit 2 ;;
esac
t0=$(date +%s.%N)
env "${envs[@]}" "$B" -m "$MODEL" "${args[@]}" -d "$D" --kv q8 --draft-vocab "$DV" --quiet --ids "${st[@]}" "${dg[@]}" \
    > "$O/$label.out" 2> "$O/$label.err"
t1=$(date +%s.%N)
grep -E '\[mk\] (KV cache|prompt|decode|state file|load )' "$O/$label.err" | cut -c1-200
echo "   wall $(echo "$t1 - $t0" | bc -l | cut -c1-6) s"
[ "$prof" = 1 ] && grep -E 'mean .* ms|^  (attn_core|mtp_attn_core|attn_comb) .*total' "$O/$label.err"
true
