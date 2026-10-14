#!/usr/bin/env bash
# Max-context probe: the CLI at several large contexts on one or more models: load + a tiny prompt; reports the KV line
# and the whole-card VRAM peak (nvidia-smi sampled every 200 ms).  Startup fit only; the full-length prefill at the chosen
# context is part of the max-context set itself (maxctx_client.py).
#
#   ctx_probe.sh -b BIN -o OUTDIR -m model.gguf [-m other.gguf ...] [-c "262144 250000 230000"] [-d 7] [-g GPU]
#
#   BIN     qwen38_mk (or qwen38_mixed) executable        OUTDIR  per-run .out / .err / smi files
#   -c      contexts to try, largest first (default "262144 250000 230000")
# Settings that change the memory footprint (MK_KV, MK_KV_WIN, MK_PF_CAP, ...) are taken from the environment.
set -u
BIN=; OUT=; CTXS="262144 250000 230000"; D=7; GPU=; MODELS=()
while getopts "b:o:m:c:d:g:" o; do
    case $o in
        b) BIN=$OPTARG ;; o) OUT=$OPTARG ;; m) MODELS+=("$OPTARG") ;; c) CTXS=$OPTARG ;; d) D=$OPTARG ;; g) GPU=$OPTARG ;;
        *) exit 2 ;;
    esac
done
[ -x "$BIN" ] && [ -n "$OUT" ] && [ ${#MODELS[@]} -gt 0 ] || { sed -n '2,9p' "$0"; exit 2; }
DV=$(dirname "$BIN")/draft_vocab_32k.bin
smi=(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200)
[ -n "$GPU" ] && smi+=(-i "$GPU")
mkdir -p "$OUT"
for model in "${MODELS[@]}"; do
    m=$(basename "$model" .gguf)
    for n in $CTXS; do
        "${smi[@]}" > "$OUT/smi_${m}_$n.txt" &
        sp=$!
        "$BIN" -m "$model" -p "Say hi." -n 16 -c "$n" -d "$D" --kv q8 --draft-vocab "$DV" --quiet \
            > "$OUT/${m}_$n.out" 2> "$OUT/${m}_$n.err"
        rc=$?
        kill $sp 2>/dev/null
        echo "$m ctx $n: rc $rc | $(grep -m1 -E 'KV cache' "$OUT/${m}_$n.err" | cut -c1-120) | peak $(sort -n "$OUT/smi_${m}_$n.txt" | tail -1) MiB | $(grep -m1 -iE 'error|out of memory|fail' "$OUT/${m}_$n.err" | cut -c1-120)"
    done
done
echo "PROBE_DONE $(date +%T)"
