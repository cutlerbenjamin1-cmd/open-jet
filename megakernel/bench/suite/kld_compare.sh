#!/usr/bin/env bash
# Score an engine against a llama.cpp reference file (KL divergence, perplexity, top-1 agreement; logit_stats.h).
#
#   kld_compare.sh -w mk|llama -m model.gguf -r ref.bin -L label [-p prefill|decode|mixed] [-b BIN] [-l LLAMA_REF] [-o OUTDIR]
#
#   -w mk     the megakernel CLI (-b, default ../../qwen38_mk) with --kld REF --score-path PATH --kv q8
#   -w llama  tools/llama_ref (-l, default ../../tools/llama_ref) with --kld REF: llama.cpp batched, the noise floor
#   -p        megakernel --score-path (default decode); ignored for llama
#   -o        the log goes to OUTDIR/kld_LABEL.log (default ./benchmark_results/kld)
# The reference is made once with `tools/llama_ref -m model.gguf -f text ... --ref-out ref.bin` (see megakernel/README.md).
# To hold the GPU against other work, wrap the whole call in your lock command: `my-gpu-lock kld_compare.sh ...`.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
MK=$HERE/../../qwen38_mk; LLAMA_REF=$HERE/../../tools/llama_ref; OUT=benchmark_results/kld
WHO=; MODEL=; REF=; SP=decode; LABEL=
while getopts "w:m:r:L:p:b:l:o:" o; do
    case $o in
        w) WHO=$OPTARG ;; m) MODEL=$OPTARG ;; r) REF=$OPTARG ;; L) LABEL=$OPTARG ;; p) SP=$OPTARG ;;
        b) MK=$OPTARG ;; l) LLAMA_REF=$OPTARG ;; o) OUT=$OPTARG ;; *) exit 2 ;;
    esac
done
[ -n "$WHO" ] && [ -f "$MODEL" ] && [ -f "$REF" ] && [ -n "$LABEL" ] || { sed -n '2,10p' "$0"; exit 2; }
mkdir -p "$OUT"
{
    echo "## $LABEL $(date -Is) who=$WHO model=$MODEL ref=$REF path=$SP bin=$([ "$WHO" = llama ] && echo "$LLAMA_REF" || echo "$MK")"
    if [ "$WHO" = llama ]; then
        "$LLAMA_REF" -m "$MODEL" --kld "$REF" 2>&1 | grep -vE 'CUDA Graph|graph warmup|^llama_|^load|^print_info|^graph_reserve|^sched_|^resolve_fused'
    else
        "$MK" -m "$MODEL" --kld "$REF" --score-path "$SP" --kv q8
    fi
    echo "EXIT=$?"
} > "$OUT/kld_$LABEL.log" 2>&1
tail -5 "$OUT/kld_$LABEL.log"
