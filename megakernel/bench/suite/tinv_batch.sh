#!/usr/bin/env bash
# Greedy T-invariance test on the q8 KV cache: the greedy output must be identical for every draft depth.  Every
# (prompt, depth) job runs in ONE process when the binary has --batch (one model load, the engine reset between jobs;
# each job's ids equal a separate run's), else one process per job.
#
#   tinv_batch.sh BIN OUTDIR ["depths"] ["prompt indexes"] [n tokens]       env MODEL (required), CTX (default 8192)
#
# Prompts are the lines of ../prompts.txt.  Writes OUTDIR/p<i>_d<d>.ids (the "IDS:" lines) and prints each job's decode
# line; compare the .ids of different depths of one prompt with `cmp`.
BIN=$1; OUT=$2; DEPTHS=${3:-"0 4"}; PIDX=${4:-"0 1 2 3"}; N=${5:-512}
[ -x "$BIN" ] && [ -n "$OUT" ] && [ -f "${MODEL:-}" ] || { sed -n '2,9p' "$0"; exit 2; }
HERE=$(cd "$(dirname "$0")" && pwd)
DV=${DV:-$(dirname "$BIN")/draft_vocab_32k.bin}
mkdir -p "$OUT"
mapfile -t P < <(grep -v '^\s*$' "$HERE/../prompts.txt")
for i in $PIDX; do printf '%s' "${P[$i]}" > "$OUT/.prompt_p$i"; done
COMMON=(-m "$MODEL" -n "$N" --kv q8 -c "${CTX:-8192}" --draft-vocab "$DV" --ids --quiet)
rc=0
if "$BIN" 2>&1 | grep -q -- '--batch'; then
    J=$OUT/.batch_jobs.txt; : > "$J"
    for i in $PIDX; do for d in $DEPTHS; do
        echo "-f '$OUT/.prompt_p$i' -d $d -o '$OUT/p${i}_d${d}'" >> "$J"
    done; done
    "$BIN" "${COMMON[@]}" --batch "$J" > "$OUT/batch.out" 2> "$OUT/batch.err"; rc=$?
else
    for i in $PIDX; do for d in $DEPTHS; do
        "$BIN" "${COMMON[@]}" -f "$OUT/.prompt_p$i" -d "$d" > "$OUT/p${i}_d${d}.out" 2> "$OUT/p${i}_d${d}.err" || rc=$?
    done; done
fi
for i in $PIDX; do for d in $DEPTHS; do
    grep '^IDS:' "$OUT/p${i}_d${d}.out" > "$OUT/p${i}_d${d}.ids" 2>/dev/null
    echo "p$i d$d $(grep -o 'decode.*' "$OUT/p${i}_d${d}.err" 2>/dev/null)"
done; done
[ $rc = 0 ] || { echo "run failed (exit $rc), see $OUT/*.err"; exit $rc; }
echo DONE
