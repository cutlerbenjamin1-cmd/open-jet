#!/usr/bin/env bash
# Decode A-B-A at long context with post-prefill state files (decode_state.sh): A | B | A' for each KV form, so a
# difference between builds A and B can be told from drift of the card (A vs A').
#
#   state_aba.sh OUTDIR BIN_A BIN_B "ctxs" "forms" [depth]            env MODEL (required), BENCH_PROMPT_DIR,
#                                                                      BENCH_WRAP (optional command prefix per timed block)
#   ctxs   101 64 38 short       forms  q8, dp (preset D' = MK_KV='K*=6,V*=5,*16=8' MK_KV_WIN=2048; needs the low-bit KV cache)
#   Labels: <form>_<ctx>_{a,b,a2}.  PRIME=1 (default): before the timed blocks, one plain run per (binary, form, ctx) creates
#   the state files, so a timed run only restores them (load + restore + 256 decode tokens, not a full prefill).
#   BENCH_WRAP="gpu-lock --timing" (or any command that takes the block as its arguments) holds the GPU for each block.
HERE=$(cd "$(dirname "$0")" && pwd)
O=$1; A=$2; B=$3; CTXS=$4; FORMS=$5; D=${6:-7}
[ -n "$O" ] && [ -x "$A" ] && [ -x "$B" ] && [ -n "$CTXS" ] && [ -n "$FORMS" ] || { sed -n '2,11p' "$0"; exit 2; }
WRAP=${BENCH_WRAP:-}
DS=$HERE/decode_state.sh
mkdir -p "$O"
envof() { [ "$1" = dp ] && echo "env MK_KV=K*=6,V*=5,*16=8 MK_KV_WIN=2048"; }
block() { if [ -n "$WRAP" ]; then $WRAP bash -c "$1"; else bash -c "$1"; fi; }
if [ "${PRIME:-1}" = 1 ]; then
    cmds="true"
    for h in $CTXS; do
        [ "$h" = short ] && continue
        for fm in $FORMS; do
            for bin in $A $B; do
                cmds="$cmds; $(envof "$fm") bash $DS $O/prime prime_${fm}_${h}_$(basename "$(dirname "$bin")") $bin $h $D > /dev/null"
            done
        done
    done
    echo "## prime $(date +%T)"
    bash -c "$cmds"
fi
for h in $CTXS; do
    cmds="bash $DS $O warm_$h $A p0 $D > /dev/null"
    for fm in $FORMS; do
        e=$(envof "$fm")
        cmds="$cmds; $e bash $DS $O ${fm}_${h}_a $A $h $D; $e bash $DS $O ${fm}_${h}_b $B $h $D; $e bash $DS $O ${fm}_${h}_a2 $A $h $D"
    done
    echo "## block $h $(date +%T)"
    t0=$SECONDS
    block "$cmds"
    echo "## block $h took $((SECONDS - t0)) s"
done
for f in "$O"/*_a.err "$O"/*_b.err "$O"/*_a2.err; do
    [ -e "$f" ] && echo "$(basename "$f" .err): $(grep -o 'decode .*' "$f" | cut -c1-120)"
done
echo "ABA_DONE $(date +%T)"
