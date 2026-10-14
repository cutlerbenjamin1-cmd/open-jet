#!/usr/bin/env python3
"""Compare two quantizations of one model from their two llama.cpp reference files (tools/llama_ref --ref-out on the
same token sequences): perplexity of each on the same text, the paired per-token NLL difference (SE by sequence
blocks), top-1 agreement and KL(A || B) over A's top-256 + one tail bucket.  Layout: logit_stats.h (MKREF01).  CPU only.

  python3 ref_ppl.py <a.ref> <b.ref>      e.g. the UD-Q4_K_XL file and the plain Q4_K_M file

Both files must hold the same token sequences (same text, chunking and tokenizer).  A positive NLL difference means A is
worse on that token.  The KL is a lower bound: B's log-probability for a token outside its own top-256 is its tail mass
spread evenly over the missing ids.
"""
import math
import struct
import sys

import numpy as np

K = 256
REC = np.dtype([("lp_next", "<f4"), ("tail", "<f4"), ("id", "<i4", K), ("lp", "<f4", K)])


def load(p):
    with open(p, "rb") as f:
        b = f.read()
    assert b[:7] == b"MKREF01", p
    nv, k, ns = struct.unpack_from("<3i", b, 8)
    assert k == K
    o, seqs = 20, []
    for _ in range(ns):
        n, first = struct.unpack_from("<2i", b, o)
        o += 8
        seqs.append((np.frombuffer(b, "<i4", n, o).copy(), first))
        o += 4 * n
    recs = np.frombuffer(b, REC, offset=o)
    return seqs, recs


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        raise SystemExit(__doc__)
    sa, ra = load(argv[0])
    sb, rb = load(argv[1])
    assert len(sa) == len(sb) and all((x[0] == y[0]).all() and x[1] == y[1] for x, y in zip(sa, sb)), "sequences differ"
    assert len(ra) == len(rb)
    na, nb = -ra["lp_next"].astype(np.float64), -rb["lp_next"].astype(np.float64)
    d = na - nb  # > 0: A worse on that token
    blk, o = [], 0
    for t, first in sa:
        m = max(0, len(t) - 1 - first)
        blk.append(d[o:o + m].sum() / max(1, m))
        o += m
    blk = np.array(blk)
    se = blk.std(ddof=1) / math.sqrt(len(blk)) if len(blk) > 1 else float("nan")
    top = (ra["id"][:, 0] == rb["id"][:, 0]).mean()
    kl = []
    for i in range(len(ra)):
        pa = np.exp(ra["lp"][i].astype(np.float64))
        mb = dict(zip(rb["id"][i].tolist(), rb["lp"][i].tolist()))
        tail_b = max(rb["tail"][i], 1e-12)
        miss = [j for j in ra["id"][i].tolist() if j not in mb]
        lb = np.array([mb.get(j, math.log(tail_b / max(1, len(miss)) + 1e-30)) for j in ra["id"][i].tolist()])
        s = float((pa * (ra["lp"][i] - lb)).sum())
        ta = ra["tail"][i]
        if ta > 0:
            s += ta * math.log(ta / tail_b)
        kl.append(s)
    print(f"positions {len(ra)} in {len(sa)} sequences")
    print(f"PPL  A {math.exp(na.mean()):.4f} | B {math.exp(nb.mean()):.4f}")
    print(f"mean NLL diff (A - B) {d.mean():+.5f} nats/token, SE by sequence {se:.5f} ({len(blk)} seqs); "
          f"per-seq A better in {(blk < 0).sum()} / {len(blk)}")
    print(f"top-1 agreement A vs B {top:.4%}; KL(A || B) mean {np.mean(kl):.5f} (top-256 + tail, lower bound)")


if __name__ == "__main__":
    main()
