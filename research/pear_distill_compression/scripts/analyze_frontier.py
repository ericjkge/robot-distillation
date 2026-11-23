"""X2 analysis: per-effort tokens and action error vs the xhigh samples of the same round (episode bootstrap).
usage: python3 analyze_frontier.py results/frontier.jsonl"""
import json, random, sys
from collections import defaultdict

AX = ("x", "y", "z")


def vec(a):
    try:
        return [float(a["arm"][k]) for k in AX]
    except Exception:
        return None


def grip(a):
    try:
        w = float(a["gripper"]["width"]); return 0 if abs(w) < 1e-9 else (1 if w > 0 else -1)
    except Exception:
        return None


def dist(u, v):
    return sum((a - b) ** 2 for a, b in zip(u, v)) ** 0.5


def medoid(vs):
    return min(vs, key=lambda v: sum(dist(v, w) for w in vs)) if vs else None


def main(path):
    R = [json.loads(l) for l in open(path)]
    by = defaultdict(dict)
    for r in R:
        by[(r["ep"], r["round"])][r["effort"]] = r["samples"]
    rounds = [k for k, v in by.items() if "xhigh" in v and sum(vec(s["action"]) is not None for s in v["xhigh"]) >= 3]
    efforts = [e for e in ("low", "medium", "high", "xhigh") if any(e in by[k] for k in rounds)]
    print(f"{len(rounds)} rounds with >=3 parsed xhigh samples")
    per = {e: defaultdict(list) for e in efforts}
    rerr = {e: defaultdict(list) for e in efforts}  # effort -> round -> errors  # effort -> ep -> list of (tok, err, grip_agree, parsed, closed)
    for k in rounds:
        X = [(vec(s["action"]), grip(s["action"])) for s in by[k]["xhigh"]]
        X = [x for x in X if x[0] is not None]
        for e in efforts:
            for i, s in enumerate(by[k].get(e, [])):
                v, g = vec(s["action"]), grip(s["action"])
                ref = X
                if e == "xhigh":  # leave-one-out: compare with the other xhigh samples only
                    ref = [(vec(t["action"]), grip(t["action"])) for j, t in enumerate(by[k]["xhigh"]) if j != i and vec(t["action"])]
                m = medoid([x[0] for x in ref])
                gm = max(set(x[1] for x in ref), key=[x[1] for x in ref].count)
                rerr[e][k].append(dist(v, m) if v else None)
                per[e][k[0]].append((s["tok"], dist(v, m) if v else None, (g == gm) if v else None, v is not None, s.get("think_closed", True)))
    rnd = random.Random(0)

    def boot(d, f):
        eps = sorted(d); vals = []
        for _ in range(2000):
            rows = [x for e in (rnd.choice(eps) for _ in eps) for x in d[e]]
            vals.append(f(rows))
        vals.sort(); allr = [x for e in eps for x in d[e]]
        return f(allr), vals[50], vals[1949]

    tok = lambda rows: sum(x[0] for x in rows) / len(rows)
    err = lambda rows: (lambda v: sum(v) / len(v))([x[1] for x in rows if x[1] is not None])
    med = lambda rows: sorted(x[1] for x in rows if x[1] is not None)[len([x for x in rows if x[1] is not None]) // 2]
    gag = lambda rows: (lambda v: sum(v) / len(v))([x[2] for x in rows if x[2] is not None])
    xt = tok([x for e in per["xhigh"] for x in per["xhigh"][e]])
    print(f"{'effort':7s} {'tokens':>22s} {'x/xhigh':>7s} {'action err (mean)':>28s} {'median':>7s} {'gripper agree':>14s} parsed  closed")
    for e in efforts:
        d = per[e]; allr = [x for q in d for x in d[q]]
        t, e_ = boot(d, tok), boot(d, err)
        print(f"{e:7s} {t[0]:8.0f} [{t[1]:5.0f},{t[2]:5.0f}] {t[0]/xt:7.2f} {e_[0]:8.2f} [{e_[1]:6.2f},{e_[2]:6.2f}] {med(allr):7.2f} "
              f"{gag(allr):14.2f} {sum(x[3] for x in allr)/len(allr):6.2f} {sum(x[4] for x in allr)/len(allr):6.2f}")

    print("\npaired per round: mean action error(effort) - mean action error(xhigh, leave-one-out)")
    for e in efforts[:-1]:
        d = defaultdict(list)
        for k in rounds:
            a = [x for x in rerr[e][k] if x is not None]; b = [x for x in rerr["xhigh"][k] if x is not None]
            if a and b:
                d[k[0]].append(sum(a) / len(a) - sum(b) / len(b))
        r_ = boot(d, lambda rows: sum(rows) / len(rows))
        print(f"  {e:7s} {r_[0]:+.2f} [{r_[1]:+.2f}, {r_[2]:+.2f}]")


if __name__ == "__main__":
    main(*sys.argv[1:])
