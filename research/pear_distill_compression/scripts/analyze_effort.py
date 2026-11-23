"""X1 analysis (DIRECTION.md): action dNLL after deleting ~f of a round's thinking, per selector; episode bootstrap.
usage: python3 analyze_effort.py results/effort.jsonl"""
import json, random, sys
from collections import defaultdict


def nll(v):
    return -sum(v) / max(1, len(v))


def spearman(x, y):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v)
        for k, i in enumerate(o):
            r[i] = k
        return r
    rx, ry = rank(x), rank(y); n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def boot(per_ep, stat, B=2000, seed=0):
    eps = sorted(per_ep); rnd = random.Random(seed); out = []
    for _ in range(B):
        rows = [r for e in (rnd.choice(eps) for _ in eps) for r in per_ep[e]]
        out.append(stat(rows))
    out.sort()
    return stat([r for e in eps for r in per_ep[e]]), out[int(0.025 * B)], out[int(0.975 * B) - 1]


def main(path):
    R = [json.loads(l) for l in open(path)]
    by_ep = defaultdict(list)
    for r in R:
        by_ep[r["episode"]].append(r)
    print(f"{len(R)} rounds, {len(by_ep)} episodes; aligned {sum(r['aligned'] for r in R)}/{len(R)}")
    print(f"thinking tokens/round mean {sum(r['n_think'] for r in R)/len(R):.0f}; action tokens {sum(r['n_act'] for r in R)/len(R):.0f}")
    # base sanity: cached path equals full_x
    d = max(abs(nll(r["base"]) - nll(r["act_x"])) for r in R)
    print(f"max |base - full_x| action NLL: {d:.5f}")
    m = lambda rows, f: sum(f(r) for r in rows) / len(rows)
    est = boot(by_ep, lambda rows: m(rows, lambda r: nll(r["act_l"]) - nll(r["act_x"])))
    print(f"\naction NLL low-effort prompt minus xhigh (same reasoning): {est[0]:+.5f} [{est[1]:+.5f}, {est[2]:+.5f}]")
    gap = [g for r in R for g in r["think_gap"]]
    print(f"thinking gap (low - xhigh) mean {sum(gap)/len(gap):+.4f}; share of tokens with gap < 0: {sum(g < 0 for g in gap)/len(gap):.2f}")
    rho = [spearman(r["chunk_gap"], r["single_dnll"]) for r in R if len(r["chunk_gap"]) >= 4]
    print(f"Spearman(chunk gap, single-deletion action dNLL), per round: mean {sum(rho)/len(rho):+.3f} (n={len(rho)}; negative = "
          "chunks the low-effort teacher disfavours are the ones the action needs)")
    print("\nby chunk position in the round (quintiles of chunk index / K):")
    print("  pos  mean gap   mean single-deletion action dNLL   share of chunks in copd@0.3 / rand0@0.3 / oracle@0.3")
    for q in range(5):
        G, D, C, Rr, O = [], [], [], [], []
        for r in R:
            K = len(r["chunk_gap"])
            for i in range(K):
                if min(4, int(5 * i / K)) == q:
                    G.append(r["chunk_gap"][i]); D.append(r["single_dnll"][i])
                    C.append(i in r["sets"]["copd@0.3"]["drop"]); Rr.append(i in r["sets"]["rand0@0.3"]["drop"])
                    O.append(i in r["sets"]["oracle@0.3"]["drop"])
        mm = lambda v: sum(v) / len(v)
        print(f"  q{q}  {mm(G):+.4f}   {mm(D):+.5f}                        {mm(C):.2f} / {mm(Rr):.2f} / {mm(O):.2f}")
    last = [r["single_dnll"][-1] for r in R]; rest = [x for r in R for x in r["single_dnll"][:-1]]
    print(f"  last chunk single dNLL mean {sum(last)/len(last):+.4f} vs other chunks {sum(rest)/len(rest):+.5f}; "
          f"top-1 necessity chunk is the last one in {sum(max(range(len(r['single_dnll'])), key=lambda i: r['single_dnll'][i]) == len(r['single_dnll'])-1 for r in R)}/{len(R)} rounds")
    for f in ("0.3", "0.6"):
        print(f"\n--- delete ~{float(f):.0%} of thinking chars: action dNLL vs undeleted (nats/token), episode bootstrap")
        dn = lambda name: (lambda r: nll(r["sets"][f"{name}@{f}"]["act"]) - nll(r["base"]))
        names = ["oracle", "copd", "rand0", "rand1", "anti", "tail"]
        for nm in names:
            e = boot(by_ep, lambda rows: m(rows, dn(nm)))
            fr = m(R, lambda r: r["sets"][f"{nm}@{f}"]["frac"])
            print(f"  {nm:7s} {e[0]:+.5f} [{e[1]:+.5f}, {e[2]:+.5f}]  frac deleted {fr:.2f}")
        rand = lambda r: (dn("rand0")(r) + dn("rand1")(r)) / 2
        for lbl, fn in (("rand - copd", lambda r: rand(r) - dn("copd")(r)),
                        ("anti - copd", lambda r: dn("anti")(r) - dn("copd")(r)),
                        ("rand - oracle", lambda r: rand(r) - dn("oracle")(r))):
            e = boot(by_ep, lambda rows: m(rows, fn))
            print(f"  {lbl:14s} {e[0]:+.5f} [{e[1]:+.5f}, {e[2]:+.5f}]")
        rc, ro, cp = m(R, rand), m(R, dn("oracle")), m(R, dn("copd"))
        print(f"  copd/rand ratio {cp/rc if rc else float('nan'):.2f}; copd share of oracle headroom (rand-copd)/(rand-oracle) "
              f"{(rc-cp)/(rc-ro) if rc != ro else float('nan'):.2f}")


if __name__ == "__main__":
    main(*sys.argv[1:])
