"""X1: does the teacher's effort contrast (COPD-style advantage) point at reasoning the action does not need?

Teacher: Qwen3.8-27B-PEAR-LIBERO_only (the distillation teacher). Data: recorded PEAR rounds (14 LIBERO episodes).
For every 3rd round r of each episode (round r's thinking must end with </think>):
  1. full_x : score round r's recorded tokens with the system prompt at effort=xhigh (training effort).
  2. full_l : same history, system prompt at effort=low.  gap_t = logp_low - logp_xhigh per thinking token
             (COPD's advantage before clipping; negative = the light-thinking teacher would not write it).
  3. Split round r's thinking into K ~equal-size chunks of whole sentences (K = clamp(chars/300, 4, 16)).
     chunk gap = mean gap_t over its tokens.
  4. Under xhigh, delete one chunk at a time and score the recorded action (answer tokens): single-deletion
     action-necessity map (oracle).
  5. Under xhigh, delete chunk sets covering ~f of the thinking chars (f = 0.3, 0.6), chosen by
       copd  : most negative chunk gap first
       anti  : most positive chunk gap first
       rand0, rand1 : random order (seeded)
       oracle: smallest single-deletion action dNLL first
       tail  : last chunks first (early-exit analogue)
     and score the action tokens.
All prefixes up to round r's assistant header are identical across 3-5, so SGLang reuses the cached prefix."""
import argparse, json, random, re, sys, urllib.request, zlib
from pathlib import Path
sys.path.insert(0, "/workspace")
from replay_bench import context  # noqa: E402
from run_episodes import IM_END, Template  # noqa: E402

SPLIT = re.compile(r"(?<=[.!?\n])\s+")


def post(url, body):
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())["meta_info"]


def score(url, text, images, start=0):
    """Tokens of the last assistant turn: list of (cls 't'/'a', logprob, token text). Also the index of that turn's
    <|im_start|> in the prompt (for prefix reuse) and the prompt length."""
    m = post(url, {"text": text, "image_data": images, "sampling_params": {"max_new_tokens": 1, "temperature": 0},
                   "return_logprob": True, "logprob_start_len": start, "return_text_in_logprobs": True})
    lp = m["input_token_logprobs"]
    cur, state, prev, hdr = None, "other", "", None
    for i, (l, tid, t) in enumerate(lp):
        if prev == "<|im_start|>" and t == "assistant":
            state, cur, hdr = "hdr", [], start + i - 1
        elif t == "<|im_start|>":
            state = "other"
        elif state == "hdr" and t == "<think>":
            state = "think"
        elif state == "think" and t == "</think>":
            state = "answer"
        elif state in ("think", "answer") and l is not None:
            if t == "<|im_end|>":
                state = "other"
            else:
                cur.append((state[0], round(l, 5), t))
        prev = t
    return cur, hdr, m["prompt_tokens"]


def chunks(think):
    """Spans (start, end) of K ~equal-size chunks of whole sentences, covering `think` exactly."""
    cuts = [m.end() for m in SPLIT.finditer(think)]
    bounds = [0] + [c for c in cuts if 0 < c < len(think)] + [len(think)]
    sents = list(zip(bounds[:-1], bounds[1:]))
    K = max(1, min(16, max(4, len(think) // 300), len(sents)))
    out, cur0 = [], 0
    for s0, s1 in sents:
        if s1 >= len(think) * (len(out) + 1) / K and len(out) < K - 1:
            out.append((cur0, s1)); cur0 = s1
    if cur0 < len(think):
        out.append((cur0, len(think)))
    return out


def chunk_gaps(think, thx, thl, spans):
    """Mean gap per chunk; tokens mapped to chunks by their character start in the thinking text."""
    assert len(thx) == len(thl)
    text = "".join(t for _, _, t in thx)
    off = text.find(think[:50]) if think else 0
    off = max(off, 0)
    sums, cnts, c, k = [0.0] * len(spans), [0] * len(spans), -off, 0
    for (_, lx, t), (_, ll, _) in zip(thx, thl):
        while k < len(spans) - 1 and c >= spans[k][1]:
            k += 1
        sums[k] += ll - lx; cnts[k] += 1; c += len(t)
    return [s / n if n else 0.0 for s, n in zip(sums, cnts)], cnts, text[off:] == think


def select(order, sizes, f):
    tot, acc, keep_out = sum(sizes), 0, []
    for i in order:
        if acc >= f * tot:
            break
        keep_out.append(i); acc += sizes[i]
    return sorted(keep_out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--results", default="/workspace/results/dflash2_main")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=3)
    a = ap.parse_args()
    TX, TL = Template("/workspace/models/pear", "xhigh"), Template("/workspace/models/pear", "low")
    res = Path(a.results); out = Path(a.out)
    done = {(d["episode"], d["round"]) for d in map(json.loads, out.open())} if out.exists() else set()
    for ep in sorted(p.stem for p in res.glob("libero_*.jsonl")):
        rows = [r for r in map(json.loads, (res / f"{ep}.jsonl").open()) if r.get("generated")]
        for r in range(0, len(rows), a.every):
            g = rows[r]["generated"]
            if (ep, r) in done or "</think>" not in g or any("</think>" not in x["generated"] for x in rows[:r]):
                continue
            think, ans = g.split("</think>", 1)
            frames = res / "frames" / ep
            tx, images = context(rows, r, frames, TX)
            tl, _ = context(rows, r, frames, TL)
            assert tx != tl, "effort does not change the prompt"
            fx, hdr, n = score(a.url, tx + g + IM_END, images)
            fl, _, _ = score(a.url, tl + g + IM_END, images)
            assert len(fx) == len(fl), (ep, r)
            thx = [x for x in fx if x[0] == "t"]; thl = [x for x in fl if x[0] == "t"]
            act = lambda toks: [x[1] for x in toks if x[0] == "a"]
            chs = chunks(think)
            gaps, cnts, aligned = chunk_gaps(think, thx, thl, chs)
            sizes = [e - b for b, e in chs]

            def ablate(drop):
                th = "".join(think[b:e] for i, (b, e) in enumerate(chs) if i not in drop)
                toks, _, _ = score(a.url, tx + th + "</think>" + ans + IM_END, images, start=hdr)
                return act(toks)

            single = [ablate({i}) for i in range(len(chs))]
            base = ablate(set())  # nothing removed, prefix-cached path: must equal full_x's action logprobs
            nll = lambda v: -sum(v) / max(1, len(v))
            sd = [nll(s) - nll(base) for s in single]
            rec = {"episode": ep, "round": r, "prompt_tokens": n, "n_think": len(thx), "n_act": len(act(fx)),
                   "act_x": act(fx), "act_l": act(fl), "think_gap": [round(l[1] - x[1], 4) for x, l in zip(thx, thl)],
                   "aligned": aligned, "chunk_chars": sizes, "chunk_tokens": cnts, "chunk_gap": gaps, "single_dnll": sd, "base": base,
                   "sets": {}}
            rnd = [random.Random(zlib.crc32(f"{ep}:{r}:{s}".encode())) for s in (0, 1)]
            orders = {"copd": sorted(range(len(chs)), key=lambda i: gaps[i]),
                      "anti": sorted(range(len(chs)), key=lambda i: -gaps[i]),
                      "rand0": rnd[0].sample(range(len(chs)), len(chs)),
                      "rand1": rnd[1].sample(range(len(chs)), len(chs)),
                      "oracle": sorted(range(len(chs)), key=lambda i: sd[i]),
                      "tail": list(range(len(chs) - 1, -1, -1))}
            cache = {}
            for f in (0.3, 0.6):
                for name, order in orders.items():
                    drop = tuple(select(order, sizes, f))
                    if drop not in cache:
                        cache[drop] = ablate(set(drop))
                    rec["sets"][f"{name}@{f}"] = {"drop": list(drop), "frac": sum(sizes[i] for i in drop) / sum(sizes),
                                                  "act": cache[drop]}
            with out.open("a") as fo:
                fo.write(json.dumps(rec) + "\n")
            print(ep, r, n, len(chs), round(nll(act(fx)), 4), round(nll(act(fl)), 4),
                  {k: round(nll(v["act"]) - nll(base), 4) for k, v in rec["sets"].items() if k.endswith("0.3")}, flush=True)
    print("EFFORT_DONE")


if __name__ == "__main__":
    main()
