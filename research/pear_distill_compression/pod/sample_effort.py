"""X2: per-round effort frontier of the distillation teacher. For the same recorded rounds, k samples at each
reasoning effort (low, medium, xhigh; the template supports no other), PEAR's sampling (T=1, top_p=0.95). Logs thinking tokens and parsed action;
analysis compares each effort's actions with the xhigh samples (scripts/analyze_frontier.py).
Rounds: recorded rounds (14 LIBERO episodes) with prompt <= --max-ctx tokens, at most --per-ep per episode.
For each: k samples at PEAR's settings (T=1, top_p=0.95), full thinking + answer. Logs tokens and parsed action."""
import argparse, json, re, sys, threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.insert(0, "/workspace")
from replay_bench import context  # noqa: E402
from run_episodes import Template  # noqa: E402
import urllib.request

JSON_BLOCK = re.compile(r"```json\s*(\{.*?\})\s*```", re.DOTALL)


def post(url, body):
    req = urllib.request.Request(url + "/generate", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30000")
    ap.add_argument("--results", default="/workspace/results/dflash2_main")
    ap.add_argument("--out", default="/workspace/results/frontier.jsonl")
    ap.add_argument("--efforts", default="low,medium,xhigh")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--per-ep", type=int, default=4)
    ap.add_argument("--max-ctx-chars", type=int, default=60000)
    ap.add_argument("--threads", type=int, default=6)
    a = ap.parse_args()
    T = {e: Template("/workspace/models/pear", e) for e in a.efforts.split(",")}
    res = Path(a.results); lock = threading.Lock(); jobs = []
    for ep in sorted(p.stem for p in res.glob("libero_*.jsonl")):
        rows = [r for r in map(json.loads, (res / f"{ep}.jsonl").open()) if r.get("generated")]
        cand = [r for r in range(len(rows)) if rows[r].get("prompt_chars", 0) <= a.max_ctx_chars and rows[r].get("action")]
        step = max(1, len(cand) // a.per_ep)
        jobs += [(ep, rows, r, e) for r in cand[::step][:a.per_ep] for e in T]
    fout = open(a.out, "w")

    def run(ep, rows, r, eff):
        text, images = context(rows, r, res / "frames" / ep, T[eff])
        sp = {"temperature": 1.0, "top_p": 0.95, "max_new_tokens": 32000, "skip_special_tokens": False, "n": a.k}
        o = post(a.url, {"text": text, "image_data": images, "sampling_params": sp})
        outs = o if isinstance(o, list) else [o]
        S = []
        for x in outs:
            g = x["text"]; ans = g.rpartition("</think>")[2]; b = JSON_BLOCK.findall(ans)
            try:
                act = json.loads(b[-1]) if b else None
            except Exception:
                act = None
            S.append({"tok": x["meta_info"]["completion_tokens"], "think_closed": "</think>" in g, "action": act})
        with lock:
            fout.write(json.dumps({"ep": ep, "round": r, "effort": eff, "recorded": rows[r]["action"], "rec_tok": rows[r]["completion_tokens"],
                                   "samples": S}) + "\n"); fout.flush()
        print(ep, r, eff, [s["tok"] for s in S], flush=True)

    with ThreadPoolExecutor(a.threads) as ex:
        for f in [ex.submit(run, *j) for j in jobs]:
            try:
                f.result()
            except Exception as e:
                print("ERR", repr(e), flush=True)
    print("SAMPLE_DONE", len(jobs))


if __name__ == "__main__":
    main()
