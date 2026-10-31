"""Online serving under Poisson arrivals: the highest request rate that meets a p99 TTFT SLO.

Open-loop load: arrival times are drawn up front (exponential gaps), and every request's
TTFT includes the time it spends queued, so an overloaded engine shows up as growing
TTFT. For each scheduling policy we sweep the request rate and report the largest rate
whose p99 TTFT stays under the SLO. Both policies run the same engine, kernels and KV
cache; only the scheduler differs.
"""

import argparse
import random
import time

import numpy as np

from bench.common import save_result
from bench.workload import random_requests
from engine.config import EngineConfig
from engine.engine import LLMEngine
from engine.sequence import SamplingParams


def serve(engine: LLMEngine, rate: float, duration_s: float, args, seed: int = 0, min_requests: int | None = None) -> dict:
    n = max(int(rate * duration_s), min_requests or args.min_requests)
    reqs = random_requests(n, args.input_len, args.output_len, seed=seed, range_ratio=args.range_ratio)
    rng = random.Random(seed)
    arrivals = np.cumsum([rng.expovariate(rate) for _ in range(n)])
    seqs, i = [], 0
    t0 = time.perf_counter()
    while i < n or engine.has_unfinished():
        now = time.perf_counter() - t0
        while i < n and arrivals[i] <= now:
            prompt, max_tokens = reqs[i]
            seqs.append(engine.add_request(prompt, SamplingParams(max_tokens=max_tokens, ignore_eos=True),
                                           arrival_time=t0 + arrivals[i]))
            i += 1
        if engine.has_unfinished():
            engine.step()
        elif i < n:
            time.sleep(max(0.0, arrivals[i] - (time.perf_counter() - t0)))
    elapsed = time.perf_counter() - t0
    ttft = np.array([s.first_token_time - s.arrival_time for s in seqs]) * 1000
    tpot = np.array([(s.finish_time - s.first_token_time) / (len(s.output_token_ids) - 1)
                     for s in seqs if len(s.output_token_ids) > 1]) * 1000
    return {"rate": rate, "num_requests": n, "elapsed_s": elapsed,
            "ttft_ms": {"mean": ttft.mean(), "p50": np.percentile(ttft, 50), "p99": np.percentile(ttft, 99)},
            "tpot_ms": {"mean": tpot.mean(), "p99": np.percentile(tpot, 99)},
            "output_tok_per_s": sum(len(s.output_token_ids) for s in seqs) / elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scheduling", choices=["continuous", "static"], required=True)
    ap.add_argument("--slo-ms", type=float, default=500.0)
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--range-ratio", type=float, default=0.5)
    ap.add_argument("--duration", type=float, default=60.0, help="seconds of arrivals per rate")
    ap.add_argument("--start-rate", type=float, default=0.5)
    ap.add_argument("--min-requests", type=int, default=200, help="per rate, so p99 is not just the max")
    args = ap.parse_args()

    engine = LLMEngine(EngineConfig(scheduling=args.scheduling))
    serve(engine, rate=4, duration_s=5, args=args, seed=123, min_requests=20)  # warmup

    runs = []

    def measure(rate: float) -> bool:
        r = serve(engine, rate, args.duration, args)
        r["meets_slo"] = bool(r["ttft_ms"]["p99"] <= args.slo_ms)
        runs.append(r)
        print(f"{args.scheduling} rate={rate:.2f} req/s  p99 TTFT={r['ttft_ms']['p99']:.0f} ms  "
              f"mean TPOT={r['tpot_ms']['mean']:.1f} ms  -> {'ok' if r['meets_slo'] else 'violates SLO'}")
        return r["meets_slo"]

    # Double the rate until the SLO breaks, then bisect between the last pass and first failure.
    lo, hi, rate = 0.0, None, args.start_rate
    while hi is None:
        if measure(rate):
            lo, rate = rate, rate * 2
        else:
            hi = rate
    for _ in range(4):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if measure(mid) else (lo, mid)

    print(f"{args.scheduling}: max rate under p99 TTFT <= {args.slo_ms:.0f} ms: {lo:.2f} req/s")
    save_result(f"serving_{args.scheduling}", {"scheduling": args.scheduling, "workload": vars(args),
                                                "max_rate_under_slo": lo, "runs": runs})


if __name__ == "__main__":
    main()
