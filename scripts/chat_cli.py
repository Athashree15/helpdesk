"""Terminal chat:  python scripts/chat_cli.py --model llmops   (choices: llm, rag, llmops)"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from helpdesk import build_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["llm", "rag", "llmops"], default="llmops")
args = ap.parse_args()
cfg, prompts, llm, kb, tracer, bots = build_all()
ok, msg = llm.health()
print(msg)
bot = bots[args.model]
print(f"\n{bot.name} - type 'exit' to quit\n")
while True:
    q = input("You: ").strip()
    if q.lower() in {"exit", "quit"}:
        break
    r = bot.answer(q, session_id="cli") if args.model == "llmops" else bot.answer(q)
    print(f"\nBot: {r.answer}\n")
    if r.sources:
        print("Sources:", ", ".join(f"[{s['rank']}] {s['source']}" for s in r.sources))
    flags = {k: v for k, v in r.meta.items() if k in ("blocked", "cache_hit", "escalated", "groundedness",
                                                     "retrieval_confidence", "pii_masked") and v}
    print(f"({r.latency_s:.1f}s) {flags}\n")
