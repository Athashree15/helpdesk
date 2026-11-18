"""Build (or rebuild with --force) the ChromaDB + BM25 index from data/kb/*.md."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from helpdesk import KnowledgeBase, OllamaClient, load_config  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--force", action="store_true")
args = ap.parse_args()
cfg = load_config()
llm = OllamaClient(cfg)
ok, msg = llm.health()
print(msg)
if not ok:
    sys.exit(1)
kb = KnowledgeBase(cfg, llm).build(force=args.force)
for c in kb.chunks[:3]:
    print("-", c.source, "|", c.section, "|", len(c.text), "chars")
print(f"{len(kb.chunks)} chunks indexed in collection '{kb.collection_name}'")
