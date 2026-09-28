"""
run_pipeline.py
==============
Runs Step 8 (Graph Builder) + Step 10 (Baseline Benchmark) in a single
Python process so both share the same filesystem namespace — required when
running under OneDrive which virtualises per-process file writes.

Usage:
    python run_pipeline.py
"""
import sys
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

print("=" * 70)
print("   PIPELINE: Step 8 -> Graph Builder")
print("=" * 70)

from src.graph.builder import FullHeteroGraphBuilder
builder = FullHeteroGraphBuilder()
data = builder.build(max_partitions=1)

# Confirm the graph is in-memory
print(f"\n   Graph loaded into memory:")
print(f"     account nodes : {data['account'].x.shape}")
print(f"     account labels: {data['account'].y.shape}")

# ── Save graph to data/graph/ ───────────────────────────────────────────────
graph_dir = ROOT / "data" / "graph"
graph_dir.mkdir(parents=True, exist_ok=True)
graph_path = graph_dir / "hetero_graph_14node.pt"

import torch
print(f"\n   Saving graph to: {graph_path}")
torch.save(data, str(graph_path))
sz = os.path.getsize(str(graph_path))
print(f"   Saved: {sz:,} bytes ({sz/1024/1024:.1f} MB)")

print("\n" + "=" * 70)
print("   PIPELINE: Step 10 -> Baseline Benchmark")
print("=" * 70)

from src.models.baselines import run_benchmark
run_benchmark(
    graph_path=str(graph_path),
    results_dir=str(graph_dir)
)

print("\n   PIPELINE COMPLETE.")
