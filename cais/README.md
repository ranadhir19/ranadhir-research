# Semantic Chain-of-Thought CAIS (cais)

Original implementation of the core computational idea from
**"Semantic Chain-of-Thought CAIS: Vision-Language Document Layout Analysis
without Geometric Ground Truth"** (paper under journal revision, 2026). No
text or code is taken from the paper; this is an independent, runnable
rendering of its central mechanism.

## The paper

Document layout analysis usually needs bounding-box annotations, which are
expensive and often unavailable. The paper proposes doing layout analysis
without any geometric ground truth: instead of matching predicted regions
to reference segments by spatial overlap, match them by *semantic content*.
A vision-language model describes what it sees in each region in words, and
regions are aligned to the document's known segment inventory through those
descriptions -- content-aware matching replacing geometry-aware matching.

## Core contribution implemented

Content-aware region matching, simulated end to end:

1. N = 40 reference segments, each with a natural-language description from
   a financial-document vocabulary (tables, flowcharts, code, paragraphs,
   charts, forms, invoices, diagrams).
2. K = 45 predicted regions with noisy descriptions (word dropout p=0.12,
   word swaps p=0.05, plus 5 spurious regions), mimicking VLM proposals.
3. TF-IDF vectors -> K x N cosine-similarity matrix -> optimal one-to-one
   assignment via the Hungarian algorithm
   (`scipy.optimize.linear_sum_assignment`), versus a greedy per-region
   argmax baseline.

Metric: mean cosine similarity of matched pairs; the experiment ablates
Hungarian matching on/off.

## How to run

```bash
cd ~/workspace/research
PYTHONPATH=src python3 experiments/cais_experiment.py   # full experiment
PYTHONPATH=src python3 examples/02_cais_quickstart.py   # 30-second demo
```

The experiment writes `results/cais/results.md` and `results/cais/charts.png`.

## Results

Measured (seed 0, 40 reference segments, 45 predicted regions, vocab size 96):

| matching | mean matched-pair similarity | pairs | collision rate |
|---|---|---|---|
| greedy argmax (Hungarian OFF) | 0.8605 | 45 | 0.178 |
| Hungarian assignment (ON) | 0.9277 | 40 | 0.000 |

Hungarian gain over greedy baseline: **+0.0673** mean similarity.
Collisions (multiple predictions claiming one segment) drop from 8 to 0.

## Insights

- Greedy matching is already decent on average similarity, but it
  double-books popular segments: ~18% of predicted regions collide, which
  in a real layout pipeline means duplicated or contradictory labels.
- The Hungarian assignment fixes collisions *and* raises mean similarity,
  because resolving a collision reassigns the weaker claimant to its
  second-best segment instead of leaving it stacked on the same one.
- The result is robust to description noise at realistic dropout rates;
  content-aware matching degrades gracefully where geometry-aware matching
  would have no signal at all -- which is exactly the paper's point about
  operating without geometric ground truth.
