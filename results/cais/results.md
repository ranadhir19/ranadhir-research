# CAIS experiment results

Config: 40 reference segments, 45 predicted regions
(5 spurious), seed=0, TF-IDF vocabulary size 96.

| matching | mean matched-pair similarity | pairs | collision rate |
|---|---|---|---|
| greedy argmax (Hungarian OFF) | 0.8605 | 45 | 0.178 |
| Hungarian assignment (ON) | 0.9277 | 40 | 0.000 |

Hungarian gain over greedy baseline: **+0.0673** mean similarity.
Collisions (multiple predictions claiming one segment) drop from
8 to 0.
