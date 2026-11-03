# SIAMESE — contrastive CAR/LAR verification

Seeds: 20261019..20261023; 4000 train / 1500 test pairs per seed; 16-d linear projection, contrastive margin 1.0, 30 SGD epochs; hard same-magnitude negatives.

| seed | raw-distance baseline | linear classifier | siamese embedding |
|------|-----------------------|-------------------|-------------------|
| 0 | 0.4973 | 0.5260 | 0.9807 |
| 1 | 0.5053 | 0.5013 | 0.9787 |
| 2 | 0.5153 | 0.5140 | 0.9787 |
| 3 | 0.4953 | 0.4987 | 0.9787 |
| 4 | 0.5000 | 0.4973 | 0.9833 |
| **mean +/- std** | **0.5027 +/- 0.0080** | **0.5075 +/- 0.0123** | **0.9800 +/- 0.0021** |

Reading: the raw-distance baseline (0.50) and the purely linear classifier (0.52) sit at chance, reproducing the paper's 'linear similarity is insufficient' intuition.  The siamese embedding reaches 0.98 on clean synthetic pairs: the 'linear projection + norm' objective is secretly a rank-16 quadratic model (||Wd||^2 = d'WW'd), which discovers compositional digit<->word bigram correspondences.  The paper's ~54% for the embedding presumably reflects real-world cheque noise (OCR errors, handwriting, abbreviations) absent from this noiseless synthetic benchmark -- a deviation from the brief's expectation, reported here as measured.
