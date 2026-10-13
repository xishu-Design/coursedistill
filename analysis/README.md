# Derived-Data Analyses

This directory contains analyses that can be reproduced without loading a
student or teacher model.

## CLS Shape Sensitivity

Run:

```bash
python analysis/analyze_cls_shape_sensitivity.py \
  --pairs analysis/data/future_pairs.csv \
  --output analysis/outputs/cls_shape_sensitivity.csv
```

The script re-ranks the same saved observations under

```text
CLS_gamma(p) = 4 * min(p, 1 - p) * (1 - p)^gamma
```

and reports pair capacity, residual error, selected difficulty composition,
and next-occurrence correctness gain under a fixed top-30% budget. Discrete
score ties are averaged over fixed random seeds.

`future_pairs.csv` is derived from saved verifier outcomes. It contains no
problem text or generated model responses.
