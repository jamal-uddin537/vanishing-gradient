| configuration | ‖δ₁‖/‖δ_L‖ at init (depth 30) | ‖∇W₁‖/‖∇W_L‖ at init (depth 30) | selected lr | final train loss (depth 20) | final test acc (depth 20) |
|---|---|---|---|---|---|
| sigmoid + naive init | 3.54e-22 | 7.81e-22 | 0.03 | 2.303 ± 0.000 | 0.101 ± 0.001 |
| sigmoid + Xavier | 2.03e-19 | 2.76e-19 | 0.003 | 2.303 ± 0.000 | 0.101 ± 0.001 |
| tanh + naive init | 7.87e-04 | 1.13e+00 | 0.01 | 0.042 ± 0.031 | 0.939 ± 0.007 |
| tanh + Xavier | 2.60e-01 | 1.50e+00 | 0.01 | 0.003 ± 0.000 | 0.954 ± 0.006 |
| ReLU + naive init | 3.32e-08 | 1.13e+00 | 0.003 | 2.302 ± 0.000 | 0.103 ± 0.001 |
| ReLU + He | 1.01e+00 | 3.76e-01 | 0.003 | 0.002 ± 0.002 | 0.931 ± 0.012 |
| sigmoid + Xavier + BN | 9.90e-01 | 5.68e-01 | 0.01 | 0.027 ± 0.006 | 0.962 ± 0.005 |
| sigmoid + Xavier + residual | 3.46e-01 | 4.61e-03 | 0.03 | 0.043 ± 0.001 | 0.965 ± 0.001 |
