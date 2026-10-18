Adam lr 3e-3, 3000 steps, seed 0; metric: test RMSE (lower is better) or accuracy (higher).

### exp(sin(pi x1) + x2^2)

| model | width | params | FLOPs | RMSE |
|---|---|---|---|---|
| MLP | 4 | 17 | 57 | 0.7365 |
| MLP | 8 | 33 | 113 | 0.8418 |
| MLP | 16 | 65 | 225 | 0.1653 |
| MLP | 32 | 129 | 449 | 0.1221 |
| MLP | 64 | 257 | 897 | 0.09461 |
| MLP | 128 | 513 | 1793 | 0.09223 |
| MLP | 256 | 1025 | 3585 | 0.09047 |
| KAN B-spline G5 k3 | 1 | 29 | 824 | 0.04104 |
| KAN B-spline G5 k3 | 2 | 57 | 1135 | 0.01068 |
| KAN B-spline G5 k3 | 4 | 113 | 1757 | 0.003746 |
| KAN B-spline G5 k3 | 8 | 225 | 3001 | 0.00178 |
| KAN B-spline G5 k3 | 16 | 449 | 5489 | 0.001933 |
| KAN B-spline G5 k3 | 32 | 897 | 10465 | 0.001498 |
| KAN Chebyshev d4 | 1 | 20 | 101 | 0.06972 |
| KAN Chebyshev d4 | 2 | 39 | 159 | 0.1051 |
| KAN Chebyshev d4 | 4 | 77 | 275 | 0.02533 |
| KAN Chebyshev d4 | 8 | 153 | 507 | 0.01795 |
| KAN Chebyshev d4 | 16 | 305 | 971 | 0.01237 |
| KAN Chebyshev d4 | 32 | 609 | 1899 | 0.008207 |

Best within a FLOP budget:

| FLOPs <= | KAN B-spline G5 k3 | KAN Chebyshev d4 | MLP |
|---|---|---|---|
| 1e+02 | — | — | 0.7365 |
| 1e+03 | 0.04104 | 0.01237 | 0.09461 |
| 1e+04 | 0.00178 | 0.008207 | 0.09047 |
| 1e+05 | 0.001498 | 0.008207 | 0.09047 |

### x1 * x2

| model | width | params | FLOPs | RMSE |
|---|---|---|---|---|
| MLP | 4 | 17 | 57 | 0.02179 |
| MLP | 8 | 33 | 113 | 0.01013 |
| MLP | 16 | 65 | 225 | 0.007712 |
| MLP | 32 | 129 | 449 | 0.00605 |
| MLP | 64 | 257 | 897 | 0.006227 |
| MLP | 128 | 513 | 1793 | 0.004981 |
| MLP | 256 | 1025 | 3585 | 0.01809 |
| KAN B-spline G5 k3 | 1 | 29 | 824 | 0.1234 |
| KAN B-spline G5 k3 | 2 | 57 | 1135 | 0.00069 |
| KAN B-spline G5 k3 | 4 | 113 | 1757 | 0.001973 |
| KAN B-spline G5 k3 | 8 | 225 | 3001 | 0.0005986 |
| KAN B-spline G5 k3 | 16 | 449 | 5489 | 0.0004267 |
| KAN B-spline G5 k3 | 32 | 897 | 10465 | 0.0008165 |
| KAN Chebyshev d4 | 1 | 20 | 101 | 0.1516 |
| KAN Chebyshev d4 | 2 | 39 | 159 | 0.008885 |
| KAN Chebyshev d4 | 4 | 77 | 275 | 0.004261 |
| KAN Chebyshev d4 | 8 | 153 | 507 | 0.002174 |
| KAN Chebyshev d4 | 16 | 305 | 971 | 0.001584 |
| KAN Chebyshev d4 | 32 | 609 | 1899 | 0.001747 |

Best within a FLOP budget:

| FLOPs <= | KAN B-spline G5 k3 | KAN Chebyshev d4 | MLP |
|---|---|---|---|
| 1e+02 | — | — | 0.02179 |
| 1e+03 | 0.1234 | 0.001584 | 0.00605 |
| 1e+04 | 0.0004267 | 0.001584 | 0.004981 |
| 1e+05 | 0.0004267 | 0.001584 | 0.004981 |

### sin(x1) exp(-x2^2) + x3 / (1 + x4^2)

| model | width | params | FLOPs | RMSE |
|---|---|---|---|---|
| MLP | 4 | 25 | 73 | 0.1081 |
| MLP | 8 | 49 | 145 | 0.06752 |
| MLP | 16 | 97 | 289 | 0.01212 |
| MLP | 32 | 193 | 577 | 0.007397 |
| MLP | 64 | 385 | 1153 | 0.006718 |
| MLP | 128 | 769 | 2305 | 0.00694 |
| MLP | 256 | 1537 | 4609 | 0.005414 |
| KAN B-spline G5 k3 | 1 | 47 | 1372 | 0.1419 |
| KAN B-spline G5 k3 | 2 | 93 | 1719 | 0.0789 |
| KAN B-spline G5 k3 | 4 | 185 | 2413 | 0.001661 |
| KAN B-spline G5 k3 | 8 | 369 | 3801 | 0.0009948 |
| KAN B-spline G5 k3 | 16 | 737 | 6577 | 0.001119 |
| KAN B-spline G5 k3 | 32 | 1473 | 12129 | 0.001408 |
| KAN Chebyshev d4 | 1 | 32 | 167 | 0.1416 |
| KAN Chebyshev d4 | 2 | 63 | 249 | 0.09131 |
| KAN Chebyshev d4 | 4 | 125 | 413 | 0.007006 |
| KAN Chebyshev d4 | 8 | 249 | 741 | 0.007707 |
| KAN Chebyshev d4 | 16 | 497 | 1397 | 0.006881 |
| KAN Chebyshev d4 | 32 | 993 | 2709 | 0.00445 |

Best within a FLOP budget:

| FLOPs <= | KAN B-spline G5 k3 | KAN Chebyshev d4 | MLP |
|---|---|---|---|
| 1e+02 | — | — | 0.1081 |
| 1e+03 | — | 0.007006 | 0.007397 |
| 1e+04 | 0.0009948 | 0.00445 | 0.005414 |
| 1e+05 | 0.0009948 | 0.00445 | 0.005414 |

### Friedman-1 (10 inputs)

| model | width | params | FLOPs | RMSE |
|---|---|---|---|---|
| MLP | 4 | 49 | 121 | 0.1092 |
| MLP | 8 | 97 | 241 | 0.07352 |
| MLP | 16 | 193 | 481 | 0.02965 |
| MLP | 32 | 385 | 961 | 0.03041 |
| MLP | 64 | 769 | 1921 | 0.02607 |
| MLP | 128 | 1537 | 3841 | 0.02244 |
| MLP | 256 | 3073 | 7681 | 0.0209 |
| KAN B-spline G5 k3 | 1 | 101 | 3016 | 0.2539 |
| KAN B-spline G5 k3 | 2 | 201 | 3471 | 0.1581 |
| KAN B-spline G5 k3 | 4 | 401 | 4381 | 0.01306 |
| KAN B-spline G5 k3 | 8 | 801 | 6201 | 0.003787 |
| KAN B-spline G5 k3 | 16 | 1601 | 9841 | 0.006346 |
| KAN B-spline G5 k3 | 32 | 3201 | 17121 | 0.007644 |
| KAN Chebyshev d4 | 1 | 68 | 365 | 0.2523 |
| KAN Chebyshev d4 | 2 | 135 | 519 | 0.2114 |
| KAN Chebyshev d4 | 4 | 269 | 827 | 0.02568 |
| KAN Chebyshev d4 | 8 | 537 | 1443 | 0.02147 |
| KAN Chebyshev d4 | 16 | 1073 | 2675 | 0.01336 |
| KAN Chebyshev d4 | 32 | 2145 | 5139 | 0.02354 |

Best within a FLOP budget:

| FLOPs <= | KAN B-spline G5 k3 | KAN Chebyshev d4 | MLP |
|---|---|---|---|
| 1e+03 | — | 0.02568 | 0.02965 |
| 1e+04 | 0.003787 | 0.01336 | 0.0209 |
| 1e+05 | 0.003787 | 0.01336 | 0.0209 |

### digits (64 inputs, 10 classes)

| model | width | params | FLOPs | accuracy |
|---|---|---|---|---|
| MLP | 4 | 310 | 634 | 0.908 |
| MLP | 8 | 610 | 1258 | 0.962 |
| MLP | 16 | 1210 | 2506 | 0.96 |
| MLP | 32 | 2410 | 5002 | 0.974 |
| MLP | 64 | 4810 | 9994 | 0.974 |
| MLP | 128 | 9610 | 19978 | 0.974 |
| MLP | 256 | 19210 | 39946 | 0.978 |
| KAN B-spline G5 k3 | 1 | 677 | 17983 | 0.432 |
| KAN B-spline G5 k3 | 2 | 1344 | 19572 | 0.758 |
| KAN B-spline G5 k3 | 4 | 2678 | 22750 | 0.898 |
| KAN B-spline G5 k3 | 8 | 5346 | 29106 | 0.968 |
| KAN B-spline G5 k3 | 16 | 10682 | 41818 | 0.972 |
| KAN B-spline G5 k3 | 32 | 21354 | 67242 | 0.974 |
| KAN Chebyshev d4 | 1 | 455 | 2264 | 0.504 |
| KAN Chebyshev d4 | 2 | 900 | 3174 | 0.744 |
| KAN Chebyshev d4 | 4 | 1790 | 4994 | 0.852 |
| KAN Chebyshev d4 | 8 | 3570 | 8634 | 0.962 |
| KAN Chebyshev d4 | 16 | 7130 | 15914 | 0.962 |
| KAN Chebyshev d4 | 32 | 14250 | 30474 | 0.972 |

Best within a FLOP budget:

| FLOPs <= | KAN B-spline G5 k3 | KAN Chebyshev d4 | MLP |
|---|---|---|---|
| 1e+03 | — | — | 0.908 |
| 1e+04 | — | 0.962 | 0.974 |
| 1e+05 | 0.974 | 0.972 | 0.978 |
