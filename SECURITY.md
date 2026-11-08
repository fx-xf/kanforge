# Security policy

kanforge loads model files with `kanforge.load` (NumPy `.npz` without
pickles) and generates source code for kernels and FPGA designs. If you find a
way to make any of these execute untrusted code, or any other vulnerability,
please report it privately through
[GitHub security advisories](https://github.com/fx-xf/kanforge/security/advisories/new)
rather than in a public issue.

Only the latest release receives fixes.
