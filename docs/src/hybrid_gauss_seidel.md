# CPU hybrid Gauss–Seidel

`HybridGaussSeidel` is a CPU smoother for real scalar CSR matrices. It targets
large meshes with a modest number of threads. Each subdomain has one logical
owner; ownership is independent of Julia's task scheduling. Local rows retain
their original order, and cross-subdomain values are lagged by one sweep.

```julia
using Jutul, Jutul.KAPreconditioners
using LinearAlgebra

# A is a sparse matrix, or a CPU StaticSparsityMatrixCSR.
config = HybridGaussSeidel(; npartitions = 10)
state = setup_smoother(A, config)
smooth!(x, state, state.matrix, b)  # forward smoothing of an existing iterate
ldiv!(x, state, b)                # forward/backward preconditioner application

# Use the same smoother on each level of the KA AMG hierarchy.
pc = AMGPreconditioner(; smoother = config)
# The default configuration is also available through a symbol.
pc = AMGPreconditioner(; smoother_type = :hybrid_gauss_seidel)
```

## Literature and choice of method

[Baker, Falgout, Kolev and Yang (2011)](https://doi.org/10.1137/100798806)
analyze hybrid GS: sequential GS within a processor and Jacobi updates between
processors. Its smoothing properties can remain comparable to serial GS when
subdomains are sufficiently large. The same decomposition maps naturally to
CPU threads. Plain hybrid GS can diverge, even for an SPD matrix; their ℓ₁
modification addresses this without estimating an eigenvalue.

The [accompanying technical report, §6 and Remark 6.1](https://digital.library.unt.edu/ark:/67531/metadc870077/m2/1/high_res_d/1122232.pdf)
also establishes convergence of the scaled variant that adds half the absolute
off-subdomain row sum to the diagonal. This is the default here. Let

```math
q_i = \frac12\sum_{j\notin\Omega_{p(i)}} |a_{ij}|,\qquad
\widehat d_i = \operatorname{sign}(a_{ii})(|a_{ii}|+q_i).
```

A forward sweep applies the splitting
`M = diag(d̂)/ω + L_local`, where `L_local` contains strictly lower entries
within each subdomain and `ω` is the damping. Equivalently, update each owned row
with `xᵢ += ω (bᵢ − Σⱼ aᵢⱼ xⱼ)/d̂ᵢ`, using new local predecessor values and
frozen external values. The backward sweep uses `U_local`. For SPD matrices and
`0 < ω ≤ 1`, the scaled method is convergent. A forward/backward application is
symmetric positive definite. These guarantees do not extend to arbitrary
nonsymmetric systems, though these are supported. `l1=false` disables the shift.

[hypre's BoomerAMG documentation](https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html)
uses forward and backward ℓ₁ hybrid GS for its default CPU relaxation. Its
[implementation](https://github.com/hypre-space/hypre/blob/master/src/parcsr_ls/par_relax.c)
distinguishes these from chaotic GS. The method here always freezes the
interface, avoiding concurrent reads of another owner's updates.

[PETSc PCSOR](https://petsc.org/release/manualpages/PC/PCSOR/)
likewise implements parallel SOR through block Jacobi with local SOR. Its
[iteration controls](https://petsc.org/release/manualpages/PC/PCSORSetIterations/)
distinguish global exchanges from repeated local sweeps. Here the interface is
refreshed every sweep; additional local sweeps with a stale interface are not
the default because they weaken coupling between subdomains.

[Thomas et al. (2021)](https://arxiv.org/html/2104.01196v2)
compare sequential, multicolor and two-stage GS. Two-stage GS replaces local
triangular solves with several Jacobi–Richardson iterations, exposing more
fine-grained parallelism at the cost of extra sparse passes. It is attractive
for GPUs; for a few CPU threads and large owned subdomains, a direct local
sweep preserves the recurrence with less memory traffic. Coloring remains
useful when exact dependencies or substantial fine-grained parallelism matter,
but requires synchronization between colors and can change convergence through
reordering. Our choice is based on the requested CPU workload and the benchmark
below; no single smoother is fastest for all matrices.

## Execution and partitioning

Each ordinary sweep has two Polyester batch phases:

1. For interface rows only, compute the external matrix contribution from the
   unchanged iterate. Completion of this phase is the global barrier.
2. Sweep all owned subdomains concurrently, reading the frozen contributions
   and current owned entries. Each output row has exactly one writer.

Interior rows traverse the original CSR arrays. Interface rows have precomputed
lists of local and remote entry positions, so the sweep has no ownership test
inside its nonzero loop. Scratch for the lagged contribution scales with the
number of interface rows. The ownership, diagonal and row-to-interface maps
use `O(n)` storage; extra nonzero lists cover interface rows only. Contiguous
ownership uses a range instead of an allocated traversal map. Single-partition
sweeps use the serial CSR kernel directly. Coefficient updates retain the
schedule and buffers, read cached diagonal positions, and recompute the shifts
using only cross-partition entries. They run independently by partition.

The zero-initial and supplied-residual paths directly solve the local triangular
correction with one batch phase. They avoid a full iterate copy, clearing
scratch, and an extra residual product. AMG uses forward smoothing on descent
and backward smoothing on ascent. Standalone application performs `steps`
forward sweeps followed by `steps` backward sweeps.

Contiguous partitions are the default: they preserve CSR locality and have
cheap setup. The number is capped by `npartitions` and by
`max(1, n ÷ min_partition_size)`, with a default minimum of 8192 rows. Thus small
coarse levels use one partition. `partitioner=:metis` uses the symmetrized
sparsity graph to reduce interfaces on irregularly numbered meshes. Matrix
coefficient changes do not repartition the graph.

For known mesh ownership, pass consecutive positive labels:

```julia
state = setup_smoother(A, config; partition = owners)
```

This override bypasses the automatic partition count and minimum size. Each
subdomain is still traversed in original row order. The vector applies to this
matrix only; AMG constructs ownership separately for its changing level sizes.
When the sparsity changes, call `setup_smoother(A, config; reuse=old_state)`;
`update_smoother!` requires an unchanged pattern. GPU storage, complex scalars
and dense-block coefficients are rejected explicitly.

## Reproducing performance measurements

Run `benchmark/hybrid_gauss_seidel.jl` with `--threads=10` in an environment
containing this checkout. It reports warmed median sweep time, setup time,
allocations and residual reduction after ten sweeps on 3D Poisson grids. Use
command-line arguments to change cube edge lengths. The smoother test file
checks Float32/Float64, both CSR index widths, noncontiguous ownership, both
directions, damping, multiple sweeps, residual reuse, the zero-initial path,
symmetry, an SPD case where unscaled hybrid GS diverges, AMG convergence and
same-pattern coefficient updates. Run it with both one and ten Julia threads.

### Norne pressure replay

The following measurements replay a saved Norne z8 pressure matrix and RHS
after three one-day report steps: 355,444 cells, 2,927,820 nonzeros,
Float64/Int32. All policies rebuild the same HMIS hierarchy with coarse size
five. They measure pressure AMG/GMRES only, not a complete reservoir simulation.
Julia 1.13.2 and Krylov 0.10.11 run under Linux/WSL on a Ryzen 9 7950X, with
BLAS limited to one thread. Ten-thread runs use physical CPUs 0,2,…,18;
one-thread runs use CPU 0. Each policy warms separately. Results are medians
over three rotated policy rounds, with seven AMG applications and five GMRES
solves per round; all timed samples have zero compilation time.

| Smoother | Julia threads | AMG application, ms | GMRES rtol=0.005, ms | GMRES rtol=1e-8, ms |
|:--|--:|--:|--:|--:|
| Serial GS | 1 | 22.62 | 152.00 | 372.08 |
| Serial GS | 10 | 16.70 | 114.50 | 279.75 |
| Hybrid, contiguous | 10 | 9.13 | 74.30 | 189.56 |
| Hybrid, METIS | 10 | 11.24 | 82.25 | 210.18 |
| Hybrid, minimum 32k rows | 10 | 10.51 | 80.78 | 201.79 |

Every policy takes five iterations at rtol=0.005 and thirteen at rtol=1e-8,
with true relative residuals below the requested tolerances. The serial-GS
ten-thread row keeps other KA operations parallel, isolating the smoother
comparison. Contiguous hybrid GS reduces pressure-solve time by 35% at the
looser tolerance and 32% at the stricter tolerance relative to that control.
Relative to the fully serial run, the pressure solves are about twice as fast.

Review improvements reduced the initial contiguous implementation's median
AMG application from 10.73 to 9.13 ms and numeric hierarchy update from 22.27
to 15.30 ms in separate matched runs. Single-partition levels bypass the
partition map, contiguous row maps are ranges, and numeric smoother updates
read cached diagonals and interface entries. METIS reduces the interface but
does not win in this replay; contiguous ownership remains the default. These
measurements support this workload choice without establishing a universal
optimum across meshes or complete reservoir schedules.

`benchmark/pressure_smoothers.jl INPUT.bin OUTPUT.toml` reproduces the replay
from a serialized named tuple containing `transposed_matrix` (the sparse CSC
transpose representing CSR storage) and `rhs`. It requires no JutulDarcy
dependency once those arrays have been extracted. Run it with both
`--threads=1` and `--threads=10`, controlling core affinity externally.
