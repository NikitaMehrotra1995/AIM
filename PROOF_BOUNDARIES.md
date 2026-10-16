# AIM v0.1.0 — Guarantee statement

AIM analyzes the *declared fixed architecture* using the stored floating weights interpreted as exact real coefficients. It produces a mathematically justified alternate representation of the declared model's **ideal-real** evaluation.

### Sound algebraic transformations within this model family

1. **Distribution semantics of attention.** A finite set of real scores has positive exponential weights and a normalized sum; expectation notation is an exact algebraic restatement of the softmax-weighted value sum.
2. **Log-partition differential.** For tied keys and values, and keys held fixed, the attention expectation equals `sqrt(d_model) * gradient_q(logsumexp(dot(q,k_j)/sqrt(d_model)))`.
3. **GELU decomposition.** Exact-erf `GELU(x) = x/2 + x erf(x/sqrt(2))/2`. For a bias-free FFN, composition with linear projections splits into an odd part and an even nonlinear correction. This is an identity for *all real input vectors*, not a sampled linear approximation.
4. **Common positive scale removal.** Dividing every readout score by the same strictly positive scalar preserves the greedy winner and tie-breaking. The denominator is retained for output categorical probabilities.
5. **Exact weight algebra.** Weight matrix reassociation is independently checked with `math/big.Rat` arbitrary-precision rationals over the finite binary-float coefficients. The report includes a weight-content SHA-256 hash.

### What does not follow

- This is **not** a machine-checked theorem about the implementation of arbitrary PyTorch networks. The derivation rules are implemented/inspected but not exported to Coq, Lean, or an SMT-certified proof checker.
- `verify` and any comparisons to PyTorch are **numeric tests**, not proof of floating-point or universal behavioral equivalence.
- The current synthesis result is an **explanatory mathematical program** retaining nonlinear operators and weight-specific coefficients. It is **not** a discovered task algorithm, nor proof of global minimality or irreducibility.
- Arbitrary values in a portable JSON file are interpreted with the stated architecture as an assumption; the binary cannot establish that the supplied numbers were trained with that architecture.
- The softmax over attention positions is a deterministic probability-vector computation, not necessarily sampling. The optional categorical output distribution is distinct from greedy decoding.

This statement is part of the distributable release. Each generated `proofs.md` file records these premises again with the analyzed weight hash.
