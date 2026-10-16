# AIM v0.1

**From learned weights to executable mathematical computation.**

AIM is an architecture-aware research tool for examining the computation implemented by a trained neural network. For its first supported transformer architecture, it derives an executable mathematical program from the primitive operations and the supplied weights. The result exposes distribution-weighted aggregation, learned nonlinear transformations, and their contributions to model decisions.

**Try it with the included checkpoint, or train the same architecture on your own dataset and compare what changes.**

## 1. Try the included model

Download the release, extract it, and run the commands below from the extracted directory.

**Linux x86-64** (requires Python 3 and PyTorch to load `.pt` checkpoints):

```bash
python -m pip install -r requirements.txt
chmod +x bin/aim-linux-x86_64
./bin/aim-linux-x86_64 analyze \
  --weights checkpoints/sample_262p.pt \
  --out results/sample
```

Look at the generated computation:

```bash
cat results/sample/learned_program.aim
cat results/sample/proofs.md
```

Run numerical checks or evaluate a token prefix:

```bash
./bin/aim-linux-x86_64 verify --weights checkpoints/sample_262p.pt --cases 100
./bin/aim-linux-x86_64 run --weights checkpoints/sample_262p.pt --prefix 1,2,3
```

The supplied `sample_262p.pt` was trained briefly on the kit's toy token sequences. **It is a compatibility test checkpoint, not evidence of a learned high-level algorithm.**

## 2. Train on your own data

The trainer accepts JSONL containing one list of integer token IDs per line, e.g.:

```text
[1, 2, 3, 4, 5]
[3, 3, 2, 1, 0]
```

With the default architecture, token IDs range from **0 to 13** and context length is at most **33**. The trainer samples causal next-token windows from these sequences.

```bash
python trainer/train.py \
  --data data/toy_sequences.jsonl \
  --steps 1000 \
  --seed 42 \
  --out runs/trained.pt

./bin/aim-linux-x86_64 analyze \
  --weights runs/trained.pt \
  --out results/trained
```

Replace `data/toy_sequences.jsonl` with a dataset in the same format. For a small character-level dataset, see `trainer/prepare_text.py`. You can also change compatible model settings, but the default configuration has received the most testing.

## 3. Compare learned computations

**Different training seeds, same dataset:**

```bash
python trainer/train.py --data data/toy_sequences.jsonl --seed 11 --steps 1000 --out runs/seed11.pt
python trainer/train.py --data data/toy_sequences.jsonl --seed 12 --steps 1000 --out runs/seed12.pt

./bin/aim-linux-x86_64 analyze --weights runs/seed11.pt --out results/seed11
./bin/aim-linux-x86_64 analyze --weights runs/seed12.pt --out results/seed12
```

Now compare `results/seed11/learned_program.aim` and `results/seed12/learned_program.aim`, and their `analysis.json` reports. Repeat with different datasets to investigate how learned transformations vary. **Textual or weight differences alone do not establish differences in computational meaning**, and AIM does not automatically identify which learned structure is noise or which model is best.

## Output files

Every `analyze` run writes:

| File | What you get |
|---|---|
| `learned_program.aim` | Executable explanatory program with distribution-valued and nonlinear operations |
| `analysis.json` | Architecture- and weight-dependent mathematical findings and numerical tests |
| `proofs.md` | Algebraic derivations, assumptions, and guarantee boundaries |

For example, the learned feedforward computation is represented using the exact GELU identity:

\[
\mathrm{GELU}(a)=\tfrac12 a+\tfrac12 a\,\mathrm{erf}(a/\sqrt2).
\]

AIM uses this to distinguish a normalized-linear contribution from an even nonlinear correction. It does **not** claim this is a task-level algorithm recovered from the model.

## Supported architecture

AIM v0.1 supports a **one-layer, one-head causal decoder transformer**, not arbitrary GPT/Llama/Hugging Face checkpoints. The tested configuration has **262 unique trainable parameters**:

| Property | Default |
|---|---|
| Hidden width / FFN width | 4 / 6 |
| Vocabulary / maximum context | 14 / 33 |
| Position embedding | Learned, rank 2 |
| Query/key/value | Shared rank-3 input factor; keys and values tied |
| Attention output / FFN factor rank | 3 / 3 |
| Normalization / activation | RMSNorm (`eps=1e-8`) / exact GELU |
| Residuals / output | Attention and FFN residuals; tied token/output embeddings |

The expected checkpoint contains `model_config` and `model_state`. See [WEIGHT_SCHEMA.md](WEIGHT_SCHEMA.md) for the tensor names, shapes, and portable JSON format.

## Platforms

| Platform | Binary | Test status |
|---|---|---|
| Linux x86-64 | `bin/aim-linux-x86_64` | Executed and tested |
| Linux ARM64 | `bin/aim-linux-arm64` | Cross-compiled; native test pending |
| macOS Apple Silicon | `bin/aim-darwin-arm64` | Cross-compiled; native test pending |
| macOS Intel | `bin/aim-darwin-amd64` | Cross-compiled; native test pending |
| Windows x86-64 | `bin/aim-windows-amd64.exe` | Cross-compiled; native test pending |

On macOS/Linux, run `chmod +x bin/<your-binary>` if necessary. For `.pt` input, the analyzer uses a local Python/PyTorch loader. The `export` command can generate native JSON weight files for subsequent runs without PyTorch.

## Guarantees and scope

AIM records exact algebraic identities under **ideal real-number semantics** and checks selected relationships using exact rational arithmetic on stored coefficients. It also performs **finite numerical regression tests**; those do not establish bit-for-bit equality with arbitrary floating-point implementations.

AIM v0.1 is an **architecture-specific research prototype**. It does not yet certify a globally minimal program, automatically recover a named algorithm, or prove model correctness or safety. Full details: [PROOF_BOUNDARIES.md](PROOF_BOUNDARIES.md).

---

**Train the same architecture. Change the data or seed. Inspect what changes in the learned computation.**
