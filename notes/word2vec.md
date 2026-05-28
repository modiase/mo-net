# Word2Vec

## Setup

We train word embeddings on the English-sentences corpus
(`s3://mo-net-resources/english-sentences.txt`, 1,534,699 sentences,
~33M tokens raw, ~7.5M CBOW context windows after stopword filtering
and `context_size=4` windowing).

Two architectures are wired into `mo_net.samples.word2vec`:

- **CBOW** — `Embedding(vocab → embed) → Average(over 2·context) → Linear(embed → vocab) → SparseCategoricalSoftmax`.
  Input: `(N, 2·context)` context-word IDs. Output target: `(N,)` centre word ID.
- **SkipGram** — same layers with input shape `(1,)` (single centre word). Training data is
  expanded so each `(centre, context_i)` becomes its own row; otherwise the
  shapes would mismatch in both the backward pass and val-loss computation.

Both use `sparse_cross_entropy` against the softmax output. Three softmax
strategies are configurable: `full`, `negative-sampling`, `hierarchical`.
A common shared backward-prop helper does negative-sampling when
`softmax_strategy == NEGATIVE_SAMPLING`; everything else falls through to
the standard full-softmax derivative.

Optimiser is **AdaM** (β₁=0.9, β₂=0.999, ε=1e-8) wrapped with
`WarmupScheduler → CosineScheduler`. Learning-rate limits are
`(lr/100, lr)` so cosine decays from the peak rate to 1% of it over
`num_epochs`, with 1 epoch of linear warmup at the start. Per-step
**global L2 gradient clipping at norm=1.0** rescales every layer's
`dP` in lock-step whenever the joint norm exceeds the cap. Regulariser:
`EmbeddingWeightDecayRegulariser` at λ=1e-5.

The cached training pairs are written as separate uncompressed `.npy`
files and loaded with `mmap_mode='r'`, so concurrent sweep tasks share
the OS page cache. Skipgram's 8× expansion is materialised in JAX after
the 80/20 train/val split.

Training runs in pyxis-launched enroot containers on herakles. The
container ships `fluent-bit` and an entrypoint wrapper that tees
stdout/stderr to per-stream files and forwards them to a host-local
Loki when `MO_NET_LOKI_URL` is set. Training metrics (every-batch
`runs`/`iterations` rows) go to a host-local Postgres via the
`--logging-backend-connection-string` flag — writes are pushed through
a `ThreadPoolExecutor` so the training loop never blocks on the DB.

### Bug fixes that gate sensible results

Three pre-sweep-141 fixes are load-bearing for interpreting the
numbers below — earlier sweep results were artefacts of these bugs:

1. **CBOW + NS used to be a no-op.** Both `NEGATIVE_SAMPLING` and
   `FULL` constructed the same model and CBOW had no `backward_prop`
   override, so the base path ran full-softmax derivatives regardless
   of strategy. CBOW now stores the strategy and dispatches in
   `backward_prop`.
2. **SkipGram + FULL silently ran NS.** SkipGram's `backward_prop`
   branched on `isinstance(SparseCategoricalSoftmaxOutputLayer, …)`
   which is true for both strategies, then always called
   `backward_prop_with_negative`. Now dispatches on the stored strategy.
3. **SkipGram val_loss + backward were shape-broken.** The (centre,
   N×context) data fed `Y_pred[(N, V)]` against `Y_true.flatten()` of
   length `N·context`, with JAX silently clipping out-of-bounds indices.
   Fixed by expanding skipgram data so each `(centre, context_i)` is
   its own row.

## Sweep 141

48 jobs (`sbatch --array=0-47%2`) on a single herakles node with two RTX
3090s. 8 epochs per task on the full corpus (~7.5M CBOW pairs),
batch size 4096.

Grid (innermost varies first):

| axis      | values                  |
| --------- | ----------------------- |
| vocab     | 3000, 10000             |
| lr (peak) | 1e-4, 3e-5              |
| embed dim | 64, 128, 256            |
| softmax   | negative-sampling, full |
| model     | cbow, skipgram          |

Cosine decay + grad clipping + skipgram expansion all enabled. LR=1e-3
and 3e-4 were dropped from the grid after sweep 123 confirmed they
overshoot catastrophically even with cosine decay and gradient
clipping at norm=1.0.

### Findings (24/48 done — all cbow complete)

**Full softmax is dominant at this scale.** Best cbow result per row:

| vocab | best cbow run                                 | min val_loss  | overshoots? |
| ----- | --------------------------------------------- | ------------- | ----------- |
| 3000  | `cbow_full_e256_lr1e-4_v3000`                 | **5.547**     | no          |
| 3000  | `cbow_full_e128_lr1e-4_v3000`                 | 5.568         | no          |
| 3000  | `cbow_full_e64_lr1e-4_v3000`                  | 5.575         | no          |
| 3000  | `cbow_ns_e128_lr1e-4_v3000`                   | 6.138 → 9.904 | **yes**     |
| 3000  | `cbow_ns_e256_lr3e-5_v3000` (best stable NS)  | 6.212         | no          |
| 10000 | `cbow_full_e256_lr1e-4_v10000`                | **7.452**     | no          |
| 10000 | `cbow_ns_e256_lr3e-5_v10000` (best stable NS) | 7.806         | no          |

Full softmax beats NS by **~0.5-0.8 cross-entropy units** across the
board and never overshoots. The earlier impression that NS was "good
enough" was the CBOW-NS bug: CBOW was silently running full softmax,
so we were comparing full-softmax to itself.

**Negative sampling instability scales with embed dim.** At
`lr=1e-4` (our "safe" rate) NS still diverges late in training when
`embed >= 128`:

| run                          | min_vl | last_vl |
| ---------------------------- | ------ | ------- |
| `cbow_ns_e128_lr1e-4_v3000`  | 6.138  | 9.904   |
| `cbow_ns_e256_lr1e-4_v3000`  | 6.317  | 14.960  |
| `cbow_ns_e128_lr1e-4_v10000` | 8.096  | 15.437  |
| `cbow_ns_e256_lr1e-4_v10000` | 7.853  | 21.295  |

Global L2 grad clipping at norm=1.0 doesn't catch this. The sampled
NS gradient is a biased estimator whose variance grows with embed dim;
per-step norms stay modest but the cumulative bias drives the
embeddings off the manifold.

**Capacity helps when training is stable.** Within cbow + full + lr=1e-4

- v=3000: embed 64 → 128 → 256 gives 5.575 → 5.568 → 5.547. Diminishing
  but monotone. The model can productively use more capacity once the
  gradient signal is clean.

**LR=3e-5 is too conservative for full softmax.** Within
cbow + full + v=3000: lr=1e-4 beats lr=3e-5 at every embed dim by 0.03-0.1
CE. The model is undertrained at 3e-5 even after 8 epochs.

**Perplexity scales superlinearly with vocab.** Sweep 123 best at v=1000
landed at CE 3.97 (PP 53); sweep 141 best at v=3000 is CE 5.55 (PP 256);
at v=10000 it's CE 7.45 (PP 1727). For a well-trained word2vec, PP at
fixed-quality embeddings shouldn't move that much with vocab — the
model is data-starved, not capacity-starved. Each v=10000 word sees
only ~750 examples (7.5M pairs / 10k words). Quality embeddings
typically want 10k+ examples per word.

### Pending

- Skipgram tasks 24-47 (all skipgram × {full, ns} × embed × lr × vocab).
  Expectation: skipgram + full will be stable but lose to cbow because each
  (centre, context_i) pair is a weaker signal than cbow's averaged-context
  setup at this training-pair count.

### What to try next

In rough order of expected gain per unit effort:

1. **Frequent-word subsampling** (Mikolov `t=1e-5`). Drops common-word
   training pairs probabilistically. Likely the single biggest win for
   our data scale — rare-word gradients aren't drowned out and the
   effective corpus density increases.
2. **Drop the weight-decay regulariser.** Embeddings need non-trivial
   magnitudes; λ=1e-5 isn't punishing but it's not helping either.
3. **More epochs on the leader config.** 8 epochs at lr=1e-4 finished
   with `min == last` — no plateau detected. 20-30 epochs would likely
   keep dropping.
4. **Bigger context window.** `context_size=5` or `6` extracts more
   training pairs per sentence without sourcing more data.
5. **Bigger corpus.** Going from 7.5M pairs to Wikipedia-scale (~500M+)
   puts PP=20 at v=10000 in reach. Cheaper structurally than
   architectural changes but bigger wall-time.

The val-loss/CE metric is also worth questioning. The downstream
quality test is the analogy/nearest-neighbour evaluation
(`python -m mo_net.samples.word2vec eval --model-path …zip`). It's
possible PP=256 embeddings already do something useful on analogies;
we haven't yet measured that on a sweep-141 model.
