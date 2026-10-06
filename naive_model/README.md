# Full-sequence one-hot Gaussian FFNN baseline

This is the naive baseline used in the manuscript. It needs no pretrained embeddings and uses the same datasets, five protein-disjoint folds and protein-weighted empirical CV+ implementation as ProbJanusDDG.

Each complete WT/mutant sequence pair is encoded per residue as `[WT, WT - MUT, (i-j)/L, |i-j|/L]`: 42 inputs. A shared FFNN (42 → 64 → 64, ReLU) processes residues. Masked mean pooling, masked max pooling and the mutation-site vector are concatenated (192 inputs), followed by a 192 → 64 → 2 readout. There are 19,394 parameters. Padding is excluded and sequences are never truncated.

Training uses the direct model with Gaussian NLL, equal mutation weights, inverse augmentation, Adam at 1e-4, batch size 6 and CUDA FP32. Inference uses `(m_AB - m_BA)/2` and `softplus((s_AB + s_BA)/2) + 0.001`. For each outer fold, three folds train the inner model and one selects the earliest minimum of protein-weighted validation MSE over 300 epochs. A fresh model is trained on the four non-outer folds for that duration. The outer fold provides calibration residuals. Test predictions use the median of five models. Calibration gives each WT protein equal total weight; this weighted variant has no newly established exact distribution-free guarantee.

From the repository root, using the main project requirements:

```bash
python naive_model/run.py --preflight
python naive_model/verify.py
NAIVE_RUN_DIR="$PWD/naive_model/reference" python naive_model/compare.py
NAIVE_RUN_DIR="$PWD/naive_model/reference" python naive_model/plot_paper_figures.py
```

`reference/` includes the five small final checkpoints, all fold predictions, selection curves, selected epochs and the original provenance manifest. Original epochs are **204, 66, 153, 152, 34**. Verification checks data hashes, checkpoint hashes, all 15 prediction sets, epoch minima, a short training/resume test and shared calibration. It does not repeat the full training.

Fresh training writes separately to `runs/naive/`; archived results are not used as training resume state:

```bash
python naive_model/run.py --workers 2
screen -L -Logfile naive_training.log -dmS naive_cvplus bash naive_model/launch.sh --workers 2
```

Set `NAIVE_RUN_DIR` to an empty directory for another independent run. Set `PYTHON` for `launch.sh` if Python is not on PATH. Relaunching resumes saved optimizer/model/RNG state at checkpoint boundaries; source/data/configuration manifest checks prevent mixing incompatible runs. Historical `reference/manifest.json` records the original source hashes; copied `run.py`, comparison and plotting scripts only have packaging/path changes, while `model.py` is unchanged. No Janus embedding files are needed for this baseline.
