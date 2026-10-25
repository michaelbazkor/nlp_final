# SEC 10-K Item 1A benchmark dataset

Standard MulTaBench track, strategy 3. The curated table is `data/BIN_TEXT_SEC_MDA_DISTRESS/data.csv`: 1,000 firm-years drawn at random with sample seed 42. The label keeps the worst 25 percent and the best 55 percent of a next-year deterioration score. Fiscal year is removed. The current ratio and asset turnover use a signed log. The text is up to eight content phrases, chosen on the firm-years outside this draw, plus the shortest Item 1A sentence that contains the first phrase, cut at 20 words.

The write-up is `paper/main.pdf`. The official five-fold scores are in `results/gap_search/five_fold_loop/single_column_host_eval/five_fold_summary.csv`. On that table every learner has both gaps positive. The mean joint gap is +0.020 and the mean fine-tuning gap is +0.012.

Raw SEC and EDGAR downloads are cached under `%USERPROFILE%\.cache\nlp-sec`, outside this OneDrive folder. The virtualenv is `%USERPROFILE%\.venvs\nlp-multabench`.

## Rebuild the table

```powershell
& "$env:USERPROFILE\.venvs\nlp-multabench\Scripts\python.exe" curation\build_sec_mda.py
```

## Train later

Training is not part of the dataset build. Before running it:

1. Create a Hugging Face account and accept the licenses for Prior-Labs TabPFN 2.0 and TabPFN 2.5.
2. Create a read token and put it in `MulTaBench/.env` as `HF_TOKEN=...`. The dataset build and the public `intfloat/e5-small-v2` download do not need this token. TabPFN training does.
3. Keep the laptop plugged in. E5-small LoRA uses batch size 8 so it fits a 4GB GPU. TabPFN 2.5 may fall back to CPU.

Then run:

```powershell
.\scripts\train_curation.ps1
```

That calls the official `evaluate_on_dataset` path for LightGBM, CatBoost, TabM, TabPFN 2.0, and TabPFN 2.5, under structured, text-only, joint frozen, and joint TAR. The reported result uses all five folds. `scripts/five_fold_eval.py` is the runner for that table. It does not start a Weights & Biases run. The five-fold scores are in `results/gap_search/five_fold_loop/single_column_host_eval/`.

Torch is installed from the CUDA 12.8 wheel index. If a later package replaces it, reinstall with:

```powershell
& "$env:USERPROFILE\.venvs\nlp-multabench\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```
