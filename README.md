# SEC 10-K Item 1A benchmark dataset

Standard MulTaBench track, strategy 3. The curated table is `data/BIN_TEXT_SEC_MDA_DISTRESS/data.csv`: 1,000 firm-years drawn at random with sample seed 42. Item 1A keeps the first 10 specific lead sentences. The label keeps the worst 25 percent and the best 55 percent of a next-year deterioration score. Ratios stay on their original scale. Training uses fold 0 only.

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

That calls the official `evaluate_on_dataset` path for LightGBM, CatBoost, TabM, TabPFN 2.0, and TabPFN 2.5, under structured, text-only, joint frozen, and joint TAR, for fold 0 only. It does not start a Weights & Biases run. Scores land in `results/BIN_TEXT_SEC_MDA_DISTRESS/evaluation_report.csv`.

Torch is installed from the CUDA 12.8 wheel index. If a later package replaces it, reinstall with:

```powershell
& "$env:USERPROFILE\.venvs\nlp-multabench\Scripts\python.exe" -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --force-reinstall --no-deps
```
