# CIPHER

Training and external validation for CIPHER.

- `pipelines/`: single-modal and multimodal training, UKB/MUCH validation.
- `release/public_models/`: the frozen ASCVD risk-based Logistic and event-based RF models.
- `tests/`: model and inference checks using synthetic data.

```bash
conda env create -f environment.yml
conda activate cipher
cp config/pipeline.example.json config/pipeline.local.json
```

Set input locations in `config/pipeline.local.json`. Paths are relative to the configuration file.

```bash
python pipelines/run.py single  # Single-modal training
python pipelines/run.py multi   # Multimodal training
python pipelines/run.py infer   # UKB and MUCH validation
```

Use `ukb` or `much` to validate one cohort. Validation writes aggregate metrics, confidence intervals and comparison figures to the configured output directory.

Participant inputs, outcomes and predictions, including UKB/MUCH records, are not distributed; analyses requiring these files need authorized data access.

```bash
python -m unittest discover -s tests -v
```
