# CIPHER

Code for training CIPHER models and plotting the main and supplementary figures.

- `notebooks/`: figure notebooks.
- `pipelines/`: single-modal training, multimodal training and external validation.
- `release/public_models/`: frozen ASCVD risk-based Logistic and event-based RF models.

## Run

Use Python 3.10 and scikit-learn 1.3.2.

```bash
python -m pip install -r requirements-clinical.txt
cp config/pipeline.example.json config/pipeline.local.json
```

Set private input locations in the local configuration, then run:

```bash
python pipelines/run.py infer
```

Training uses `single` or `multi` and `requirements-training.txt`.
Figure inputs are listed in [docs/FIGURE_INPUTS.md](docs/FIGURE_INPUTS.md).

Participant inputs, outcomes, predictions, embeddings and UKB/MUCH records are not distributed.
Only approved aggregate figure tables may be shared. Private cohort analyses require authorized data access.

```bash
python -m unittest discover -s tests -v
python scripts/check_public_release.py
```
