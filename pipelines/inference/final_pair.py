"""Reproduce frozen UKB/MUCH five-model comparisons using exactly two CIPHER weights."""
import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from pipelines.models.final_pair import FEATURES, load_pair, score, sha256
from pipelines.inference.clinical_scores import SCORE_FEATURES, scores
from pipelines.inference.prepared import safe_cindex, safe_event_auc, read_table

LABELS = ['Framingham', 'CHINA-PAR', 'CKM2S2-BAG', 'Best event-based', 'Best risk-based']


def normalize_ids(frame, column):
    result = frame.copy()
    if column not in result or result[column].isna().any():
        raise ValueError('Missing participant identifiers in protected input')
    result['eid'] = result[column].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    if result.eid.eq('').any() or result.eid.duplicated().any():
        raise ValueError('Empty or duplicate identifiers in protected input')
    return result


def load_ukb(spec, resolve):
    original = normalize_ids(read_table(resolve(spec['model_features'])), spec.get('id_column', 'eid'))
    enriched = normalize_ids(read_table(resolve(spec['score_features'])), spec.get('id_column', 'eid'))
    outcomes = read_table(resolve(spec['outcomes']))
    if 'endpoint' in outcomes:
        outcomes = outcomes.loc[outcomes.endpoint.astype(str).eq('ASCVD')].copy()
    outcomes = normalize_ids(outcomes, spec.get('id_column', 'eid'))
    outcomes = outcomes.rename(columns={spec.get('event_column', 'event'): 'event', spec.get('time_column', 'time'): 'time'})
    original = original[['eid', *FEATURES]]
    enriched = enriched[['eid', *SCORE_FEATURES]].rename(columns={c: 'score__' + c for c in SCORE_FEATURES})
    data = outcomes[['eid', 'event', 'time']].merge(original, on='eid', validate='one_to_one').merge(enriched, on='eid', validate='one_to_one')
    for col in data.columns:
        if col != 'eid':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    required = [*FEATURES, *['score__' + c for c in SCORE_FEATURES]]
    keep = data.event.isin([0, 1]) & data.time.gt(0) & np.isfinite(data[required + ['time']]).all(axis=1)
    return data.loc[keep].reset_index(drop=True)


def load_much(spec, resolve):
    data = normalize_ids(read_table(resolve(spec['table'])), spec.get('id_column', 'PATIENT_SN'))
    data = data.rename(columns={spec.get('event_column', 'ASCVD_EVENT'): 'event', spec.get('time_column', 'FOLLOWUP_YEARS'): 'time'})
    for col in set(FEATURES + SCORE_FEATURES + ['time', 'event']):
        if col not in data:
            raise ValueError('Missing prepared MUCH input column: ' + col)
        data[col] = pd.to_numeric(data[col], errors='coerce')
    keep = data.event.isin([0, 1]) & np.isfinite(data.time) & data.time.gt(0)
    return data.loc[keep].reset_index(drop=True)


def evaluate(time, event, matrix):
    return np.asarray([[safe_cindex(time, event, matrix[:, j]), safe_event_auc(event, matrix[:, j])] for j in range(matrix.shape[1])])


def replicate(seed, time, event, matrix):
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(event), len(event))
    if len(np.unique(event[indices])) != 2:
        return None
    return evaluate(time[indices], event[indices], matrix[indices])


def verify_reference(data, reference, age_band):
    reference = read_table(reference)
    if 'age_band' in reference:
        reference = reference.loc[reference.age_band.astype(str).eq(age_band)].copy()
    if {'label', 'score'}.issubset(reference.columns):
        reference['eid'] = reference.eid.astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
        reference = reference.loc[reference.label.isin(LABELS)]
        if reference.duplicated(['eid', 'label']).any():
            raise ValueError('Duplicate reference participant/model pairs')
        reference = reference.pivot(index=['eid', 'time', 'event'], columns='label', values='score').reset_index()
    reference = normalize_ids(reference, 'eid')
    joined = data.merge(reference[['eid', 'time', 'event', *LABELS]], on='eid', suffixes=('', '_reference'), validate='one_to_one')
    if len(joined) != len(data) or len(joined) != len(reference):
        raise ValueError('Reference and inferred evaluation cohorts differ')
    for col in ['time', 'event', *LABELS]:
        if not np.allclose(joined[col], joined[col + '_reference'], rtol=1e-8, atol=1e-10, equal_nan=True):
            raise ValueError('Historical prediction mismatch: ' + col)


def run(config_path, cohort):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    def resolve(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else (config_path.parent / p).resolve()
    pair, model_manifest = load_pair(resolve(cfg['models']))
    spec = cfg['inference'][cohort.lower()]
    data = load_ukb(spec, resolve) if cohort == 'UKB' else load_much(spec, resolve)
    rows, summaries = [], []
    bands = spec.get('age_bands', [[40, 74], [60, 74]]) if cohort == 'UKB' else [None]
    for band in bands:
        age_band = '-'.join(str(v) for v in band) if band else 'all'
        block = data.loc[data.age_feat.between(*band)].copy() if band else data.copy()
        score_frame = block[['score__' + c for c in SCORE_FEATURES]].rename(columns={('score__' + c): c for c in SCORE_FEATURES}) if cohort == 'UKB' else block
        for name, values in scores(score_frame, clip_age=(cohort == 'MUCH')).items():
            block[name] = values
        block['Best risk-based'] = score(pair['risk'], block)
        block['Best event-based'] = score(pair['event'], block)
        finite = np.isfinite(block[LABELS]).all(axis=1)
        if cohort == 'UKB' and not finite.all():
            raise ValueError('Unexpected nonfinite UKB score in the locked complete-case cohort')
        block = block.loc[finite].reset_index(drop=True)
        if spec.get('bootstrap_sort_ids', cohort == 'UKB'):
            block = block.sort_values('eid', kind='mergesort').reset_index(drop=True)
        if len(block) < 5 or block.event.nunique() != 2:
            raise ValueError('The evaluation cohort requires valid events and controls')
        expected = spec.get('expected_counts', {}).get(age_band, {})
        for key, actual in [('n', len(block)), ('events', int(block.event.sum()))]:
            if key in expected and expected[key] != actual:
                raise ValueError('Locked ' + cohort + ' cohort count mismatch: ' + key)
        if spec.get('reference_predictions'):
            verify_reference(block, resolve(spec['reference_predictions']), age_band)
        time, event = block.time.to_numpy(float), block.event.to_numpy(int)
        matrix = block[LABELS].to_numpy(float)
        point = evaluate(time, event, matrix)
        reps = int(spec.get('bootstrap', 300))
        if reps < 0:
            raise ValueError('Bootstrap count must be nonnegative')
        raw = Parallel(n_jobs=int(spec.get('jobs', 1)))(delayed(replicate)(int(spec.get('seed', 20260923)) + i, time, event, matrix) for i in range(reps))
        valid = [value for value in raw if value is not None]
        for rep, values in [(0, point)] + [(i + 1, value) for i, value in enumerate(raw) if value is not None]:
            for j, label in enumerate(LABELS):
                rows.append({'cohort': cohort, 'endpoint': 'ASCVD', 'age_band': age_band, 'label': label,
                             'replicate': rep, 'n': len(block), 'events': int(event.sum()),
                             'cindex': float(values[j, 0]), 'event_auroc': float(values[j, 1])})
        array = np.asarray(valid)
        for j, label in enumerate(LABELS):
            for k, metric in enumerate(['cindex', 'event_auroc']):
                values = array[:, j, k] if len(valid) else np.asarray([point[j, k]])
                summaries.append({'cohort': cohort, 'age_band': age_band, 'label': label, 'metric': metric,
                                  'estimate': float(point[j, k]), 'ci_low': float(np.quantile(values, .025)),
                                  'ci_high': float(np.quantile(values, .975)), 'bootstrap_mean': float(values.mean()),
                                  'valid_bootstraps': len(valid), 'n': len(block), 'events': int(event.sum())})
        print(cohort + ' ' + age_band + ': n=' + str(len(block)) + ', events=' + str(int(event.sum())))
    output = resolve(cfg['private_output']) / cohort.lower()
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / 'aggregate_metrics.csv', index=False)
    pd.DataFrame(summaries).to_csv(output / 'aggregate_summary.csv', index=False)
    safe_manifest = {'cohort': cohort, 'endpoint': 'ASCVD', 'models': [{'role': row['role'], 'sha256': row['sha256']} for row in model_manifest],
                     'bootstrap': spec.get('bootstrap', 300), 'seed': spec.get('seed', 20260923), 'participant_outputs': False,
                     'reference_verified': bool(spec.get('reference_predictions'))}
    (output / 'run_manifest.json').write_text(json.dumps(safe_manifest, indent=2) + '\n')
    plot(pd.DataFrame(summaries), output, cohort)
    print('Saved aggregate metrics, confidence intervals and figures. No participant output was written.')
    return pd.DataFrame(rows)


def plot(summary, output, cohort):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = ['#929292', '#4F73AD', '#35A5D8', '#70777F', '#B9525C']
    for band in summary.age_band.unique():
        fig, axes = plt.subplots(2, 1, figsize=(6.3, 8.2), sharex=True)
        for ax, metric, ylabel in zip(axes, ['event_auroc', 'cindex'], ['Event AUROC', 'C-index']):
            block = summary.loc[summary.age_band.eq(band) & summary.metric.eq(metric)].set_index('label').reindex(LABELS)
            values = block.estimate.to_numpy(float)
            err = np.maximum(np.vstack([values - block.ci_low, block.ci_high - values]), 0)
            ax.bar(np.arange(5), values, color=colors, edgecolor='black', linewidth=.8, yerr=err, capsize=2)
            ax.axhline(values[-1], color=colors[-1], linestyle='--', linewidth=1.3)
            ax.set_ylabel(ylabel)
            ax.set_ylim(max(.45, float(block.ci_low.min()) - .035), min(1., float(block.ci_high.max()) + .065))
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        axes[0].set_title(cohort + (' age ' + band if band != 'all' else ''))
        axes[1].set_xticks(np.arange(5), ['Framingham', 'China-PAR', 'CKM2S2-BAG', 'Event-based', 'Risk-based'], rotation=30, ha='right')
        fig.tight_layout()
        fig.savefig(output / (cohort + '_' + band + '_comparison.pdf'), bbox_inches='tight', transparent=True)
        plt.close(fig)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, default=Path('config/pipeline.local.json'))
    p.add_argument('--cohort', choices=['UKB', 'MUCH'], required=True)
    args = p.parse_args()
    run(args.config, args.cohort)
