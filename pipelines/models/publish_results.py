"""Add only schema-validated UKB/MUCH aggregate metric tables to the release."""
import argparse
import json
from pathlib import Path
from .export import clean_metrics
from .final_pair import sha256, load_pair


def publish(config_path):
    config_path = Path(config_path).resolve()
    cfg = json.loads(config_path.read_text())
    def resolve(value):
        p = Path(value)
        return p if p.is_absolute() else (config_path.parent/p).resolve()
    release = resolve(cfg['models'])
    load_pair(release)
    manifest_path = release/'release_manifest.json'
    receipt = json.loads(manifest_path.read_text())
    tables = []
    for cohort in ['ukb','much']:
        path = resolve(cfg['private_output'])/cohort/'aggregate_metrics.csv'
        if path.exists():
            frame = clean_metrics(path)
            if set(frame.cohort) != {cohort.upper()} or set(frame.label) != {'Framingham','CHINA-PAR','CKM2S2-BAG','Best event-based','Best risk-based'}:
                raise ValueError('Unexpected aggregate cohort/model labels')
            tables.append((cohort, path, frame))
    if not tables:
        raise ValueError('Run final inference before publishing results')
    for cohort, source, frame in tables:
        name = 'public_results/metrics_' + cohort + '.csv'
        target = release/name
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target,index=False)
        receipt['files'] = [entry for entry in receipt['files'] if entry['file'] != name]
        receipt['files'].append({'file':name,'sha256':sha256(target),'source_sha256':sha256(source),'rows':len(frame),'schema':list(frame.columns)})
    manifest_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print('Published aggregate metrics for: ' + ', '.join(c.upper() for c,_,_ in tables))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--config',type=Path,default=Path('config/pipeline.local.json'))
    args=p.parse_args()
    publish(args.config)
