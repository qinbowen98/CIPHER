"""Reproduce the supplied CHD/ischemic-stroke PLINK association-based SNP selection."""
import hashlib
import argparse
import json
import subprocess
from pathlib import Path

import pandas as pd


def discovery_phenotypes(labels, metadata, soma_valid, endpoints, excluded_site, discovery_ids=None):
    if not {'ID', *endpoints} <= set(labels):
        raise ValueError('Outcome table requires ID and the configured endpoint columns.')
    if not {'CMCS_ID', 'V2'} <= set(metadata):
        raise ValueError('Discovery metadata requires CMCS_ID and V2.')
    if 'SampleId' not in soma_valid:
        raise ValueError('External Soma table requires SampleId.')
    labels = labels.copy()
    labels['ID'] = labels.ID.astype(str)
    if labels.ID.duplicated().any():
        raise ValueError('Duplicate IDs in outcome table.')
    eligible = set(metadata.loc[~metadata.V2.fillna('').astype(str).str.contains(excluded_site, regex=False), 'CMCS_ID'].astype(str))
    heldout = set(soma_valid.SampleId.astype(str).str.split('-').str[-1])
    eligible -= heldout
    if discovery_ids is not None:
        eligible &= set(discovery_ids)
    return labels.loc[labels.ID.isin(eligible)].copy()


def phenotype_for_plink(labels, fam, endpoint):
    frame = labels[['ID', endpoint]].dropna().copy()
    if not frame[endpoint].isin([0, 1]).all():
        raise ValueError('Endpoint must be binary 0/1 before PLINK 1/2 coding.')
    fam = fam.copy()
    fam['ID'] = fam.IID.str.split('_').str[-1]
    if fam.ID.duplicated().any():
        raise ValueError('PLINK IID suffixes do not identify participants uniquely.')
    joined = fam.merge(frame, on='ID', how='inner', validate='one_to_one')
    if joined.empty or joined[endpoint].nunique() != 2:
        raise ValueError('Discovery set requires both cases and controls matched to the PLINK FAM.')
    joined[endpoint] = joined[endpoint].astype(int) + 1
    return joined[['FID', 'IID', endpoint]]


def select_variants(results, top_k, p_threshold):
    ranked = []
    significant = set()
    for endpoint, result in results.items():
        result = result.loc[result.TEST.eq('ADD')].dropna(subset=['P']).sort_values('P', kind='mergesort')
        ranked.append(set(result.head(top_k).SNP))
        significant |= set(result.loc[result.P.lt(p_threshold), 'SNP'])
    return sorted(set.intersection(*ranked) | significant)


def run(args):
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    labels = pd.read_csv(args.outcomes, dtype={'ID': str})
    metadata = pd.read_csv(args.metadata, dtype={'CMCS_ID': str})
    valid = pd.read_csv(args.soma_valid, usecols=['SampleId'])
    discovery_ids = None
    if args.discovery_ids:
        discovery_ids = pd.read_csv(args.discovery_ids, dtype=str)['ID'].tolist()
    labels = discovery_phenotypes(labels, metadata, valid, args.endpoints, args.excluded_site, discovery_ids)
    fam = pd.read_csv(str(args.bfile) + '.fam', sep=r'\s+', header=None, usecols=[0, 1], names=['FID', 'IID'], dtype=str)
    results = {}
    for endpoint in args.endpoints:
        phenotype = phenotype_for_plink(labels, fam, endpoint)
        pheno_path = output / (endpoint + '_phenotype.tsv')
        phenotype.to_csv(pheno_path, sep='\t', header=False, index=False)
        prefix = output / endpoint
        subprocess.run([args.plink, '--bfile', str(args.bfile), '--pheno', str(pheno_path),
                        '--allow-no-sex', '--logistic', '--out', str(prefix)], check=True)
        results[endpoint] = pd.read_csv(str(prefix) + '.assoc.logistic', sep=r'\s+')
    selected = select_variants(results, args.top_k, args.p_threshold)
    if not selected:
        raise ValueError('No variants passed the configured selection rule.')
    variant_file = output / 'selected_snps.txt'
    variant_file.write_text('\n'.join(selected) + '\n')
    subprocess.run([args.plink, '--bfile', str(args.bfile), '--extract', str(variant_file),
                    '--make-bed', '--out', str(output / 'selected')], check=True)
    (output / 'selection.json').write_text(json.dumps({
        'endpoints': args.endpoints, 'top_k': args.top_k, 'p_threshold': args.p_threshold,
        'rule': 'intersection of endpoint top-k lists, union all endpoint P-threshold lists',
        'phenotype_coding': '0/1 to 1/2', 'n_selected': len(selected),
        'selected_snps_sha256': hashlib.sha256(variant_file.read_bytes()).hexdigest(),
        'n_threshold_per_endpoint': {name: int(frame.loc[frame.TEST.eq('ADD'), 'P'].lt(args.p_threshold).sum()) for name, frame in results.items()},
        'discovery_ids_restricted': bool(args.discovery_ids), 'participant_outputs': True}, indent=2) + '\n')
    print('Selected {} variants. All genotype and phenotype outputs remain protected.'.format(len(selected)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bfile', required=True, type=Path)
    parser.add_argument('--outcomes', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--soma-valid', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--discovery-ids', help='Optional ID-column CSV restricting association discovery.')
    parser.add_argument('--plink', default='plink', help='PLINK 1.9 executable.')
    parser.add_argument('--endpoints', nargs='+', default=['CHD', 'istroke'])
    parser.add_argument('--excluded-site', default='\u9996\u94a2')
    parser.add_argument('--top-k', type=int, default=1000000)
    parser.add_argument('--p-threshold', type=float, default=5e-5)
    args = parser.parse_args()
    if args.top_k < 1 or not 0 < args.p_threshold <= 1:
        parser.error('top-k must be positive and p-threshold must be in (0, 1].')
    if len(args.endpoints) != len(set(args.endpoints)):
        parser.error('Endpoint names must be unique.')
    run(args)


if __name__ == '__main__':
    main()
