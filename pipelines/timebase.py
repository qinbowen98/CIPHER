"""Follow-up measured from a declared calendar-year origin."""
import pandas as pd

VISIT_YEARS = {'02': 2002, '07': 2007, '12': 2012, '14': 2014, '20': 2020}
MODALITY_VISITS = {'snp': '07', 'ecg': '07', 'fundus': '12'}


def attach_followup(frame, origin_year=2002, modality=None):
    result = frame.copy()
    visits = result['visit'] if modality is None or modality == 'soma' else pd.Series(
        MODALITY_VISITS[modality], index=result.index)
    visits = visits.astype(str).str.replace(r'\.0$', '', regex=True).str.zfill(2)
    years = visits.map(VISIT_YEARS)
    if years.isna().any():
        raise ValueError('Unknown measurement visit in follow-up input.')
    result['visit'] = visits
    result['offset'] = years - int(origin_year)
    result['adjusted_time'] = pd.to_numeric(result['time'], errors='raise') - result['offset']
    return result.loc[result.adjusted_time.ge(0)].copy()
