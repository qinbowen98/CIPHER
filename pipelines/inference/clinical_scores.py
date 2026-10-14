"""Historical five-model comparison score formulas, without external fitting."""
import numpy as np
import pandas as pd

SCORE_FEATURES = ['age_feat', 'sex', 'SBP', 'TC', 'HDLC', 'waist', 'smoking', 'diabetes', 'treated_hypertension', 'family_history', 'FBG', 'eGFR']

def log(values):
    return np.log(values.where(values.gt(0)))

def scores(frame, clip_age=False):
    age = frame.age_feat.clip(30, 74) if clip_age else frame.age_feat
    sbp, waist, sex, smoking = frame.SBP, frame.waist, frame.sex, frame.smoking
    diabetes, treated, family = frame.diabetes, frame.treated_hypertension, frame.family_history
    tc, hdl = frame.TC * 38.67, frame.HDLC * 38.67
    la, ls, ltc, lh, lw = map(log, [age, sbp, tc, hdl, waist])
    male = (31.97*la + np.where(treated.eq(1),27.39,26.15)*ls + .62*ltc - .69*lh - .71*lw + 3.96*smoking + .36*diabetes + .48 - .16 + 6.22*family + np.where(treated.eq(1),-6.02,-5.73)*la*ls - .94*la*smoking - 1.53*la*family)
    female = (24.87*la + np.where(treated.eq(1),20.71,19.98)*ls + .06*ltc - .22*lh + 1.48*lw + .49*smoking + .57*diabetes + .54 + np.where(treated.eq(1),-4.53,-4.36)*la*ls)
    lp = np.where(sex.eq(1), male, female)
    china = np.clip(1 - np.power(np.where(sex.eq(1), .97, .99), np.exp(np.clip(lp - np.where(sex.eq(1),140.68,117.26), -30, 30))), 0, 1)
    fm = 3.06117*la + 1.12370*ltc - .93263*lh + np.where(treated.eq(1),1.99881,1.93303)*ls + .65451*smoking + .57367*diabetes
    ff = 2.32888*la + 1.20904*ltc - .70833*lh + np.where(treated.eq(1),2.82263,2.76157)*ls + .52873*smoking + .69154*diabetes
    flp = np.where(sex.eq(1), fm, ff)
    framingham = np.clip(1 - np.power(np.where(sex.eq(1),.88936,.95012), np.exp(np.clip(flp-np.where(sex.eq(1),23.9802,26.1931),-30,30))), 0, 1)
    nonhdl, glucose = (frame.TC-frame.HDLC)*38.67, frame.FBG*18.0182
    ckm = 2*sex.eq(1).astype(float) + 2*smoking.eq(1).astype(float) + np.maximum(np.floor((age-40)/10),0)
    ckm += np.where(sbp>=120,np.floor((sbp-110)/10),0) + np.where(glucose>=100,np.floor((glucose-90)/10),0)
    ckm += np.where(nonhdl>=160,np.floor((nonhdl-150)/10),0) + np.where(frame.eGFR<90,np.floor((90-frame.eGFR)/10)+1,0)
    return {'Framingham': np.asarray(framingham,float), 'CHINA-PAR': np.asarray(china,float), 'CKM2S2-BAG': np.asarray(ckm,float)}
