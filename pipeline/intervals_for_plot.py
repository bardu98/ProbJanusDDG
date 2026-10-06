import os
"""Read or recompute protein-weighted empirical CV+ bounds for plotting."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parent.parent
HERE=Path(os.environ.get("PROBJANUS_RUN_DIR", Path(__file__).resolve().parent)).resolve()
sys.path.insert(0,str(ROOT/'conformal'))
from protein_balanced import weighted_bounds

def bounds(name, level):
    if not 50 <= level <= 95:
        raise ValueError('Expected nominal level between 50 and 95 percent')
    z=dict(np.load(HERE/'predictions'/f'{name}_cvplus.npz'))
    oof=dict(np.load(HERE/'predictions/oof_all.npz'))
    d=pd.read_parquet(ROOT/'data/S2450.parquet').reset_index(drop=True)
    np.testing.assert_array_equal(oof['y'],d.ddG)
    w=(1/(d.wt_seq.nunique()*d.groupby('wt_seq').wt_seq.transform('size'))).to_numpy()
    result=[]
    for method,adaptive in [('standard',False),('adaptive',True)]:
        lo,hi=weighted_bounds(oof,z['mu_folds'],z['sigma_folds'],w,np.array([level/100]),adaptive)
        lo,hi=lo[:,0],hi[:,0]
        prefix=method+'_cvplus_protein_empirical_a0.1'
        if level==90:
            np.testing.assert_array_equal(lo,z[prefix+'_lower'])
            np.testing.assert_array_equal(hi,z[prefix+'_upper'])
        elif level<90:
            assert (lo>=z[prefix+'_lower']-1e-12).all()
            assert (hi<=z[prefix+'_upper']+1e-12).all()
        if level==50 and adaptive:
            saved=HERE/'sign_exclusion'/f'{name}_intervals.npz'
            if saved.exists():
                with np.load(saved) as a:
                    np.testing.assert_array_equal(lo,a['lower'][:,0])
                    np.testing.assert_array_equal(hi,a['upper'][:,0])
        result.extend([lo,hi])
    return result
