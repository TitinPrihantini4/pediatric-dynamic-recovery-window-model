import importlib.util, numpy as np, pandas as pd
from scipy.stats import qmc, rankdata, pearsonr
spec=importlib.util.spec_from_file_location('m','/mnt/data/ackss_pediatric_recovery_package/pediatric_recovery_model.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
keys=list(m.RANGES); N=1000
sampler=qmc.LatinHypercube(d=len(keys),seed=20260812)
U=sampler.random(N); X=qmc.scale(U,[m.RANGES[k][0] for k in keys],[m.RANGES[k][1] for k in keys])
Y=[]
for vals in X:
    p=m.BASE.copy(); p.update(dict(zip(keys,vals))); d=m.simulate(2.0,p)
    Y.append(d.recovery_index.iloc[-1])
Y=np.array(Y); ranks=np.column_stack([rankdata(X[:,i]) for i in range(X.shape[1])]); yr=rankdata(Y)
rows=[]
for i,k in enumerate(keys):
    others=np.delete(ranks,i,axis=1); A=np.column_stack([np.ones(N),others])
    rx=ranks[:,i]-A@np.linalg.lstsq(A,ranks[:,i],rcond=None)[0]
    ry=yr-A@np.linalg.lstsq(A,yr,rcond=None)[0]
    r,pv=pearsonr(rx,ry); rows.append((k,r,pv))
pd.DataFrame(rows,columns=['parameter','PRCC','p_value']).sort_values('PRCC',key=lambda s:abs(s),ascending=False).to_csv('/mnt/data/ackss_pediatric_recovery_package/Global_Sensitivity_PRCC.csv',index=False)
pd.DataFrame({'terminal_recovery_index_delay_2h':Y}).describe(percentiles=[.025,.5,.975]).to_csv('/mnt/data/ackss_pediatric_recovery_package/LHS_Recovery_Distribution.csv')
print(pd.DataFrame(rows,columns=['parameter','PRCC','p_value']).sort_values('PRCC',key=lambda s:abs(s),ascending=False).head(10))
