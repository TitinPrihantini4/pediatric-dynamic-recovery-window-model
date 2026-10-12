import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.stats import qmc, rankdata, pearsonr
from sklearn.linear_model import LogisticRegression
from pathlib import Path

OUT = Path('/mnt/data/ackss_pediatric_recovery_package')
SEED = 20260811
rng = np.random.default_rng(SEED)

# Literature-informed clinical anchors (not model coefficients)
CLINICAL_ANCHORS = {
    'lactate_reference_mmol_L': 4.0,
    'lactate_range_mmol_L': (2.0, 6.0),
    'lactate_clearance_6h_reference': 0.20,
    'lactate_clearance_range': (0.00, 0.40),
    'crt_reference_seconds': 3.0,
    'crt_range_seconds': (1.5, 5.0),
    'delay_reference_hours': 1.0,
    'delay_range_hours': (0.0, 4.0),
}

# Provisional normalized coefficients. Literature informs direction/broad scale; clinical calibration is not claimed.
BASE = dict(
    lactate_prod=0.48,
    lactate_clear=0.34,
    lactate_clear_post=0.62,
    crt_impair=0.72,
    crt_recover=0.58,
    perfusion_restore=0.42,
    perfusion_lactate=0.20,
    perfusion_crt=0.16,
    perfusion_delay=0.22,
    reperfusion_gain=0.95,
    reserve_recover=0.28,
    reserve_injury=0.34,
    reserve_delay=0.18,
    injury_gain=0.34,
    injury_clear=0.24,
    injury_clear_post=0.52,
    delay_scale=3.0,
    hazard_lactate=0.55,
    hazard_crt=0.48,
    hazard_reserve=0.72,
    hazard_delay=0.62,
    q01=0.06,
    q12=0.05,
    q23=0.035,
    q10=0.035,
    q21=0.026,
    q32=0.018,
    discount=0.94,
    wait_cost=0.045,
    intervention_cost=0.08,
)

RANGES = {
    'lactate_prod': (0.30,0.68), 'lactate_clear': (0.20,0.48), 'lactate_clear_post': (0.45,0.82),
    'crt_impair': (0.45,0.95), 'crt_recover': (0.38,0.80), 'perfusion_restore': (0.28,0.60),
    'perfusion_lactate': (0.10,0.32), 'perfusion_crt': (0.08,0.26), 'perfusion_delay': (0.10,0.35),
    'reperfusion_gain': (0.65,1.25), 'reserve_recover': (0.16,0.42), 'reserve_injury': (0.20,0.50),
    'reserve_delay': (0.08,0.30), 'injury_gain': (0.20,0.50), 'injury_clear': (0.12,0.36),
    'injury_clear_post': (0.34,0.72), 'hazard_lactate': (0.30,0.80), 'hazard_crt': (0.25,0.72),
    'hazard_reserve': (0.45,1.00), 'hazard_delay': (0.35,0.90)
}

X0 = np.array([4.2, 3.1, 0.58, 0.62, 0.30], float) # lactate, CRT, perfusion reserve, recovery reserve, injury burden


def clamp_state(y):
    z = y.copy()
    z[0] = np.clip(z[0], 0.5, 12.0)
    z[1] = np.clip(z[1], 1.0, 8.0)
    z[2:] = np.clip(z[2:], 0.0, 1.0)
    return z


def rhs(t, y, p, tau):
    L, C, P, R, I = y
    u = 1.0 if t >= tau else 0.0
    D = min(t / p['delay_scale'], 1.5) if t < tau else max((tau / p['delay_scale']) * np.exp(-0.7*(t-tau)), 0.0)
    lnorm = np.clip((L-1.5)/4.5, 0, 1.5)
    cnorm = np.clip((C-1.5)/3.5, 0, 1.5)
    clearance = p['lactate_clear']*(1-u) + p['lactate_clear_post']*u
    dL = p['lactate_prod']*(1-P)*(1+0.35*I) - clearance*(L-1.5)
    dC = p['crt_impair']*((1-P)+0.35*I) - p['crt_recover']*(C-1.5) - 0.30*u*(C-1.5)
    dP = p['perfusion_restore']*R*(1-P) - p['perfusion_lactate']*lnorm*P - p['perfusion_crt']*cnorm*P - p['perfusion_delay']*D*P + p['reperfusion_gain']*u*(1-P)
    dR = p['reserve_recover']*P*(1-R) - p['reserve_injury']*I*R - p['reserve_delay']*D*R + 0.10*u*P*(1-R)
    dI = p['injury_gain']*((1-P)+0.30*lnorm+0.25*cnorm)*(1-I) - (p['injury_clear'] + u*(p['injury_clear_post']-p['injury_clear']))*R*I
    return np.array([dL,dC,dP,dR,dI])


def simulate(tau=1.0, p=None, x0=None, horizon=10.0, dt=0.2, method='RK4'):
    if p is None: p = BASE
    if x0 is None: x0 = X0
    if method == 'RK4':
        ts=np.arange(0,horizon+1e-9,dt); y=x0.copy(); Y=[y.copy()]
        for i in range(1,len(ts)):
            t=ts[i-1]
            k1=rhs(t,y,p,tau); k2=rhs(t+dt/2,y+dt*k1/2,p,tau); k3=rhs(t+dt/2,y+dt*k2/2,p,tau); k4=rhs(t+dt,y+dt*k3,p,tau)
            y=clamp_state(y+dt*(k1+2*k2+2*k3+k4)/6); Y.append(y.copy())
        Y=np.asarray(Y); times=ts
    else:
        t_eval=np.arange(0,horizon+1e-9,dt)
        sol=solve_ivp(lambda t,y: rhs(t,y,p,tau), (0,horizon), x0, t_eval=t_eval, method=method, rtol=1e-8, atol=1e-10)
        Y=np.array([clamp_state(v) for v in sol.y.T]); times=sol.t
    df=pd.DataFrame(Y, columns=['lactate_mmol_L','crt_seconds','perfusion_reserve','recovery_reserve','injury_burden'])
    df.insert(0,'time_h',times)
    df['delay_h']=np.minimum(df.time_h,tau)
    df['intervened']=(df.time_h>=tau).astype(int)
    df['lactate_clearance_pct']=(x0[0]-df.lactate_mmol_L)/x0[0]*100
    df['recovery_index']=(df.perfusion_reserve*df.recovery_reserve*(1-df.injury_burden))**(1/3)
    df['hazard']=forward_hazard(df,p,tau)
    return df


def forward_hazard(df,p,tau):
    lnorm=np.clip((df.lactate_mmol_L.values-1.5)/4.5,0,1.5)
    cnorm=np.clip((df.crt_seconds.values-1.5)/3.5,0,1.5)
    delay=np.minimum(df.time_h.values,tau)/p['delay_scale']
    z=-3.0 + p['hazard_lactate']*lnorm + p['hazard_crt']*cnorm + p['hazard_reserve']*(1-df.recovery_reserve.values) + p['hazard_delay']*delay
    return 1/(1+np.exp(-z))


def classify(df):
    f=df.iloc[-1]
    minP=df.perfusion_reserve.min(); minR=df.recovery_reserve.min(); maxI=df.injury_burden.max()
    if f.perfusion_reserve>=0.75 and f.recovery_reserve>=0.60 and f.injury_burden<=0.25:
        return 'Reversible'
    if f.perfusion_reserve>=0.58 and f.recovery_reserve>=0.45 and f.injury_burden<=0.42:
        return 'Partially reversible'
    return 'Irreversible'


def window_threshold(p=None,x0=None):
    delays=np.linspace(0,6,13)
    rows=[]
    for tau in delays:
        d=simulate(tau,p,x0)
        rows.append({'delay_h':tau,'class':classify(d),'terminal_recovery_index':d.recovery_index.iloc[-1],
                     'terminal_perfusion':d.perfusion_reserve.iloc[-1],'terminal_reserve':d.recovery_reserve.iloc[-1],
                     'terminal_injury':d.injury_burden.iloc[-1],'peak_hazard':d.hazard.max(),
                     'lactate_6h':float(np.interp(6,d.time_h,d.lactate_mmol_L)),
                     'crt_6h':float(np.interp(6,d.time_h,d.crt_seconds))})
    res=pd.DataFrame(rows)
    rev=res.loc[res['class']=='Reversible','delay_h']
    part=res.loc[res['class']!='Irreversible','delay_h']
    return res, (rev.max() if len(rev) else np.nan), (part.max() if len(part) else np.nan)


def latent_generator(state,p,tau,t):
    L,C,P,R,I = state
    lnorm=np.clip((L-1.5)/4.5,0,1.5); cnorm=np.clip((C-1.5)/3.5,0,1.5); d=min(t,tau)/p['delay_scale']
    z=0.34*lnorm+0.22*cnorm+0.26*(1-P)+0.28*(1-R)+0.20*I+0.18*d
    q01=p['q01']*np.exp(1.7*z); q12=p['q12']*np.exp(2.0*(z-0.22)); q23=p['q23']*np.exp(2.3*(z-0.38))
    q10=p['q10']*np.exp(1.5*(R-0.5)); q21=p['q21']*np.exp(1.4*(P-0.5)); q32=p['q32']*np.exp(1.3*(R-0.4))
    Q=np.array([[-q01,q01,0,0],[q10,-q10-q12,q12,0],[0,q21,-q21-q23,q23],[0,0,q32,-q32]])
    return Q,z


def regime_probs(df,p,tau):
    pr=np.array([0.68,0.24,0.07,0.01],float); rows=[]
    prevt=df.time_h.iloc[0]
    for _,r in df.iterrows():
        Q,z=latent_generator(r[['lactate_mmol_L','crt_seconds','perfusion_reserve','recovery_reserve','injury_burden']].values,p,tau,r.time_h)
        h=max(r.time_h-prevt,0)
        if h>0:
            for _ in range(5):
                pr=pr+(h/5)*(pr@Q); pr=np.maximum(pr,0); pr=pr/pr.sum()
        rows.append([r.time_h,*pr,z]); prevt=r.time_h
    return pd.DataFrame(rows,columns=['time_h','p_compensated','p_vulnerable','p_decompensating','p_refractory','latent_score'])


def policy_value(tau,p=None,x0=None):
    if p is None:p=BASE
    if x0 is None:x0=X0
    d=simulate(tau,p,x0)
    f=d.iloc[-1]
    benefit=0.40*f.perfusion_reserve+0.34*f.recovery_reserve+0.18*(1-f.injury_burden)-0.08*np.log1p(f.lactate_mmol_L)
    delay_pen=p['wait_cost']*tau + p['intervention_cost']*(1+0.6*d.injury_burden.iloc[int(min(len(d)-1,tau/0.05))])
    return benefit-delay_pen


def optimal_stopping_surface(p=None):
    if p is None:p=BASE
    Ls=np.linspace(2.0,6.0,15); Rs=np.linspace(0.35,0.85,15); delays=np.arange(0,4.01,0.5)
    rows=[]
    for L0 in Ls:
        for R0 in Rs:
            x=X0.copy();x[0]=L0;x[3]=R0
            vals=[policy_value(t,p,x) for t in delays]
            j=int(np.argmax(vals))
            rows.append({'initial_lactate':L0,'initial_recovery_reserve':R0,'optimal_delay_h':delays[j],'policy_value':vals[j]})
    return pd.DataFrame(rows)

# Reference outputs
ref=simulate(1.0)
reg=regime_probs(ref,BASE,1.0)
ref=ref.merge(reg,on='time_h')
ref.to_csv(OUT/'Reference_Trajectory.csv',index=False)

win,t_rev,t_part=window_threshold()
win.to_csv(OUT/'Delay_Window_Surface.csv',index=False)

# Scenario analysis
scenarios={
    'Reference':({},X0.copy()),
    'Faster lactate clearance':({'lactate_clear_post':BASE['lactate_clear_post']*1.25},X0.copy()),
    'Stronger reperfusion':({'reperfusion_gain':BASE['reperfusion_gain']*1.20},X0.copy()),
    'Slower injury resolution':({'injury_clear_post':BASE['injury_clear_post']*0.70},X0.copy()),
    'Higher initial lactate':({},np.array([6.0,3.2,0.55,0.58,0.34])),
    'Prolonged capillary refill':({},np.array([4.5,4.5,0.50,0.58,0.38])),
    'Low reserve profile':({},np.array([4.8,3.8,0.46,0.42,0.44])),
}
rows=[]
for name,(chg,x0) in scenarios.items():
    p=BASE.copy();p.update(chg)
    surf,tr,tp=window_threshold(p,x0)
    d=simulate(1.0,p,x0)
    f=d.iloc[-1]
    rows.append(dict(scenario=name,reversible_window_h=tr,partial_window_h=tp,terminal_recovery_index=f.recovery_index,
                     terminal_perfusion=f.perfusion_reserve,terminal_reserve=f.recovery_reserve,terminal_injury=f.injury_burden,
                     peak_hazard=d.hazard.max(),lactate_6h=float(np.interp(6,d.time_h,d.lactate_mmol_L)),crt_6h=float(np.interp(6,d.time_h,d.crt_seconds))))
pd.DataFrame(rows).to_csv(OUT/'Scenario_Results.csv',index=False)

# LHS global sensitivity on reversible window
keys=list(RANGES)
sampler=qmc.LatinHypercube(d=len(keys),seed=SEED)
U=sampler.random(60)
X=qmc.scale(U,[RANGES[k][0] for k in keys],[RANGES[k][1] for k in keys])
Y=[]
for vals in X:
    p=BASE.copy();p.update(dict(zip(keys,vals)))
    _,tr,tp=window_threshold(p)
    Y.append(0 if np.isnan(tr) else tr)
Y=np.array(Y)
# rank-residualized PRCC
ranks=np.column_stack([rankdata(X[:,i]) for i in range(X.shape[1])]); yr=rankdata(Y)
prcc=[]
for i,k in enumerate(keys):
    others=np.delete(ranks,i,axis=1)
    A=np.column_stack([np.ones(len(others)),others])
    bx=np.linalg.lstsq(A,ranks[:,i],rcond=None)[0]; by=np.linalg.lstsq(A,yr,rcond=None)[0]
    rx=ranks[:,i]-A@bx; ry=yr-A@by
    r,pv=pearsonr(rx,ry);prcc.append((k,r,pv))
pd.DataFrame(prcc,columns=['parameter','PRCC','p_value']).sort_values('PRCC',key=lambda s:abs(s),ascending=False).to_csv(OUT/'Global_Sensitivity_PRCC.csv',index=False)

# Uncertainty intervals
udf=pd.DataFrame({'reversible_window_h':Y})
for tau in [0.5,1.0,2.0,3.0]:
    arr=[]
    for vals in X[:50]:
        p=BASE.copy();p.update(dict(zip(keys,vals)))
        d=simulate(tau,p)
        arr.append(d.recovery_index.iloc[-1])
    udf[f'recovery_index_delay_{tau:g}h']=arr+([np.nan]*(len(udf)-len(arr)))
metrics=[]
for col in udf.columns:
    vals=udf[col].dropna().values
    metrics.append({'metric':col,'P2.5':np.percentile(vals,2.5),'Median':np.median(vals),'P97.5':np.percentile(vals,97.5)})
pd.DataFrame(metrics).to_csv(OUT/'Uncertainty_Intervals.csv',index=False)

# Local elasticity +/-1%
base_window=t_rev
el=[]
for k in keys:
    vals=[]
    for mult in [0.99,1.01]:
        p=BASE.copy();p[k]*=mult
        _,tr,_=window_threshold(p)
        vals.append(max(tr,1e-6) if not np.isnan(tr) else 1e-6)
    elast=(np.log(vals[1])-np.log(vals[0]))/(np.log(1.01)-np.log(0.99))
    el.append({'parameter':k,'elasticity_reversible_window':elast})
pd.DataFrame(el).sort_values('elasticity_reversible_window',key=lambda s:abs(s),ascending=False).to_csv(OUT/'Local_Elasticity.csv',index=False)

# Stochastic perturbation around reference tau=1.0
st=[]
for i in range(100):
    p=BASE.copy()
    for k in keys: p[k]=np.clip(rng.normal(BASE[k],0.08*BASE[k]),RANGES[k][0],RANGES[k][1])
    x=X0.copy(); x[0]=np.clip(rng.normal(4.2,0.45),2,6);x[1]=np.clip(rng.normal(3.1,0.35),1.5,5);x[3]=np.clip(rng.normal(0.62,0.06),0.35,0.85)
    d=simulate(1.0,p,x)
    st.append({'simulation':i+1,'terminal_recovery_index':d.recovery_index.iloc[-1],'terminal_perfusion':d.perfusion_reserve.iloc[-1],
               'terminal_reserve':d.recovery_reserve.iloc[-1],'terminal_injury':d.injury_burden.iloc[-1],
               'peak_hazard':d.hazard.max(),'class':classify(d)})
pd.DataFrame(st).to_csv(OUT/'Stochastic_Perturbation_Results.csv',index=False)

# Synthetic model recovery/AIC: generate virtual states from full structural rule; compare nested logistic models. Internal only.
N=5000
virt=pd.DataFrame({
    'lactate':rng.uniform(2,7,N),'crt':rng.uniform(1.5,5.5,N),'perfusion':rng.uniform(.3,.9,N),
    'reserve':rng.uniform(.3,.9,N),'injury':rng.uniform(.05,.65,N),'delay':rng.uniform(0,4,N)})
logit=-1.4+0.55*(virt.lactate-4)+0.60*(virt.crt-3)-2.0*(virt.perfusion-.55)-1.7*(virt.reserve-.55)+1.25*(virt.injury-.3)+0.52*(virt.delay-1)
prob=1/(1+np.exp(-logit)); virt['irreversible']=rng.binomial(1,prob)
models={
    'Lactate only':['lactate'],
    'Lactate plus CRT':['lactate','crt'],
    'Perfusion reserve model':['lactate','crt','perfusion','reserve'],
    'Clinical five-variable':['lactate','crt','perfusion','reserve','injury'],
    'Full delay-sensitive state':['lactate','crt','perfusion','reserve','injury','delay']}
mr=[]
for name,cols in models.items():
    m=LogisticRegression(max_iter=2000).fit(virt[cols],virt.irreversible)
    pp=np.clip(m.predict_proba(virt[cols])[:,1],1e-9,1-1e-9)
    ll=np.sum(virt.irreversible*np.log(pp)+(1-virt.irreversible)*np.log(1-pp)); k=len(cols)+1
    aic=2*k-2*ll;acc=np.mean((pp>=.5)==virt.irreversible)
    mr.append({'model':name,'variables':len(cols),'AIC':aic,'synthetic_accuracy':acc})
pd.DataFrame(mr).sort_values('AIC').to_csv(OUT/'Model_Comparison_AIC.csv',index=False)

# Optimal stopping surface
optimal_stopping_surface().to_csv(OUT/'Optimal_Stopping_Surface.csv',index=False)

# Numerical robustness (different integrators/tolerances)
rob=[]
for method in ['RK4','RK45','DOP853','Radau','BDF']:
    d=simulate(1.0,BASE,X0,method=method)
    rob.append({'method':method,'terminal_recovery_index':d.recovery_index.iloc[-1],'terminal_perfusion':d.perfusion_reserve.iloc[-1],
                'terminal_reserve':d.recovery_reserve.iloc[-1],'terminal_injury':d.injury_burden.iloc[-1]})
pd.DataFrame(rob).to_csv(OUT/'Numerical_Robustness.csv',index=False)

# Parameter table
par=[]
for k,v in BASE.items():
    par.append({'parameter':k,'reference_value':v,'uncertainty_low':RANGES[k][0] if k in RANGES else '', 'uncertainty_high':RANGES[k][1] if k in RANGES else '',
                'status':'provisional normalized coefficient' if k not in ['discount'] else 'decision-analytic assumption'})
pd.DataFrame(par).to_csv(OUT/'Model_Parameters.csv',index=False)

print('reversible_window',t_rev,'partial_window',t_part)
print(pd.read_csv(OUT/'Scenario_Results.csv').to_string(index=False))
print(pd.read_csv(OUT/'Global_Sensitivity_PRCC.csv').head(8).to_string(index=False))
