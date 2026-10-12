from pathlib import Path
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D  # noqa

OUT=Path('/mnt/data/ackss_pediatric_recovery_package')
plt.rcParams.update({'font.size':10,'figure.dpi':160,'axes.titleweight':'bold'})

ref=pd.read_csv(OUT/'Reference_Trajectory.csv')
win=pd.read_csv(OUT/'Delay_Window_Surface.csv')
sc=pd.read_csv(OUT/'Scenario_Results.csv')
pr=pd.read_csv(OUT/'Global_Sensitivity_PRCC.csv')
aic=pd.read_csv(OUT/'Model_Comparison_AIC.csv')
stop=pd.read_csv(OUT/'Optimal_Stopping_Surface.csv')
rob=pd.read_csv(OUT/'Numerical_Robustness.csv')

# Fig 1 architecture — revised spacing and routing
fig,ax=plt.subplots(figsize=(13,7.5));ax.axis('off')
ax.set_xlim(0,1);ax.set_ylim(0,1)
boxes={
 'Lactate clearance\nL(t)':(0.05,0.66),'Capillary refill\nC(t)':(0.05,0.31),'Perfusion reserve\nP(t)':(0.37,0.66),
 'Recovery reserve\nR(t)':(0.37,0.31),'Injury burden\nI(t)':(0.68,0.45),'Delay load\nD(t)':(0.37,0.06),
 'Latent perfusion regimes\nCTMC':(0.68,0.72),'Optimal intervention\nboundary':(0.78,0.12)}
box_w,box_h=0.20,0.13
for label,(x,y) in boxes.items():
    p=FancyBboxPatch((x,y),box_w,box_h,boxstyle='round,pad=0.018',
                     facecolor='#e8f1fb' if x<.65 else '#f3e8ff',
                     edgecolor='#2b4c7e',linewidth=1.5)
    ax.add_patch(p)
    ax.text(x+box_w/2,y+box_h/2,label,ha='center',va='center',fontsize=10)

def center(label):
    x,y=boxes[label]
    return x+box_w/2,y+box_h/2

def edge_arrow(a,b,rad=0.0):
    xa,ya=center(a); xb,yb=center(b)
    dx,dy=xb-xa,yb-ya
    L=(dx*dx+dy*dy)**0.5
    sx,sy=xa+dx/L*(box_w/2), ya+dy/L*(box_h/2)
    ex,ey=xb-dx/L*(box_w/2), yb-dy/L*(box_h/2)
    ax.add_patch(FancyArrowPatch((sx,sy),(ex,ey),arrowstyle='-|>',
                                 mutation_scale=14,linewidth=1.35,color='#4a5568',
                                 connectionstyle=f'arc3,rad={rad}'))

edge_arrow('Lactate clearance\nL(t)','Perfusion reserve\nP(t)')
edge_arrow('Capillary refill\nC(t)','Perfusion reserve\nP(t)')
edge_arrow('Perfusion reserve\nP(t)','Injury burden\nI(t)')
edge_arrow('Recovery reserve\nR(t)','Injury burden\nI(t)')
edge_arrow('Delay load\nD(t)','Recovery reserve\nR(t)')
edge_arrow('Perfusion reserve\nP(t)','Latent perfusion regimes\nCTMC')
edge_arrow('Injury burden\nI(t)','Optimal intervention\nboundary',rad=0.05)
edge_arrow('Latent perfusion regimes\nCTMC','Optimal intervention\nboundary',rad=-0.05)

fig.suptitle('Figure 1. Hybrid Dynamic Recovery Window Architecture',
             fontsize=16,fontweight='bold',y=.98)
ax.set_title('Literature-informed physiology + state transitions + financial-engineering decision layer',
             fontsize=11,pad=8)
fig.subplots_adjust(top=.86,left=.03,right=.97,bottom=.04)
fig.savefig(OUT/'Figure1_Model_Architecture.png',bbox_inches='tight');plt.close(fig)

# Fig2 trajectories multi-panel
fig,axs=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
axs[0,0].plot(ref.time_h,ref.lactate_mmol_L,label='Lactate (mmol/L)',lw=2.4,color='#c0392b');axs[0,0].plot(ref.time_h,ref.crt_seconds,label='CRT (s)',lw=2.4,color='#8e44ad');axs[0,0].axvline(1,color='black',ls='--',alpha=.6);axs[0,0].legend();axs[0,0].set_title('Observable perfusion markers')
axs[0,1].plot(ref.time_h,ref.perfusion_reserve,label='Perfusion reserve',lw=2.4,color='#16a085');axs[0,1].plot(ref.time_h,ref.recovery_reserve,label='Recovery reserve',lw=2.4,color='#2980b9');axs[0,1].plot(ref.time_h,ref.injury_burden,label='Injury burden',lw=2.4,color='#e67e22');axs[0,1].axvline(1,color='black',ls='--',alpha=.6);axs[0,1].legend();axs[0,1].set_ylim(0,1.05);axs[0,1].set_title('Latent physiological states')
axs[1,0].plot(ref.time_h,ref.p_compensated,label='Compensated',lw=2,color='#27ae60');axs[1,0].plot(ref.time_h,ref.p_vulnerable,label='Vulnerable',lw=2,color='#f39c12');axs[1,0].plot(ref.time_h,ref.p_decompensating,label='Decompensating',lw=2,color='#d35400');axs[1,0].plot(ref.time_h,ref.p_refractory,label='Refractory',lw=2,color='#922b21');axs[1,0].legend(ncol=2);axs[1,0].set_ylim(0,1);axs[1,0].set_title('Regime-switching probabilities')
axs[1,1].plot(ref.time_h,ref.recovery_index,label='Recovery index',lw=2.6,color='#2c3e50');axs[1,1].plot(ref.time_h,ref.hazard,label='Forward hazard',lw=2.2,color='#c0392b');axs[1,1].axvline(1,color='black',ls='--',alpha=.6);axs[1,1].legend();axs[1,1].set_title('Decision-facing outputs')
for a in axs.ravel():a.set_xlabel('Time (h)');a.grid(alpha=.2)
fig.suptitle('Figure 2. Reference Theoretical Trajectory With Intervention at 1 Hour',fontsize=15,fontweight='bold')
fig.savefig(OUT/'Figure2_Reference_Trajectory.png',bbox_inches='tight');plt.close(fig)

# Fig3 delay window
fig,ax1=plt.subplots(figsize=(10,6))
colors={'Reversible':'#27ae60','Partially reversible':'#f39c12','Irreversible':'#c0392b'}
for cls,g in win.groupby('class'):
    ax1.scatter(g.delay_h,g.terminal_recovery_index,s=80,label=cls,color=colors[cls],edgecolor='white',linewidth=.8,zorder=3)
ax1.plot(win.delay_h,win.terminal_recovery_index,color='#34495e',lw=2,alpha=.65)
ax1.set_xlabel('Modeled intervention delay (h)');ax1.set_ylabel('Terminal recovery index');ax1.set_title('Figure 3. Dynamic Recovery Window and Reversibility Zones')
ax1.legend();ax1.grid(alpha=.2)
fig.savefig(OUT/'Figure3_Delay_Recovery_Window.png',bbox_inches='tight');plt.close(fig)

# Fig4 stopping-policy surface
# If optimal_delay_h is constant, plotting it as a heatmap is visually
# uninformative. In that case show the policy-value surface and explicitly
# report the constant optimal delay instead of manufacturing variation.
delay_pivot=stop.pivot(index='initial_recovery_reserve',columns='initial_lactate',values='optimal_delay_h')
value_pivot=stop.pivot(index='initial_recovery_reserve',columns='initial_lactate',values='policy_value')
fig,ax=plt.subplots(figsize=(10,7))
extent=[value_pivot.columns.min(),value_pivot.columns.max(),
        value_pivot.index.min(),value_pivot.index.max()]

if np.nanmax(delay_pivot.values)-np.nanmin(delay_pivot.values) < 1e-12:
    im=ax.imshow(value_pivot.values,origin='lower',aspect='auto',cmap='viridis',
                 extent=extent,vmin=np.nanmin(value_pivot.values),vmax=np.nanmax(value_pivot.values))
    cb=fig.colorbar(im,ax=ax)
    cb.set_label('Optimal policy value')
    d=float(np.nanmean(delay_pivot.values))
    ax.text(.02,.98,f'Optimal delay = {d:.1f} h across the evaluated grid',
            transform=ax.transAxes,ha='left',va='top',fontsize=10,
            bbox=dict(boxstyle='round,pad=.35',facecolor='white',alpha=.88,edgecolor='0.5'))
    title='Figure 4. Optimal Stopping Policy Value Across Lactate and Recovery Reserve'
else:
    im=ax.imshow(delay_pivot.values,origin='lower',aspect='auto',cmap='viridis',
                 extent=extent,vmin=np.nanmin(delay_pivot.values),vmax=np.nanmax(delay_pivot.values))
    cb=fig.colorbar(im,ax=ax)
    cb.set_label('Optimal modeled waiting time before intervention (h)')
    title='Figure 4. Optimal Stopping Surface Across Lactate and Recovery Reserve'

ax.set_xlabel('Initial lactate (mmol/L)')
ax.set_ylabel('Initial recovery reserve')
ax.set_title(title)
fig.savefig(OUT/'Figure4_Optimal_Stopping_Surface.png',bbox_inches='tight');plt.close(fig)

# Fig5 sensitivity
p=pr.head(12).sort_values('PRCC')
fig,ax=plt.subplots(figsize=(10,7));barcolors=['#b03a2e' if x<0 else '#1f618d' for x in p.PRCC]
ax.barh(p.parameter,p.PRCC,color=barcolors);ax.axvline(0,color='black',lw=1);ax.set_xlabel('Partial rank correlation coefficient');ax.set_title('Figure 5. Global Sensitivity of Recovery Index at 2-Hour Delay');ax.grid(axis='x',alpha=.2)
fig.savefig(OUT/'Figure5_Global_Sensitivity_PRCC.png',bbox_inches='tight');plt.close(fig)

# Fig6 AIC comparison — revised label placement
p=aic.sort_values('AIC',ascending=True)
fig,ax=plt.subplots(figsize=(11,6.5))
bars=ax.barh(p.model,p.AIC,color=plt.cm.Blues(np.linspace(.45,.9,len(p))))
xmax=float(p.AIC.max())
ax.set_xlim(0,xmax*1.13)
label_x=xmax*1.015
for b,acc in zip(bars,p.synthetic_accuracy):
    ax.text(label_x,b.get_y()+b.get_height()/2,f'acc={acc:.3f}',
            va='center',ha='left',fontsize=9)
ax.set_xlabel('AIC in synthetic model-recovery exercise')
ax.set_title('Figure 6. Internal Model-Recovery Comparison — Not Clinical Validation',pad=10)
ax.grid(axis='x',alpha=.2)
fig.subplots_adjust(left=.25,right=.94,top=.88,bottom=.14)
fig.savefig(OUT/'Figure6_Synthetic_AIC_Comparison.png',bbox_inches='tight');plt.close(fig)

# Fig7 scenario radar-like normalized heatmap
metrics=['reversible_window_h','terminal_recovery_index','terminal_reserve','terminal_injury','peak_hazard']
M=sc.set_index('scenario')[metrics].copy();M['terminal_injury']=1-M['terminal_injury'];M['peak_hazard']=1-M['peak_hazard']
for c in M.columns:
    lo,hi=M[c].min(),M[c].max();M[c]=(M[c]-lo)/(hi-lo+1e-9)
fig,ax=plt.subplots(figsize=(11,6));im=ax.imshow(M.values,aspect='auto',cmap='RdYlGn',vmin=0,vmax=1);fig.colorbar(im,ax=ax,label='Scenario-relative favorable score')
ax.set_xticks(range(len(metrics)),['Recovery window','Recovery index','Terminal reserve','Low injury','Low hazard'],rotation=25,ha='right');ax.set_yticks(range(len(M.index)),M.index);ax.set_title('Figure 7. Scenario Stress-Test Matrix')
fig.savefig(OUT/'Figure7_Scenario_Stress_Test.png',bbox_inches='tight');plt.close(fig)

# Table figures helper
def table_fig(df,title,path,fontsize=8,figsize=(14,5)):
    fig,ax=plt.subplots(figsize=figsize);ax.axis('off');tbl=ax.table(cellText=df.values,colLabels=df.columns,loc='center',cellLoc='center');tbl.auto_set_font_size(False);tbl.set_fontsize(fontsize);tbl.scale(1,1.5)
    for (r,c),cell in tbl.get_celld().items():
        if r==0: cell.set_facecolor('#d9eaf7');cell.set_text_props(weight='bold')
        elif r%2==0: cell.set_facecolor('#f7f9fb')
    ax.set_title(title,fontsize=14,fontweight='bold',pad=18);fig.savefig(path,bbox_inches='tight');plt.close(fig)

states=pd.DataFrame([
 ['L(t)','Lactate','mmol/L','Observable metabolic-perfusion stress'],['C(t)','Capillary refill time','seconds','Observable peripheral perfusion marker'],['P(t)','Perfusion reserve','0–1','Latent capacity to sustain effective tissue perfusion'],['R(t)','Recovery reserve','0–1','Latent capacity to return toward stable physiology'],['I(t)','Injury burden','0–1','Accumulated ischemic/metabolic injury'],['D(t)','Delay load','normalized time','Time-dependent burden before definitive intervention']
],columns=['Symbol','State','Scale','Clinical interpretation'])
table_fig(states,'Table 1. State Variables',OUT/'Table1_State_Variables.png',9,(12,4))

pars=pd.read_csv(OUT/'Model_Parameters.csv').head(20)
table_fig(pars,'Table 2. Provisional Model Parameters and Uncertainty Ranges',OUT/'Table2_Model_Parameters.png',7,(15,8))
table_fig(sc.round(3),'Table 3. Scenario-Based Simulation Results',OUT/'Table3_Scenario_Results.png',7,(15,5))
table_fig(aic.round(3),'Table 4. Synthetic Model-Recovery Comparison',OUT/'Table4_Model_Comparison.png',8,(12,4))

print('figures created')
