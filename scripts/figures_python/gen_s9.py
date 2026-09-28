import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import h5py, glob, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from paperstyle import apply_paper_style, PCOLOR, PMARK
apply_paper_style(); matplotlib.rcParams['savefig.bbox']='standard'
def arr(f,key):
    v=f[key][()]
    return f[v['parent_']][:] if hasattr(v,'dtype') and v.dtype.names and 'parent_' in v.dtype.names else v
def dd(f,r):
    ks=[x.decode() if isinstance(x,bytes) else x for x in f[r['keys_']][:]]; return dict(zip(ks,f[r['values_']]))
def deref(f,ref):
    v=f[ref][()]
    return f[v['parent_']][:] if hasattr(v,'dtype') and v.dtype.names and 'parent_' in (v.dtype.names or ()) else v
ICS=['slic','aic','aicc','bic','hqic','kic','bc']; LBL=dict(zip(ICS,['SLIC','AIC','AICc','BIC','HQIC','KIC','BC']))
SYS=[('lorenz','Lorenz'),('rossler','Rossler'),('lotka-volterra','Lotka-Volterra'),('brusselator','Brusselator'),('van_der_pol','Van der Pol'),('nonlin_pendulum','Nonlinear pendulum')]
NOISE=[0,5,10,20,30,40]; D=f'{RESULTS}/figS9_no_projection/'
data={}
for s,_ in SYS:
    with h5py.File(D+f'{s}_noproj_results.jld','r') as f:
        Xt=np.asarray(arr(f,'Xitrue')); true=int((np.abs(Xt)>1e-12).sum()); res=dd(f,f['results'][()]); rows={}
        for nl in NOISE:
            cond=dd(f,f[res[nl]][()]); c=dd(f,f[cond[list(cond.keys())[0]]][()]); xis=dd(f,f[c['Xis']][()])
            rows[nl]={ic: float(np.mean([(np.abs(A)>1e-12).sum() for A in np.asarray(deref(f,xis[ic]))])) for ic in ICS}
    data[s]=(true,rows)
fig,axes=plt.subplots(2,3,figsize=(22,12.5)); axes=axes.ravel()
for a,(s,lab) in zip(axes,SYS):
    true,rows=data[s]
    for ic in ['aic','aicc','bic','hqic','kic','bc','slic']:
        a.plot(NOISE,[rows[n][ic] for n in NOISE],marker=PMARK[ic],ms=14 if ic=='slic' else 11,color=PCOLOR[ic],lw=3.4 if ic=='slic' else 2.2,
               label=LBL[ic],markeredgecolor='black',markeredgewidth=0.7,zorder=6 if ic=='slic' else 3)
    a.axhline(true,color='k',ls=':',lw=2.2,zorder=1)
    vals=[rows[n][ic] for n in NOISE for ic in ICS]+[true]
    tk=MaxNLocator(nbins=5,integer=True,min_n_ticks=3).tick_values(min(vals),max(vals))
    tk=tk[(tk>=np.floor(min(vals))-1e-9)|(tk==tk.min())]
    lo=max([t for t in tk if t<=min(vals)]); hi=min([t for t in tk if t>=max(vals)])
    ticks=[t for t in tk if lo<=t<=hi]; pad=0.04*(hi-lo)
    a.set_ylim(lo-pad,hi+pad); a.set_yticks(ticks)
    a.set_xlim(-2,42); a.set_xticks([0,20,40]); a.set_title(lab,fontsize=30,pad=10); a.tick_params(labelsize=24)
for a in axes[3:]: a.set_xlabel('Noise level (%)',fontsize=28)
for a in (axes[0],axes[3]): a.set_ylabel('Terms selected',fontsize=28)
h,l=axes[0].get_legend_handles_labels(); order=[l.index(x) for x in ['SLIC','AIC','AICc','BIC','HQIC','KIC','BC']]
plt.tight_layout(rect=(0,0,0.86,1)); fig.legend([h[i] for i in order],[l[i] for i in order],loc='center left',bbox_to_anchor=(0.865,0.5),fontsize=25)
fig.canvas.draw(); r=fig.canvas.get_renderer()
ov=lambda p,q: not (p.x1<=q.x0 or q.x1<=p.x0 or p.y1<=q.y0 or q.y1<=p.y0)
it=[t.get_window_extent(renderer=r) for a in fig.axes for t in [a.title,a.xaxis.label,a.yaxis.label]+a.get_xticklabels()+a.get_yticklabels() if t.get_text().strip()]+[fig.legends[0].get_window_extent(renderer=r)]
print("text overlaps:",sum(1 for i in range(len(it)) for j in range(i+1,len(it)) if ov(it[i],it[j])))
out=str(PLOTS/'SI_no_projection')
for e in ('png','pdf','svg'): fig.savefig(f'{out}.{e}')
