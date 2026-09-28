import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import h5py, numpy as np, glob, re
from collections import Counter
from scipy.integrate import solve_ivp
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from paperstyle import *
apply_paper_style()
def dd(f,r):
    ks=[x.decode() if isinstance(x,bytes) else x for x in f[r['keys_']][:]]; return dict(zip(ks,f[r['values_']]))
def deref(f,ref):
    v=f[ref][()]
    return f[v['parent_']][:] if hasattr(v,'dtype') and v.dtype.names and 'parent_' in (v.dtype.names or ()) else v
files=sorted(glob.glob(f'{RESULTS}/figS11_duffing/**/*.jld',recursive=True),key=lambda p:float(re.search(r'alpha([\d.]+)_',p).group(1)))
al=[];sh=[];keep={i:[] for i in PORDER};nsel={i:[] for i in PORDER};modal={}
for fp in files:
    with h5py.File(fp,'r') as f:
        a=float(f['alpha'][()]); al.append(a); sh.append(100*float(f['cubic_ratio'][()]))
        Xt=np.asarray(f['Xitrue'][()]); res=dd(f,f['results'][()]); xis=dd(f,f[dd(f,f[res[5]][()])['Xis']][()])
        for ic in PORDER:
            A=np.asarray(deref(f,xis[ic])); k=[];n=[]
            for r in range(A.shape[0]):
                M=A[r]; M=M if M.shape==Xt.shape else M.T
                k.append(float(abs(M[1,5])>1e-12)); n.append((np.abs(M)>1e-12).sum())
            keep[ic].append(np.mean(k)); nsel[ic].append(np.mean(n))
            if ic=='slic':
                sup=[tuple(np.abs(A[r][1])>1e-12) for r in range(A.shape[0])]; c=Counter(sup).most_common(1)[0][0]
                modal[a]=np.mean([A[r][1] for r in range(A.shape[0]) if tuple(np.abs(A[r][1])>1e-12)==c],axis=0)
al=np.array(al); sh=np.array(sh); pos=sh>0
def traj(cx,cv,cc,T=60.):
    ts=np.linspace(0,T,3000)
    return ts,solve_ivp(lambda t,u:[u[1],cx*u[0]+cv*u[1]+cc*u[0]**3],(0,T),[1.5,0.],t_eval=ts,rtol=1e-10,atol=1e-12).y
fig=plt.figure(figsize=(24,15.5))
gs=GridSpec(2,2,figure=fig,hspace=0.62,wspace=0.52,left=0.085,right=0.775,top=0.93,bottom=0.09)
errs={}
for k,(a0,lab) in enumerate([(0.02,'(a) Cubic pruned'),(0.20,'(b) Cubic recovered')]):
    ax=fig.add_subplot(gs[0,k]); cr=sh[np.argmin(np.abs(al-a0))]; m=modal[a0]
    ts,yt=traj(-1.,-.1,-a0); _,yp=traj(m[0],m[1],m[5])
    ax.plot(ts,yt[0],color='#111111',lw=4.0,label='True'); ax.plot(ts,yp[0],color=PCOLOR['slic'],lw=3.0,ls='--',label='SLIC model')
    ax.set_xlim(-1,61); ax.set_xticks([0,20,40,60]); ax.set_ylim(-1.6,1.6); ax.set_yticks([-1.5,-1,-0.5,0,0.5,1,1.5]); ax.set_xlabel('Time',fontsize=30); ax.set_ylabel('$x(t)$',fontsize=30)
    ax.set_title(lab+'\n'+rf'$\alpha={a0}$, cubic share {cr:.2f}%',fontsize=28,pad=14)
    ax.tick_params(labelsize=25); ax.legend(fontsize=25,frameon=False,loc='upper right')
    e=np.abs(yt[0]-yp[0]); errs[a0]=100*e.max()/(yt[0].max()-yt[0].min())
    ax.text(0.97,0.06,f'max error {errs[a0]:.1f}%',transform=ax.transAxes,fontsize=23,ha='right')
for k,(data,yl,ttl) in enumerate([(keep,'Fraction retaining $x^3$','(c) Recovery of the cubic term'),
                                  (nsel,'Terms selected','(d) Total terms selected')]):
    ax=fig.add_subplot(gs[1,k])
    for ic in PORDER:
        ax.plot(sh[pos],np.array(data[ic])[pos],marker=PMARK[ic],ms=PMS[ic]-2,color=PCOLOR[ic],lw=PLW[ic]-0.5,
                label=PLABEL[ic],zorder=6 if ic=='slic' else 2,markeredgecolor='black',markeredgewidth=0.6)
    ax.set_xscale('log'); ax.set_xticks([0.01,0.1,1,10,50]); ax.set_xticklabels(['0.01','0.1','1','10','50']); ax.minorticks_off()
    ax.set_xlabel('Cubic share of restoring force (%)',fontsize=29,labelpad=18); ax.set_ylabel(yl,fontsize=30)
    ax.set_title(ttl,fontsize=30,pad=12); ax.tick_params(labelsize=25,pad=6)
    if k==0: ax.set_ylim(-0.05,1.10)
    else:
        ax.axhline(4,color='k',ls=':',lw=2.4); ax.set_ylim(2.8,6.2); ax.set_yticks([3,4,5,6])
        ax.text(0.03,0.93,'dotted line: true model (4 terms)',transform=ax.transAxes,fontsize=22,va='top')
h,l=fig.axes[2].get_legend_handles_labels()
fig.legend(h,l,loc='center left',bbox_to_anchor=(0.79,0.28),fontsize=27,handlelength=0.9,handletextpad=0.3,labelspacing=0.25,borderpad=0.4)
# overlap audit on the laid-out figure
fig.canvas.draw(); r=fig.canvas.get_renderer()
ov=lambda a,b: not (a.x1<=b.x0 or b.x1<=a.x0 or a.y1<=b.y0 or b.y1<=a.y0)
items=[]
for ax in fig.axes:
    for t in [ax.title,ax.xaxis.label,ax.yaxis.label]+list(ax.texts)+ax.get_xticklabels()+ax.get_yticklabels():
        if t.get_text().strip(): items.append(t.get_window_extent(renderer=r))
    if ax.get_legend(): items.append(ax.get_legend().get_window_extent(renderer=r))
for lg in fig.legends: items.append(lg.get_window_extent(renderer=r))
bad=sum(1 for i in range(len(items)) for j in range(i+1,len(items)) if ov(items[i],items[j]))
out=str(PLOTS/'SI_duffing')
for e in ('png','pdf','svg'): fig.savefig(f'{out}.{e}')
print(f"text objects {len(items)} | overlaps {bad} | shares at a=0.02,0.2: {sh[np.argmin(np.abs(al-0.02))]:.4f}%, {sh[np.argmin(np.abs(al-0.2))]:.3f}% | max errors {errs}")
