import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paperstyle import apply_paper_style, PCOLOR
apply_paper_style()
d=np.load(RESULTS/'fig1b_figS1'/'fig1b_final.npy',allow_pickle=True).item()
fig,axes=plt.subplots(1,2,figsize=(19,7.4),sharey=True)
S=[('AIC','^',2,'#1a1a1a',15),('BIC','s',3,'#1e9614',15),('SLIC','o',4,'#e8121c',17)]
for a,arm,ttl,note in [(axes[0],'A','(a) Exact derivatives','no criterion changes sign'),(axes[1],'B','(b) Approximated derivatives','AIC and BIC change sign;\nSLIC does not')]:
    rows=d[arm]; ns=[r[0] for r in rows]
    for lab,m,i,c,ms in S:
        a.plot(ns,[r[i] for r in rows],marker=m,ms=ms,color=c,lw=4.2 if lab=='SLIC' else 3.0,label=lab,markeredgecolor='black',markeredgewidth=0.8,zorder=6 if lab=='SLIC' else 3)
    a.axhline(0,ls='--',color='0.35',lw=2.2,zorder=1)
    a.set_xscale('log'); a.set_yscale('symlog',linthresh=1e-4); a.set_ylim(-2e-3,30); a.set_yticks([-1e-3,0,1e-3,1e-1,1e1]); a.minorticks_off()
    a.set_xlim(6e1,1.6e7); a.set_xticks([1e2,1e3,1e4,1e5,1e6,1e7]); a.set_xticklabels(['$10^2$','$10^3$','$10^4$','$10^5$','$10^6$','$10^7$'])
    a.set_xlabel(r'Number of samples $n$',fontsize=30); a.set_title(ttl,fontsize=31,pad=14); a.tick_params(labelsize=25)
    a.text(0.05,0.08,note,transform=a.transAxes,fontsize=21)
axes[0].set_ylabel('Score difference per datum',fontsize=30); axes[0].legend(fontsize=24,frameon=False,loc='lower left',bbox_to_anchor=(0.0,0.36))
plt.tight_layout(); plt.subplots_adjust(wspace=0.09)
fig.canvas.draw(); r=fig.canvas.get_renderer()
ov=lambda p,q: not (p.x1<=q.x0 or q.x1<=p.x0 or p.y1<=q.y0 or q.y1<=p.y0)
it=[t.get_window_extent(renderer=r) for a in fig.axes for t in [a.title,a.xaxis.label,a.yaxis.label]+list(a.texts)+a.get_xticklabels()+a.get_yticklabels() if t.get_text().strip()]+[axes[0].get_legend().get_window_extent(renderer=r)]
print('last tick >= last data point:',all(max(a.get_xticks())>=max(r[0] for r in d['B']) for a in axes)); print("overlaps:",sum(1 for i in range(len(it)) for j in range(i+1,len(it)) if ov(it[i],it[j])))
out=str(PLOTS/'SI_derivative_control')
for e in ('png','pdf','svg'): fig.savefig(f'{out}.{e}')
