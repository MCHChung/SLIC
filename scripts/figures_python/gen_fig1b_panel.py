import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import numpy as np, matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from paperstyle import apply_paper_style
apply_paper_style(); matplotlib.rcParams['savefig.bbox']='standard'
d=np.load(RESULTS/'fig1b_figS1'/'fig1b_final.npy',allow_pickle=True).item()
rows=d['B']; ns=np.array([r[0] for r in rows]); k1,k2=d['k1'],d['k2']
S={'AIC':(np.array([r[2] for r in rows]),'#1a1a1a','^',15),'BIC':(np.array([r[3] for r in rows]),'#1e9614','s',15),'SLIC':(np.array([r[4] for r in rows]),'#e8121c','o',17)}
fig=plt.figure(figsize=(9,6)); ax=fig.add_axes([0.13,0.15,0.84,0.82])
for lab,(y,c,m,ms) in S.items():
    ax.plot(ns,y,marker=m,ms=ms,color=c,lw=4.4 if lab=='SLIC' else 3.2,label=lab,markeredgecolor='black',markeredgewidth=0.9,zorder=6 if lab=='SLIC' else 4)
ax.axhline(0,ls='--',color='0.3',lw=2.2,zorder=1); ax.axhline(np.log(k2/k1),ls=':',color='#e8121c',lw=2.6,zorder=2)
ax.text(1.6e2,np.log(k2/k1)+0.12,r'$\log(k_{\rm over}/k_{\rm true})$',color='#e8121c',fontsize=26)
ax.set_xscale('log'); ax.set_xlim(6e1,1.6e7); ax.set_ylim(-0.12,2.55)
ax.set_xticks([1e2,1e3,1e4,1e5,1e6,1e7]); ax.minorticks_off(); ax.tick_params(labelsize=26)
ax.set_xlabel(r'Number of samples $n$',fontsize=30); ax.set_ylabel('Score difference per datum',fontsize=28)
ax.legend(loc='center left',bbox_to_anchor=(0.14,0.5),fontsize=27,handlelength=1.2,handletextpad=0.4,labelspacing=0.3,borderpad=0.4)
ins=fig.add_axes([0.53,0.28,0.41,0.36]); sel=ns>=1e5
for lab in ['AIC','BIC']:
    y,c,m,ms=S[lab]; ins.plot(ns[sel],1e3*y[sel],marker=m,ms=9,color=c,lw=2.2,markeredgecolor='black',markeredgewidth=0.7)
ins.axhline(0,ls='--',color='0.3',lw=1.6); ins.set_xscale('log'); ins.set_xlim(7e4,1.5e7); ins.set_ylim(-2.4,6.4); ins.set_yticks([-2,0,2,4,6])
ins.set_xticks([1e5,1e6,1e7]); ins.minorticks_off(); ins.tick_params(labelsize=17)
ins.set_title('detail: classical criteria cross zero',fontsize=17,pad=26,loc='right'); ins.text(0.0,1.02,r'$\times10^{-3}$',transform=ins.transAxes,fontsize=15,ha='left',va='bottom')
ttl=ins.title; mult=ins.texts[0]
# overlap audit
fig.canvas.draw(); r=fig.canvas.get_renderer()
ov=lambda a,b: not (a.x1<=b.x0 or b.x1<=a.x0 or a.y1<=b.y0 or b.y1<=a.y0)
items=[]
for a in (ax,ins):
    for t in [a.title,a.xaxis.label,a.yaxis.label]+list(a.texts)+a.get_xticklabels()+a.get_yticklabels():
        if t.get_text().strip(): items.append((t.get_text()[:18],t.get_window_extent(renderer=r)))
items.append(('LEGEND',ax.get_legend().get_window_extent(renderer=r))); items.append(('INSET',ins.get_tightbbox(r)))
bad=[(items[i][0],items[j][0]) for i in range(len(items)) for j in range(i+1,len(items)) if ov(items[i][1],items[j][1]) and 'INSET' not in (items[i][0],items[j][0])]
bad+= [(n,'INSET') for n,b in items if n not in ('INSET',) and ov(b,items[-1][1]) and n not in [t.get_text()[:18] for t in ins.get_xticklabels()+ins.get_yticklabels()+[ins.title]+list(ins.texts)]]
out=str(PLOTS/'Fig1b_panel')
fig.savefig(out+'.png',dpi=200,bbox_inches=None); fig.savefig(out+'.pdf',bbox_inches=None); fig.savefig(out+'.svg',bbox_inches=None)
from PIL import Image; fig.canvas.draw(); rr=fig.canvas.get_renderer(); tb,mb=ins.title.get_window_extent(renderer=rr),ins.texts[0].get_window_extent(renderer=rr); allb=[t.get_window_extent(renderer=rr) for a in (ax,ins) for t in [a.title,a.xaxis.label,a.yaxis.label]+list(a.texts)+a.get_xticklabels()+a.get_yticklabels() if t.get_text().strip()]; nov=sum(1 for i in range(len(allb)) for j in range(i+1,len(allb)) if ov(allb[i],allb[j])); xmax=max(ns); print('last tick >= last data point: main',max(ax.get_xticks())>=xmax,'| inset',max(ins.get_xticks())>=xmax); print('title vs x10^-3 overlap:',ov(tb,mb),'| all text-pair overlaps:',nov)
print("png size:",Image.open(out+'.png').size,"| overlaps:",bad or "none")
