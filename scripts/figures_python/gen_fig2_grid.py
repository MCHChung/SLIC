"""Figure 2 metrics grid (rows a-e of the published composite, without panel letters, which are
PowerPoint text). Five metrics x (six systems + mean over systems with standard-error shading)."""
import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from paths import RESULTS, EXP, PLOTS
import h5py, glob, numpy as np, sys, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from paperstyle import apply_paper_style, PCOLOR, PMARK
apply_paper_style(); matplotlib.rcParams['savefig.bbox']='standard'
def get(f,key):
    v=f[key][()]
    return f[v['parent_']][:] if hasattr(v,'dtype') and v.dtype.names and 'parent_' in v.dtype.names else v
ICS=['slic','aic','aicc','bic','hqic','kic','bc']; LBL=dict(zip(ICS,['SLIC','AIC','AICc','BIC','HQIC','KIC','BC']))
SYS=[('lorenz','Lorenz'),('rossler','Rossler'),('lotka-volterra','Lotka-Volterra'),('brusselator','Brusselator'),('van_der_pol','Van der Pol'),('nonlin_pendulum','Nonlinear pendulum')]
NOISE=[0,5,10,20,30,40]
def metrics(A,Xt):
    XtT=np.asarray(Xt).T; tnz=np.abs(XtT)>1e-12; l1=np.abs(XtT).sum(); o={k:[] for k in ['acc','fpr','rec','smae','fnr']}
    for M in A:
        M=M if M.shape==XtT.shape else M.T; p=np.abs(M)>1e-12
        TP=np.sum(p&tnz);FP=np.sum(p&~tnz);TN=np.sum(~p&~tnz);FN=np.sum(~p&tnz)
        o['acc'].append((TP+TN)/tnz.size); o['fpr'].append(FP/(FP+TN) if FP+TN else 0); o['fnr'].append(FN/(FN+TP) if FN+TP else 0)
        o['rec'].append(float(np.all(p==tnz))); o['smae'].append(np.abs(XtT-M).sum()/l1)
    return {'acc':np.mean(o['acc']),'fpr':np.mean(o['fpr']),'rec':np.mean(o['rec']),'fnr':np.mean(o['fnr']),'smae':np.log10(np.median(o['smae']))}
V={}
for s,_ in SYS:
    for ni,nl in enumerate(NOISE,1):
        with h5py.File(glob.glob(f'{RESULTS}/fig2_benchmarks/{s}_neffall_noise{ni}_results.jld')[0],'r') as f:
            Xt=get(f,'Xitrue')
            for ic in ICS: V[(s,nl,ic)]=metrics(np.asarray(get(f,f'Xis_neffall_{ic}')),Xt)
ROWS=[('acc','Per-term\naccuracy'),('fpr','False-positive\nrate'),('rec','System-level\nrecovery'),('smae','$\\log_{10}$ SMAE\n(median)'),('fnr','False-negative\nrate')]
COLS=[s for s,_ in SYS]+['mean']; TITLE=dict(SYS); TITLE['mean']='All systems\n(mean)'
def cover_ticks(lo,hi):
    unit=1.0 if hi-lo<=3.2 else 2.0
    a=np.floor(lo/unit+1e-9)*unit; b=np.ceil(hi/unit-1e-9)*unit
    return np.arange(a,b+unit/2,unit)
fig,axes=plt.subplots(5,7,figsize=(26,16.5))
plt.subplots_adjust(left=0.075,right=0.99,top=0.955,bottom=0.12,wspace=0.28,hspace=0.30)
for r,(m,ylab) in enumerate(ROWS):
    for c,col in enumerate(COLS):
        ax=axes[r,c]
        for ic in ['bc','kic','hqic','bic','aicc','aic','slic']:
            if col=='mean':
                M=np.array([[V[(s,nl,ic)][m] for nl in NOISE] for s,_ in SYS]); y=M.mean(0); se=M.std(0,ddof=1)/np.sqrt(len(SYS))
                ax.fill_between(NOISE,y-se,y+se,color=PCOLOR[ic],alpha=0.22,lw=0,zorder=1)
            else: y=[V[(col,nl,ic)][m] for nl in NOISE]
            ax.plot(NOISE,y,marker=PMARK[ic],ms=9 if ic=='slic' else 7,color=PCOLOR[ic],lw=3.0 if ic=='slic' else 1.6,
                    markeredgecolor='black',markeredgewidth=0.5,zorder=6 if ic=='slic' else 3,label=LBL[ic])
        if m=='smae':
            lines=[l for l in ax.get_lines()]; ys=np.concatenate([l.get_ydata() for l in lines])
            if col=='mean':
                for coll in ax.collections: ys=np.concatenate([ys,coll.get_paths()[0].vertices[:,1]])
            tk=cover_ticks(ys.min(),ys.max()); pad=0.06*(tk[-1]-tk[0]); ax.set_ylim(tk[0]-pad,tk[-1]+pad); ax.set_yticks(tk); ax.set_yticklabels([f'{t:g}' for t in tk])
        else:
            ax.set_ylim(-0.05,1.07); ax.set_yticks([0,0.5,1.0])
            if c==0: ax.set_yticklabels(['0.0','0.5','1.0'])
            else: ax.set_yticklabels([])
        ax.set_xlim(-2,42); ax.set_xticks([0,5,10,20,30,40]); ax.tick_params(labelsize=17,length=5)
        if r==4: ax.set_xlabel('Noise level (%)',fontsize=20)
        else: ax.set_xticklabels([])
        if r==0: ax.set_title(TITLE[col],fontsize=21,pad=10)
        if c==0: ax.set_ylabel(ylab,fontsize=20)
h,l=axes[0,0].get_legend_handles_labels(); order=[l.index(x) for x in ['SLIC','AIC','AICc','BIC','HQIC','KIC','BC']]
fig.legend([h[i] for i in order],[l[i] for i in order],loc='lower center',ncol=7,fontsize=22,bbox_to_anchor=(0.53,0.0),markerscale=1.4)
if __name__=='__main__':
    from tickcheck import check; check(fig,'Figure 2 grid')
    fig.canvas.draw(); rr=fig.canvas.get_renderer(); ov=lambda p,q: not (p.x1<=q.x0 or q.x1<=p.x0 or p.y1<=q.y0 or q.y1<=p.y0)
    it=[t.get_window_extent(renderer=rr) for a in fig.axes for t in [a.title,a.xaxis.label,a.yaxis.label]+a.get_xticklabels()+a.get_yticklabels() if t.get_text().strip()]+[fig.legends[0].get_window_extent(renderer=rr)]
    print("text overlaps:",sum(1 for i in range(len(it)) for j in range(i+1,len(it)) if ov(it[i],it[j])))
    out=str(PLOTS/'Fig2_grid')
    fig.savefig(out+'.pdf'); fig.savefig(out+'.png',dpi=250); fig.savefig(out+'.svg')
