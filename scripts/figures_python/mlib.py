import numpy as np, h5py, os
def get(f,key):
    o=f[key]; v=o[()]
    if hasattr(v,'dtype') and v.dtype.names and 'parent_' in v.dtype.names: return f[v['parent_']][:]
    return v
def all_metrics(A,Xt):
    """A: (runs, n_state, n_lib); Xt: (n_lib, n_state)."""
    A=np.asarray(A); Xtn=np.asarray(Xt)
    tnz=(np.abs(Xtn)>1e-12).T; ntot=tnz.size
    XtT=Xtn.T; l1=np.sum(np.abs(XtT))
    rec=[];acc=[];f1=[];fpr=[];fnr=[];nsel=[];smae=[];prec=[];bacc=[]
    for r in range(A.shape[0]):
        M=A[r]
        if M.shape!=XtT.shape: M=M.T
        p=np.abs(M)>1e-12
        TP=np.sum(p&tnz);FP=np.sum(p&~tnz);TN=np.sum(~p&~tnz);FN=np.sum(~p&tnz)
        rec.append(float(np.all(p==tnz))); acc.append((TP+TN)/ntot)
        pr=TP/(TP+FP) if TP+FP else 0.0; rc=TP/(TP+FN) if TP+FN else 0.0
        tnr=TN/(TN+FP) if TN+FP else 0.0
        f1.append(2*pr*rc/(pr+rc) if pr+rc else 0.0)
        fpr.append(FP/(FP+TN) if FP+TN else 0.0)
        fnr.append(FN/(FN+TP) if FN+TP else 0.0)
        prec.append(pr); bacc.append(0.5*(rc+tnr)); nsel.append(p.sum())
        smae.append(np.sum(np.abs(XtT-M))/l1 if l1>0 else np.nan)
    # SMAE is an unbounded heavy-tailed ratio: report the median, not the mean.
    return {'rec':np.mean(rec),'acc':np.mean(acc),'f1':np.mean(f1),'fpr':np.mean(fpr),
            'fnr':np.mean(fnr),'nsel':np.mean(nsel),'smae':np.median(smae),
            'prec':np.mean(prec),'bacc':np.mean(bacc),
            'smae90':np.percentile(smae,90),'badfrac':float(np.mean(np.asarray(smae)>0.5))}
