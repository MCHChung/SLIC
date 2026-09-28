"""Report any axis whose plotted data extend past its outermost visible ticks.
A small tolerance (0.5% of the axis span) ignores sub-pixel excursions such as a
value of -3e-4 plotted on a 0 to 2.5 axis."""
import numpy as np
def data_extent(ax):
    xs,ys=[],[]
    for ln in ax.get_lines():
        x,y=np.asarray(ln.get_xdata(),float),np.asarray(ln.get_ydata(),float)
        if x.size>2 and np.ptp(x)==0 and len(set(y))==2: continue   # axvline
        if y.size>=2 and np.ptp(y)==0 and len(set(x))==2 and set(x)<= {0.0,1.0}: continue  # axhline
        m=np.isfinite(x)&np.isfinite(y); xs+=list(x[m]); ys+=list(y[m])
    for c in ax.collections:
        o=c.get_offsets()
        if len(o): xs+=list(np.asarray(o)[:,0]); ys+=list(np.asarray(o)[:,1])
    return (min(xs),max(xs),min(ys),max(ys)) if xs else None
def check(fig,name=''):
    problems=[]
    for i,ax in enumerate(fig.axes):
        e=data_extent(ax)
        if e is None: continue
        x0,x1,y0,y1=e
        for axis,(lo,hi),lim,scale in [('x',(x0,x1),ax.get_xlim(),ax.get_xscale()),('y',(y0,y1),ax.get_ylim(),ax.get_yscale())]:
            ticks=np.array(ax.get_xticks() if axis=='x' else ax.get_yticks(),float)
            ticks=ticks[(ticks>=min(lim)-1e-12)&(ticks<=max(lim)+1e-12)]
            if ticks.size==0: continue
            ax.figure.canvas.draw()
            k=0 if axis=='x' else 1
            f=lambda v,k=k,axis=axis: ax.transData.transform((v,0) if axis=='x' else (0,v))[k]
            span=abs(f(max(lim))-f(min(lim))); tol=0.005*span
            vis_lo,vis_hi=max(lo,min(lim)),min(hi,max(lim))            # only what is drawn inside the axes
            clipped=(lo<min(lim)) or (hi>max(lim))
            if f(vis_hi)>f(ticks.max())+tol: problems.append(f"axes {i} ({ax.get_title()[:24]!r}) {axis}: data to {vis_hi:.4g} beyond last tick {ticks.max():.4g}")
            if f(vis_lo)<f(ticks.min())-tol: problems.append(f"axes {i} ({ax.get_title()[:24]!r}) {axis}: data from {vis_lo:.4g} before first tick {ticks.min():.4g}")
            if clipped: problems.append(f"axes {i} ({ax.get_title()[:24]!r}) {axis}: data clipped by the axis limits ({lo:.4g} to {hi:.4g} vs {lim[0]:.4g} to {lim[1]:.4g})")
    print(f"[tick check] {name}: " + ("OK" if not problems else f"{len(problems)} problem(s)"))
    for p in problems: print("     -",p)
    return problems
