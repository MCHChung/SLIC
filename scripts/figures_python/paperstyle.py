import matplotlib as mpl
def apply_paper_style():
    mpl.rcParams.update({'font.family':'serif','font.serif':['CMU Serif','cmr10','DejaVu Serif'],
        'mathtext.fontset':'cm','axes.formatter.use_mathtext':True,'axes.unicode_minus':False,
        'font.size':26,'axes.titlesize':30,'axes.labelsize':31,'xtick.labelsize':26,'ytick.labelsize':26,
        'legend.fontsize':30,'axes.linewidth':2.1,'xtick.direction':'in','ytick.direction':'in',
        'xtick.major.size':9,'ytick.major.size':9,'xtick.major.width':1.9,'ytick.major.width':1.9,
        'xtick.top':True,'ytick.right':True,'legend.frameon':True,'legend.edgecolor':'black',
        'legend.framealpha':1.0,'figure.dpi':150,'savefig.dpi':220,'savefig.bbox':'tight'})
PCOLOR={'slic':'#e8121c','aic':'#1a1a1a','aicc':'#1c1cc8','bic':'#1e9614','hqic':'#e8a020','kic':'#f0e01e','bc':'#9622a0'}
PMARK={'slic':'o','aic':'^','aicc':'v','bic':'s','hqic':'*','kic':'o','bc':'p'}
PLW={k:(4.4 if k=='slic' else 3.0) for k in PCOLOR}
PMS={'slic':15,'aic':13,'aicc':13,'bic':13,'hqic':19,'kic':14,'bc':14}
PORDER=['slic','aic','aicc','bic','hqic','kic','bc']
PLABEL={'slic':'SLIC','aic':'AIC','aicc':'AICc','bic':'BIC','hqic':'HQIC','kic':'KIC','bc':'BC'}
