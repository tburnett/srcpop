import sys
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
import seaborn as sns
from pylib.fermi_sources import FermiSources
from pylib.ipynb_docgen import capture_hide, show

def outline():
    show("""# Fermi-LAT Unid classification studies
    Basic context: use ML prediction to select pulsar-like Unids<br>
    Topics:
    * ML selection: compare
      1. Multiclass (6 classes, 10 features including position)
      2. UW  (3 classes, 4 features, no position) 
    * Galactic diffuse correlation
    * Characterization of pulsar-like Unids, select non-pulsar subset via KDE
    * Curvature issue
    """)


def ternary_plot(df, columns=None, ax=None):
    import ternary
    if columns is None: 
        columns=df.columns
    assert len(columns==3)
    fig, ax = plt.subplots(figsize=(8,8))

    tax = ternary.TernaryAxesSubplot(ax=ax,)
    
    tax.right_corner_label(columns[0], fontsize=16)
    tax.top_corner_label(columns[1], fontsize=16)
    tax.left_corner_label(columns[2], fontsize=16)
    tax.scatter(df.iloc[:,0:3].to_numpy(), marker='o',
                s=10,c=None, vmin=None, vmax=None)#'cyan');
    ax.grid(False); ax.axis('off')
    tax.clear_matplotlib_ticks()
    tax.set_background_color('0.3')
    tax.boundary()
    return fig

def uw_version():

    show("""---\n## UW version
    Three classes: bll, fsrq, (psr+msp)->psr""")
    self = FermiSources()
    self.train_predict(show_confusion=True)

    unid_prob = self.predict_prob()['p_psr p_fsrq p_bll'.split()]
    show(f"""* Get prediction probabilies applied to the {len(unid_prob)} Unid sources.""")

    show('## UW Unid classification probabilites')
    axx = unid_prob.hist(bins=50, layout=(1,3), figsize=(12,3))
    show(axx[0,0].figure)

    show(ternary_plot(unid_prob)  )
    return self

def multiclass_version():

    show(f"""---\n## Multi-class version  """)
    show("""[Dima, Aakash paper](https://arxiv.org/abs/2301.07412)""")

    def get_six(filename='tmp/Ternary-20231020T115713Z-001/Ternary/4FGL-DR4_6classes_GMM_prob_cat.csv'):
        filename = Path(filename)
        assert filename.is_file()
        sixclass = pd.read_csv(filename, index_col=0)    
        sixclass.index = list(map(lambda s:s[:17], sixclass.index))
        show(f'* Read the six-class file "{filename.name}", found {len(sixclass)} sources.')
        show(f""" Columns:<br> {list(sixclass.columns)}""")
        return sixclass
    sixdf = get_six(); 

    show("* Assign associations, combining msp  with psr")
    def associate(cls):
        t = cls.lower()
        if t in 'fsrq bll glc bcu unk spp psr'.split(): return t
        if t=='msp': return 'psr' 
        if t=='unas': return 'unid' 
        return 'other'
    sixdf['association'] = sixdf.CLASS1.apply(associate)

    show(sixdf.groupby('association').size())

    # fig, axx = plt.subplots(2,3, figsize=(15,10), 
    #                     gridspec_kw=dict(wspace=0.1), sharey=True)
    # for i,ax in enumerate(axx.flat):

    #     hkw = dict(hue_order='bll fsrq psr'.split(),
    #                             palette='yellow magenta cyan'.split(), #edgecolor=None,
    #                                 hue ='association')
    #     sns.histplot(sixdf, ax=ax, x=f'{i+1}_NN', element='step', #kde=True,
    #                 **hkw,
    #                 bins=np.arange(0,1.01,0.05), log_scale=(False, True) )

    # show(f"""## Multi-class probabilites""")
    # show(fig)

    unid = sixdf.query('association=="unid"').copy()
    pcols = np.array([unid['2_NN']+unid['3_NN'], unid['4_NN'], unid['6_NN']])#, unid['1_NN']+unid['5_NN']])
    pnorm = pcols.sum(axis=0)
    z = (pcols/pnorm).T; z
    norm_probs = pd.DataFrame(z, index=unid.index, columns='psr fsrq bll'.split()) 
    
    # unid.rename(columns={'3_NN':'p_psr', "4_NN":'p_fsrq', "6_NN":'p_bll'}, inplace=True)
    cols='p_psr p_fsrq p_bll'.split()
    # probs = unid.loc[:,cols]

    # psum = probs.sum(axis=1)
    # imax = norm_probs.to_numpy().argmax(axis=1)
    # # norm_probs = probs.iloc[:,0:3]/psum.values.reshape(len(psum),1)
    # norm_probs['imax'] = [cols[i] for i in imax]
    # show(norm_probs.groupby('imax').size())

    show("""## Multi-class unid probabilites """)
    axx = norm_probs.iloc[:,0:3].hist(bins=50, layout=(1,3), figsize=(12,3))
    show(axx[0,0].figure)
    show(ternary_plot(norm_probs)   ) 


if 'talk' in sys.argv:
    outline()
    multiclass_version()
    self=uw_version()
    # show(f"""## Galactic diffuse flux as a feature
    # """)
    
    # from summary import Summary
    # summary=Summary()
    # plt.style.use('dark_background') #s
    # plt.rcParams['grid.color']='0.5'
    # dark_mode=True

    # summary.setup_diffuse()
    # summary.ait()
    # summary.show_diffuse_flux()
    # show("""## Apply KDE to GU candidates.""")
    # from kde import kde_analysis
    # kde_analysis(summary.df)
    # show("""### Anti-center: Need to take this into account""")
    # show(summary.zea(180,0).figure)

