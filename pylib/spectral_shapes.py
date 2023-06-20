
from fermi_sources import *

fs_data = None
train_df= unid_df=None

def intro():
    show(f"""# Spectral shape analysis""")
    def show_date():
        import datetime
        date=str(datetime.datetime.now())[:16]
        show(f"""<h5 style="text-align:right; margin-right:15px"> {date}</h5>""")
    show_date()
    show("""This examines the role of the spectral shape parameters curvature and peak energy in an ML 
    analysis of associated sources, used to \
    predict identies of the unassociated. The spectral fits are from uw1410, which may differ from the
    4FGL-DR4 catalog, especially for weak curved sources.
    """)

def setup():
    global fs_data , train_df, unid_df
    show(f"""## Run the classification 
        We use scikit-learn to implement ML techniques. So far only one model, `GaussianNB`.
        """)
    fs_data = FermiSources('files/fermi_sources_v2.csv')
    from sklearn.naive_bayes import GaussianNB 
    model = GaussianNB()

    classifier = fs_data.fit(model=model)
    fs_data.df.loc[:,'prediction'] = fs_data.predict(classifier, 'association=="unid"')

    fs_data.confusion_display()

    # global references to the data sets for plots below.
    target_names =fs_data.mlspec.target_names
    df = fs_data.df
    train_df = df[df.association.apply(lambda x: x in target_names)]
    unid_df = df[df.association=='unid']



def scatter_train_predict(df, x,y, caption, **kwargs):
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6),
                                  sharex=True,sharey=True,
                                 gridspec_kw=dict(wspace=0.02))

    target_names =fs_data.mlspec.target_names

    kw = dict()
    kw.update(kwargs)
    ax1.set(**kw)

    sns.scatterplot(train_df, x=x, y=y, 
                    hue_order=target_names, hue='association', ax=ax1)
    sns.scatterplot(unid_df, x=x, y=y,  
                    hue_order=target_names, hue='prediction', ax=ax2)
    show(fig, caption=caption)

def curvature_epeak_flux():
    show(f"""### Curvature vs Epeak: compare training and unid sets""")


    scatter_train_predict(fs_data.df,x='log_epeak', y='curvature',
            caption=f"""Curvature vs peak energy for the training set on the
        left, the unid on the right.""",
                        xticks = [-1,0,1,2,3], yticks=[0,0.5,1])
    show(f"""Note that the curvature distribution is shifted to higher values in for the unid 
    data.
    """)

    show(f"""### Curvature vs. eflux
        Check the dependence of the curvature on the flux.
        """)

    scatter_train_predict(fs_data.df,
                        x='log_eflux', y='curvature',
            caption=f"""Curvature vs eflux for associated sources on the
        left, the unid on the right.""",
                    xticks=[-12,-11,-10,],    yticks=[0,0.5,1])

def curvature_scatter_by_type():

    show(f"""### Scatter plots of curvature vs. Epeak for associated source types""")
    fig, axx = plt.subplots(ncols=2, nrows=2, figsize=(12,12), sharex=True, sharey=True,
                        gridspec_kw=dict(wspace=0.05))
    bname = dict(fsrq='FSRQ', bll='BL Lac', psr='PSR')
    for btype, ax in zip(('fsrq', 'bll','psr'), axx.flatten()):

        df = train_df.query(f'association=="{btype}" & -0.9<log_epeak<1')
        ax.set(xticks=[-1,-0.5,0,0.5,1], title=f'{bname[btype]}s ({len(df)})', )
        (sns.scatterplot(df,  x='log_epeak', y='curvature', ax=ax, hue='log_eflux', 
                        size='log_eflux', sizes=(20,200))
                .set(xlabel='Epeak (GeV)', xticks = [-1,0,1],xticklabels='0.1 1 10'.split())
        )
        ax.legend(loc='upper left', fontsize=12)
    axx[1,1].set_visible(False)
    show(fig)
    show(f"""Comments:
    * FSRQ's mostly peak below 1 GeV, while BL Lacs extend up to 10 GeV
    * For blazars, lower fluxes apparently correspond to an increase in curvature and peak energy, 
    especially for curvature>0.5. I suspect that it is a 
    systematic, but these papers ascribe it to a "blazar sequence". https://arxiv.org/abs/1702.02571  and
    https://arxiv.org/pdf/2305.02087.pdf .
    * Pulsars are all fit with a PLEX, complicating the curvature comparison. 
    Strong pulsars can have large curvatures, but those above 0.6 are very weak.
    The correlation pointed out in 3PC is apparent.
        """)
    
def sed_plots():
    show(f"""## SED plots ordered by flux
         Reference plots of 100 each of the SEDs with moderate Epeaks. The maps have tooltips with source info. 
         """)
    epeak_range = '-0.5<log_epeak<0.5 & curvature>0.1'
    dfs = fs_data.df.query(epeak_range).sort_values('log_eflux', ascending=False)
    show(f"""
    Initially, select epeak with {epeak_range}, result in {len(dfs)} sources.
    Sort in descending eflux.
    """)  

    for src_type in 'psr fsrq bll'.split():

        dfx = dfs.loc[dfs.association==src_type,
                'ts glat log_epeak curvature association prediction log_eflux uw_name'.split() ]
        n = len(dfx)
        
        show(f'* {src_type}: showing every {n//100}')
        
        sedplotgrid( dfx.iloc[::n//100],  nrows=10, ncols=10, ylim=(0.1,100)
        )

def main():
    intro()
    setup()
    show("""## Plots showing spectral correlations""")
    curvature_epeak_flux()
    curvature_scatter_by_type()
    sed_plots()

import sys
if len(sys.argv)>1:
    arg = sys.argv[1]
    if arg=='doc':
        main()
    elif arg=='setup':
        setup()



# (
# sns.PairGrid(train_df, vars='log_eflux log_epeak curvature'.split(), 
#              hue='association', hue_order='bll fsrq psr'.split(),
#              corner=True, height=2) 

#     .map_diag(sns.kdeplot, fill=True, ) #histplot,  bins=50, shrink=1, element='step')
#     .map_lower(sns.scatterplot)
#     .add_legend(loc=(0.8,0.8))
# );

# (
# sns.PairGrid(unid_df, vars='log_eflux log_epeak curvature abs_sin_b'.split(), 
#              hue='prediction', hue_order='bll fsrq psr'.split(),
#              corner=True, height=2) 

#     .map_diag(sns.kdeplot, fill=True) #histplot,  bins=50, shrink=1, element='step')
#     .map_lower(sns.scatterplot)
#     .add_legend(loc=(0.8,0.8))
# );