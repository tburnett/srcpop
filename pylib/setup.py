"""
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from utilities.ipynb_docgen import capture_hide, show
from utilities.catalogs import Fermi4FGL, UWcat

sns.set()
plt.rcParams['font.size']=16

def show_date():
    import datetime
    date=str(datetime.datetime.now())[:16]
    show(f"""<h5 style="text-align:right; margin-right:15px"> {date}</h5>""")

def load_data(datafile= 'files/fermi_sources.csv',
              selection='delta<0.2 & curvature<0.7'):

    t = pd.read_csv(datafile,index_col=0 )
    df = t.query(selection).copy()
    show(f"""Read {len(t)} source entries from `{datafile}`, selected {len(df)} with criteria '{selection}'""")  

    # descriptive rename and clean
    df.rename(columns=dict(singlat='sin_b', eflux100='eflux', variability='var'),inplace=True)
    df.drop(columns='ts r95 delta glat glon'.split(),inplace=True)
    # need log10 versions for seaborn
    df.loc[:,'log_nbb']   = np.log10(df.nbb.clip(1, 100))
    df.loc[:,'log_var']   = np.log10(df['var'].clip(0,1e4))
    df.loc[:,'log_eflux'] = np.log10(df.eflux.clip(1e-13,1e-9))
    df.loc[:,'log_e0']    = np.log10(df.e0)
    df.loc[:,'abs_sin_b'] = np.abs(df.sin_b)
    return df

fgl = uw = None
def load_cats():
    global fgl, uw
    with capture_hide("""Load UW, 4FGL-DR4 catalogs to look up spectral functions""") as cat:
        from utilities.catalogs import Fermi4FGL, UWcat
        fgl = Fermi4FGL()
        uw  = UWcat('uw1410')
        uw.index = uw.jname
        show(cat)
    return fgl, uw

def plot_seds(df, start=0, ncols=10, 
              tooltips=None, **kwargs):
    """Blue: uw1410, red 4FGL-DR4
    """
    from utilities.spectral_functions import MultiSED
    # df has 4FGL name index
    # df.uw_name is UW jnames
    dfsel = df.iloc[slice(start,100)]
    sfcat = fgl.loc[dfsel.index].specfunc
    sfuw  =  uw.loc[dfsel.uw_name].specfunc

    ms = MultiSED(len(dfsel),tooltips=tooltips,
                  caption='Blue: uw1410, red 4FGL-DR4')
    ms.plots(sfuw, lw=3, color='blue', alpha=0.5)
    ms.plots(sfcat, color='red', alpha=0.5)
    
    return ms
    
def show_data(df):
    
    show(f"""* Summary of the numerical contents of selected data,<br>
    the "features" that can be used for population analysis.""")
  
    pd.set_option('display.precision', 3)
    show(df['eflux  pindex curvature e0 sin_b nbb var'.split()].describe(percentiles=[0.5]))
    show(r"""
        | Feature   | Description 
        |-------    | ----------- 
        |`eflux`    | Energy flux for E>100 Mev, in erg cm-2 s-1 
        |`pindex`   | Spectral index
        |`curvature`| Spectral curvature, the parameter $\beta$ for log-parabola
        |`e0`       | Spectral scale energy, close to the "pivot"
        | `sin_b`   | $\sin(b)$, where $b$ is the Galactic latitude 
        |`var`      | `Variability_Index` parameter from 4FGL-DR4 
        |`nbb`      | Number of Bayesian Block intervals from the wtlike analysis 
    """)

    show(f"""* Values and counts of the `category` column""")
    cats = np.unique(df.category) #= 'bll fsrq psr'.split()
    cnt = pd.Series(dict([(cat ,sum(df.category==cat)) for cat in cats]))
    show(cnt, index=False)

def explore_correlations(df):

    training = 'bll fsrq psr'.split()
    dft = df.loc[df.category.apply(lambda cat: cat in training)].copy()

    features = 'log_nbb log_var sin_b category log_eflux'.split()

    show(f"""### The well-associated categories set: {training}
    These can be used for ML training.
    
    """)

    sns.pairplot(dft[features],kind='kde',hue='category');
    show(plt.gcf())
    show("""Notes:
    * For the blazars, the `sin_b` distribution, which should be
    flat, shows a selection bias:
    Catalogs used for association do not include the galactic plane.
    * The `log_var` vs. `log_nbb` comparison shows that the latter is more
    sensitive to detect BL Lac variability
    * Pulsars, as expected, show very little variability with either measure.   
    <br>
    """)
    
    
def the_unknowns(df):
    dfu = df.query('category=="unid"').copy()
    show(rf"""### The {len(dfu)} unknown sources 
    Add the spectral index and curvature to the four features and leave off variability.
    <br>Use `abs_sin_b` = $|(\sin(b)|$ since expect it to be symmetric.
    """)

    unk_features='log_nbb abs_sin_b log_eflux pindex curvature category'.split()
    sns.pairplot(dfu[unk_features], kind='kde',hue='category');
    show(plt.gcf())
    show(f"""Notes:
    * Almost every pair correlation clearly shows a separtion between pulsars and blazars!
    """)
    
def compare_bcu_bll(df):
    
    dfv = df.query('category=="bcu" | category=="bll" ').copy()

    show(rf"""### Compare the BCU and BLL sources 
    Expect BCU to be a mixture of FSRQ and BL Lac

    """)

    unk_features='log_nbb log_var log_eflux pindex curvature category'.split()
    sns.pairplot(dfv[unk_features], kind='kde',hue='category');
    show(plt.gcf())
    show("""BCU has a soft tail? """)
    
def chi2_plots(df, cuts=['nbb==1','nbb>1'] , cats='unid bcu', vmax=40):
    
    fig, axx  = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10,3),
                            gridspec_kw=dict(wspace=0.05))
    hkw = dict(bins=np.linspace(0,vmax,26), density=True, histtype='step', lw=2)
    
    for cut, ax in zip(cuts, axx.flatten()):
        df1 = df.query(cut)
        var = df1['var'].clip(0,vmax)

        for cat in cats.split():
            sub = df1.category==cat
            assert(sum(sub))>0, f'No {cat} entries'
            ax.hist(var[sub], label=f'{cat} ({sum(sub)})', **hkw)

        from scipy.stats import chi2
        x = np.arange(0,40, 0.5)
        ax.plot(x, chi2.pdf(x,13),ls='--', label='$\chi^2_{13}$')
        ax.legend(loc='lower center');
        ax.set(xlabel='Variability Index', yscale='log', ylim=(1e-4,0.1),
               title=cut);
    show(fr"""### Variability_Index is a chi-squared
    This is defined as -2 times the log of the likelihood ratio for the 14 yearly DR4
    flux measurments to be the same, which is distributed as $\chi^2$ with 13 degrees of
    freedom for the null hypothesis.
    
    Here we compare the distributions for nbb==1 vs >1. with the {cats} categories.""")
    show(fig)
    show("""For nbb==1 and the unid, the comparison is pretty good. The tail for the bcu
    category needs investigating.""")
    
def summary():    
    import datetime
    date=str(datetime.datetime.now())[:16]
    show(f"""# Population studies to estimate components of the unids
    <h5 style="text-align:right"> {date}</h5>""")
    show("""
    These input data are derived from the uw1410 model used as seeds for 4FGL-DR4. For all 
    with TS>25, Bayesian-block light curves, with weekly time bins, were determined. The `nbb` 
    variability measure being used here is a count of the blocks so determined. 
    The `Variability_Index` quantity from the yearly light curve determined in the standard catalog analysis was included in the table. All other quantities are from the uw1410 spectral analysis.
    
    """)
    df = load_data()
    show_data(df)
    show(f"""## Explore correlations using `seaborn.pairplot`    """)
    explore_correlations(df)
    the_unknowns(df)
    compare_bcu_bll(df)
    chi2_plots(df)
    
if len(sys.argv)>1:
    name = sys.argv[1]
    if name=='summary': summary()
    else:
        eval(f'{sys.argv[1]}({df})')