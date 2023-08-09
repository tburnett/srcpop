"""
"""
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from astropy.coordinates import SkyCoord
from utilities.ipynb_docgen import capture_hide, show
from utilities.catalogs import UWcat, Fermi4FGL

import seaborn as sns
sns.set_theme(font_scale=1.25)
plt.rcParams['font.size']=14


from dataclasses import dataclass

@dataclass
class FigNum:
    n = 0
    @property
    def current(self): return self.n
    @property
    def next(self):
        self.n +=1
        return self.n
    
def fpeak_kw(axis='x'):
    return {axis+'label':r'$F_p \ \ \mathrm{(eV\ s^{-1}\ cm^{-2})}$', 
            axis+'ticks': np.arange(-2,5,2),
            axis+'ticklabels': '$10^{-2}$ 1 100 $10^4$'.split(), 
            }
def epeak_kw(axis='x'):
    return {axis+'label':'$E_p$  (GeV)',
            axis+'ticks': np.arange(-1,3.1,1),
            axis+'ticklabels':'0.1 1 10 100 1000'.split(),
            }

def show_date():
    import datetime
    date=str(datetime.datetime.now())[:16]
    show(f"""<h5 style="text-align:right; margin-right:15px"> {date}</h5>""")

@dataclass
class MLspec:
    features: tuple = tuple("""log_var log_fpeak log_epeak curvature """.split())

    target : str ='association'
    target_names:tuple = tuple('bll fsrq psr'.split())

    def __repr__(self):
        return f"""
        Scikit-learn specifications: 
        * features: {self.features}
        * target: {self.target}
        * target_names: {self.target_names}"""

    
class FermiSources:
    """
    Manage the set of Fermi sources
    """
    
    def __init__(self, datafile= 'files/fermi_sources_v2.csv',
                mlspec=None,
                selection='delta<0.25 & curvature<2.1'):
        
        t = pd.read_csv(datafile, index_col=0 )
        t.curvature *=2 # temporary fix to be "spectral curvature" here

        with capture_hide('setup printout') as self.setup_output:
            self.fermicat = Fermi4FGL(); 
            self.uwcat = UWcat().set_index('jname')
            # look up the beta uncertainty from the UW cat
            t['d_unc'] = 2*self.uwcat.errs.apply(lambda s: np.array(s[1:-1].split(), float)[2])
            self.df = df = t.query(selection).copy()
            print(f"""Read {len(t)} source entries from `{datafile}`, selected {len(df)} with criteria '{selection}'""")  




        # descriptive rename and clean
        df.rename(columns=dict(singlat='sin_b', eflux100='eflux', variability='var',
                               category='association'),inplace=True)
        self.skycoord = SkyCoord(df.glon, df.glat, unit='deg', frame='galactic')
        #df.drop(columns='delta glat glon'.split(),inplace=True)
        # need log10 versions for sklearn
        df.loc[:,'log_nbb']   = np.log10(df.nbb.clip(1, 100))
        df.loc[:,'log_var']   = np.log10(df['var'].clip(0,1e4))
        df.loc[:,'log_eflux'] = np.log10(df.eflux.clip(1e-13,1e-9))
        df.loc[:,'log_e0']    = np.log10(df.e0)
        df.loc[:,'log_ts']    = np.log10(df.ts.clip(25,3316))
        df.loc[:,'abs_sin_b'] = np.abs(df.sin_b)

        # add log epeak, need lookup 
        sfl = SpecFunLookup(df) #[unid.index[1]]
        specfun = pd.Series(dict([ (idx,sfl[idx]) for idx in df.index]))
        df.loc[:,'log_epeak'] = specfun.apply(lambda f: f.sedfun.peak)
        df['log_fpeak'] = specfun.apply(lambda f: f.fpeak)

        self.mlspec = MLspec() if mlspec is None else mlspec  # default for classification

    def show_data(self):
        """ Make a summary of the """

        # show(f"""* Summary of the numerical contents of selected data""")
                 # show(self.df['eflux  pindex curvature e0 sin_b nbb var'.split()].describe(percentiles=[0.5]))
        show("""The "features" that can be used for population analysis.""")

        pd.set_option('display.precision', 3)
    
        show(r"""
            | Feature   | Description 
            |-------    | ----------- 
            |`eflux`    | Energy flux for E>100 Mev, in erg cm-2 s-1 
            |`pindex`   | Spectral index (problematical since defined differently for PLEX and LP)
            |`curvature`| Spectral curvature, twice the log-parabola parameter $\beta$
            |`e0`       | Spectral scale energy, close to the "pivot"
            |`epeak`    | $E_p$, Energy of SED maximum. limited to (100 MeV-1TeV)
            |`fpeak`    | $F_p$,  differential flux, in eV s-1 cm-2, at `epeak`
            |`sin_b`    | $\sin(b)$, where $b$ is the Galactic latitude 
            |`var`      | `Variability_Index` parameter from 4FGL-DR4 
            |`nbb`      | Number of Bayesian Block intervals from the wtlike analysis 
             
        """)

        show(f"""* Values and counts of the `association` column""")
        fig, ax =plt.subplots(figsize=(6,3))
        sns.countplot(self.df, x='association').set(title='4FGL-DR4 source categories');
        t = self.df.groupby('association').size()
        id = (t.bcu+t.bll+t.fsrq+t.fsrq+t.psr)
        show(f"""Note that the number of pulsar+blazars (including bcu) is {100*id/(id+t.other):.0f}% of the total
            associated.""")
        
    def show_positions(self, xds, figsize=(12,12), colorbar=True, colorbar_kw={},
                       title=None, fignum=None, caption=None):
        """
        * xds - a DataFrame indexed with 4FGL names (like a dataset from here) and 
        columns `glon`, `glat` and `log_flux`

        Show an Aitoff plot with the 
        """
        from wtlike.skymaps import AitoffFigure
        # Couldn't figure out how to modify my own code to avoid this warning
        import warnings
        # warnings.filterwarnings("ignore")
        def fxn():
            warnings.warn("deprecated", DeprecationWarning)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fxn()
        assert len(xds)>0, 'No data to plot'
        xpos = SkyCoord( *xds['glon glat'.split()].values.T, unit='deg', frame='galactic')
        afig = AitoffFigure(figsize=figsize)
        afig.ax.set_facecolor('lavender')
        if title is not None:
            afig.ax.set(title=title)
        scat = afig.scatter(xpos, c=xds.log_eflux);
        cbar_kw = dict(shrink=0.35, ticks=[-12,-11,-10], label='log Eflux')
        if colorbar:
            cbar_kw.update(colorbar_kw)
            cb = plt.colorbar(scat,  **cbar_kw)
        show(afig.fig, fignum=fignum, caption=caption)
    


    def getXy(self, mlspec=None) : #, features, target, target_names=None):
        """Return an X,y pair for ML training
        """
        if mlspec is not None:
            self.mlspec = mlspec
        ml = self.mlspec
        tsel = self.df[ml.target].apply(lambda c: c in ml.target_names)\
            if ml.target_names is not None else slice(None)
        assert sum(tsel)>0, 'No data selected for training.'
        return self.df.loc[tsel, ml.features], self.df.loc[tsel, ml.target]

    def fit(self, model, mlspec=None):
        """
        """
        X,y = self.getXy(mlspec) 
        return model.fit(X,y)

    def predict(self, query=None):
        """Return a "prediction" vector using the classifier, required to be a trained model

        - query -- optional query string

        return a Seriies 
        """
        # the feature names used for the classification -- expect all to be in the dataframe
        assert hasattr(self, 'classifier'), 'Model was not fit'
        fnames = getattr(self.classifier, 'feature_names_in_', [])
        assert np.all(np.isin(fnames, self.df.columns)), f'classifier not set properly'
        dfq = self.df if query is None else self.df.query(query) 
        assert len(dfq)>0, 'No data selected'
        ypred = self.classifier.predict(dfq.loc[:,fnames])
        return pd.Series(ypred, index=dfq.index, name='prediction')
    
    def train_predict(self,  model_name='SVC', show_confusion=False, hide=False, save_to=None):

        def get_model(model_name):
            from sklearn.naive_bayes import GaussianNB 
            from sklearn.svm import  SVC
            from sklearn.tree import DecisionTreeClassifier
            from sklearn.neural_network import MLPClassifier
            from sklearn.ensemble import RandomForestClassifier

            # instantiate the model by looking up the name

            cdict = dict(GNB = (GaussianNB, {}),
                        SVC = (SVC, dict(gamma=2, C=1)), 
                        tree= (DecisionTreeClassifier, {}),
                        RFC = (RandomForestClassifier, dict(n_estimators=100, max_features=2)),
                        NN  = (MLPClassifier, dict(alpha=1, max_iter=1000)),
                    )
            F,kw = cdict[model_name]
            return F(**kw)
        
        model = get_model(model_name)
        self.classifier = self.fit( model )
        self.df.loc[:,'prediction'] = self.predict()

        if show_confusion:
            self.confusion_display(model=model, hide=hide)

        # global references to the data sets for plots below.
        target_names =self.mlspec.target_names
        df = self.df
        self.train_df = df[df.association.apply(lambda x: x in target_names)]
        self.unid_df = df[df.association=='unid']
        # return fs_data, train_df, unid_df
        if save_to is not None:
            try:
                df.to_pickle(save_to)
                show(f'Saved to {save_to}')
            except Exception as e:
                print(f'Attempt to ssve pickle to {save_to} failed, {e}', file=sys.stderr)
                

    
    def pairplot(self, **kwargs):
        
        ml = self.mlspec

        tsel = self.df[ml.target].apply(lambda c: c in ml.target_names)\
            if ml.target_names is not None else slice(None)
        kw = dict(kind='kde', hue=ml.target, height=2, )
        kw.update(**kwargs)
        sns.pairplot(self.df.loc[tsel], vars=ml.features,  **kw)
        return plt.gcf()
    
        
    def confusion_display(self, model, mlspec=None, hide=False, test_size=0.25, **kwargs):
        """
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay 

        Xy = self.getXy(mlspec)
        split_kw = dict(test_size=test_size, shuffle=True)
        split_kw.update(kwargs)
        Xtrain, Xtest, ytrain, ytest  =  train_test_split(*Xy, **split_kw)

        # model = GaussianNB()  if model is None else model     # 2. instantiate model
        classifier = model.fit(Xtrain, ytrain)     # 3. fit model to data
        y_model = model.predict(Xtest)             # 4. predict on new data

        show(f"""### Confusion analysis, test size = {100*test_size} %
        * Model: {str(model)}<br>
        * Features: {list(Xy[0].columns)}<br>
        Accuracy: {100*accuracy_score(ytest, y_model):.0f}%
        """)

        fig, axx = plt.subplots(ncols=2, figsize=(12,5),
                               gridspec_kw=dict(wspace=0.4))
        # Plot non-normalized confusion matrix
        titles_options = [
            ("Unnormalization", None),
            ("Normalized", "true"),
        ]
        for ax, (title, normalize)  in zip(axx.flatten(), titles_options):
            disp = ConfusionMatrixDisplay.from_estimator(
                    classifier, Xtest, ytest,
                    display_labels='bll fsrq psr'.split(),
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=ax,
                )

            ax.set_title(title)
        show(fig, summary=None if not hide else 'Confusion matix plot')

    def scatter_train_predict(self,  x,y,fignum=None, caption='', target='unid', **kwargs):
        
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6),
                                    sharex=True,sharey=True,
                                    gridspec_kw=dict(wspace=0.2))

        target_names = self.mlspec.target_names
        df = self.df

        kw = dict()
        kw.update(kwargs)

        ax1.set_title('Training')
        ax2.set_title(f'{target} prediction')
        sns.scatterplot(self.train_df, x=x, y=y, 
                        hue_order=target_names, hue='association', ax=ax1)
        ax1.set(ylabel='Spectral curvature', **kw)
        ax1.legend(loc='upper right')

        target_df = df.query(f'association=="{target}"')
        assert len(target_df)>0, f'Failed to find target {target}'
        sns.scatterplot(target_df, x=x, y=y,  
                        hue_order=target_names, hue='prediction', ax=ax2)
        ax2.legend(loc='upper right')
        ax2.set(**kw)
        ax2.set(xlabel = kw['xlabel']) #
        show(f'Labels? {ax1.get_xlabel()}, {ax2.get_xlabel()}')
        
        fig.text(0.51, 0.5, '⇨', fontsize=50, ha='center')
        show(fig, fignum=fignum, caption=caption)

    def curvature_epeak_flux(self, fignum=None):
        show(f"""### Curvature vs $E_p$: compare training and unid sets""")


        self.scatter_train_predict(x='log_epeak', y='curvature', fignum=fignum,
                caption=f"""Curvature vs $E_p$ for the training set on the
            left, the unid on the right.""",
                            **epeak_kw('x'),
                            yticks=[0,0.5,1,1.5,2]
                            )
        show(f"""Note that the curvature distribution is shifted to higher values for the unid 
        data.
        """)

        show(f"""### Curvature vs. $F_p$
            Check the dependence of the curvature on the peak flux.
            """)

        self.scatter_train_predict( x='log_fpeak', y='curvature',fignum=fignum+1 if fignum is not None else None,
                caption=f"""Curvature vs $F_p$ for associated sources on the
            left, the unid on the right.""",
                        **fpeak_kw('x'),
                          yticks=[0,0.5,1,1.5,2])

   
def ait_plots(df, hue, hue_order=None, ncols=2, width=6, cbar=True):
    
    class Gfun:
        def __init__(self, hue, hue_order):
            self.hue=hue; self.hue_order=hue_order
        def __call__(self, idx):
            try:
                return self.hue_order.index(df.loc[idx, self.hue])
            except ValueError:
                return

    g = df.groupby(Gfun(hue, hue_order)) if hue_order is not None else df.groupby(hue)
    n = len(g)
    nrows = (n-1)//ncols +1
    fig, axx = plt.subplots(nrows=nrows, ncols=ncols,
                            figsize=(width*ncols+1, width/2*nrows+1, ),
                            gridspec_kw=dict(wspace=0.05, hspace=0.1),
                            subplot_kw=dict(projection='aitoff'));
    
    for (name, dfx), ax in zip(g, axx.flat):
        if hue_order is not None:
            name = hue_order[int(name)]
        ax.set(xticklabels=[], yticklabels=[], visible=True)
        ax.grid(color='grey')
        ax.set_facecolor('lightskyblue') #'lavender')
        ax.text(0.0,0.9, f'{name}', transform=ax.transAxes)
        scat=ax.scatter(*translate_coords(dfx), s=30, c=dfx.log_eflux,
                vmin=-12.5, vmax=-10,
                )
    for ax in axx.flat[n:]: ax.set_visible(False)
    if cbar:
        h = 0.6/nrows
        cb=plt.colorbar(scat, fig.add_subplot(position=[0.92, 0.85-h, 0.015, h]) )  
        cb.set_label('log eflux', fontsize=10)
        cb.set_ticks([-12, -11, -10],)
        cb.ax.tick_params(labelsize=10)
    return fig

class SpecFunLookup:
    """dict allowing lookup of the uw spectral function name using 4FGL name

    df is indexed by fshas uw_name
    """
    def __init__(self, df):
        with capture_hide():
            from utilities.catalogs import UWcat
            uwcat = UWcat('uw1410')

        self.d1 = dict(zip(df.index, df.uw_name));
        self.d2 = dict(zip(uwcat.jname, uwcat.specfunc))

    def __getitem__(self, catname):    
        return self.d2[self.d1[catname]]
    
    def get(self):
        return pd.Series(dict([ (idx, self[idx]) for idx in self.df.index]))



def show_unique(sclass):
    v,n = np.unique(sclass,  return_counts=True)
    show(
        pd.DataFrame.from_dict(
            dict(list(zip(v,n))), orient='index',
        )
     )

class SEDplotter:
    """ Manage SED plots showing both UW and 4FGL-DR4 

    """
    def __init__(self, plec=False):
        """ set up the catalogs that have specfunc columns"""
        from utilities.catalogs import UWcat, Fermi4FGL
        self.uw = UWcat('uw1410')
        self.uw.index = self.uw.jname  # make it indexsed by the uw jname
        self.uw.index.name = 'UW jname'
        self.fcat = Fermi4FGL()
        self.plec =plec
        self.plot_kw=[dict(lw=4,color='blue', alpha=0.5),dict(color='red')]
        
    def fgl_plec(self, name):
        from utilities.catalogs import PLSuperExpCutoff4
        fgl_names =list(self.fcat.index) 
        row = self.fcat.data[fgl_names.index(name)]
        pars = [row[par] for par in 'PLEC_Flux_Density PLEC_IndexS PLEC_ExpfactorS PLEC_Exp_Index'.split()]
        return  PLSuperExpCutoff4(pars, e0=row['Pivot_Energy'])

    def funcs(self, src):
        """ src is a Series object with index the 4FGL name, a "uw_name" entry
        return the spectral functions"""
        try:
            uwf = self.uw.loc[src.uw_name,'specfunc']
        except:
            uwf = None

        # fcatf =  self.fcat.loc[src.name,'specfunc']
        name = src.name
        if name not in self.fcat.index: name+='c' # cloud ??
        fcatf = self.fgl_plec(name) if self.plec else \
                self.fcat.loc[name,'specfunc']
            
        # except:
        #     fcatf = None
        return uwf, fcatf
        
    def plots(self, src, ax=None, **kwargs):
        fig, ax = plt.subplots(figsize=(2,2)) if ax is None else  (ax.figure, ax)
        kw = dict(xlabel='', ylabel=''); kw.update(kwargs)

        for f, pkw in zip(self.funcs(src), self.plot_kw):
            try:
                if callable(f):   f.sed_plot(ax, plot_kw=pkw)
            except:
                ax.text(0.4,0.5, 'Failed', color='red', transform=ax.transAxes ) 
        ax.set(**kw)


def sedplotgrid(df, ncols=10, height=1, fignum=None, **kwargs):
    """ - height -- height of a row in inches """
    N = len(df)
    assert N>0, 'No data'
    nrows = (N-1)//ncols +1
    figsize= (ncols*height+0.5, nrows*height+0.5)
    def fmt_info(info, sep='\n  '):
        with pd.option_context('display.precision', 3):
            t = str(info).split('\n')[:-1]
        return info.name +sep+ sep.join(t)
    with capture_hide():
        sp = SEDplotter(plec=kwargs.pop('plec', False))
    fig, axx = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize,
                    sharex=True,sharey=True,
                    gridspec_kw=dict(top =0.99, bottom=0.01, hspace=0.1,
                                        left=0.01, right =0.99, wspace=0.1,))
    kw = dict(xticks=[], yticks=[], xlabel='', ylabel='', ylim=(0.02,10))    
    kw.update(kwargs)
    tt=[]
    for ax, (name,info) in zip(axx.flat, df.iterrows()):
        tt.append(fmt_info(info))
        sp.plots(info, ax=ax)
        ax.set(**kw)
    
    show(fig, 
         fignum=fignum,
         tooltips=tt, 
         caption=f"""SED plots. Scales for the x and y axes are {ax.get_xlim()} GeV and 
         {ax.get_ylim()} eV cm-2 s-1. uw1410 in blue, 4FGL-DR4 in red.""")
    

def show_sed_plots(df, fignum=None, ncols=15, height=0.5):
    df = df.copy()
    df['Fp'] = 10**df.log_fpeak
    df['Ep'] = 10**df.log_epeak
    cols = 'ts r95 glat glon Fp Ep curvature class1 nbb sgu uw_name'.split()
    sedplotgrid(df[cols], fignum=None, ncols=ncols, height=height)
    
def counts(df, hue, name='counts'):
    t = df.groupby(hue).size()
    t.name=name
    return t
def translate_coords(*args):
    """ 
    Helper for aitoff projection: Translate degrees to radians (cleaner if aitoff did it)
    Expect first arg or args to be:
    * a SkyCoord object (perhaps a list of positions)
    - or -
    *  lists of l, b in degrees
    - or -
    * a DataFrame with glon and glat columns
    """
    nargs = len(args)
    if nargs>0 and isinstance(args[0], SkyCoord):
        sc = args[0].galactic
        l, b = sc.l.deg, sc.b.deg
        rest = args[1:]
    elif nargs>0 and isinstance(args[0], pd.DataFrame):
        df = args[0]
        sc = SkyCoord(df.glon, df.glat, unit='deg', frame='galactic')
        l, b = sc.l.deg, sc.b.deg
        rest = args[1:]
    elif nargs>1:
        l, b = args[:2]
        rest = args[2:]
    else:
        raise ValueError('Expect positional parameters l,b, or skycoord or DataFrame with glon glat')

    # convert to radians 
    x  = -np.radians(np.atleast_1d(l))
    x[x<-np.pi] += 2*np.pi # equivalent to mod(l+pi,2pi)-pi I think
    y = np.radians(np.atleast_1d(b))
    return [x,y] + list(rest) 

    