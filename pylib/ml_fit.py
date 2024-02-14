import sys
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.catalogs import Fermi4FGL
from pylib.tools import FigNum, show_date, update_legend, epeak_kw, fpeak_kw
from pylib.scikit_learn import SKlearn

if 'show' in sys.argv:
    from pylib.ipynb_docgen import show, show_fig, capture_show

sns.set_theme('notebook' if 'talk' not in sys.argv else 'talk', font_scale=1.25) 
if 'dark' in sys.argv:
    plt.style.use('dark_background') #s
    plt.rcParams['grid.color']='0.5'
    dark_mode=True
else:
    dark_mode=False
fontsize = plt.rcParams["font.size"] # needed to be persistent??

dataset='dr3' if 'dr3' in sys.argv else 'dr4'

#=======================================================================
class MLfit(SKlearn):

    def __init__(self, skprop:dict, 
                 fgl:str=dataset,
    ):
        self.df, self.cat_len = self.load_data(fgl)
        super().__init__(self.df, skprop)
        self.palette =['cyan', 'magenta', 'yellow'] if dark_mode else 'green red blue'.split()
        self.cache_file = f'files/{dataset}_{len(self.target_names)}_class_classification.csv'
        
        self.hue_kw={}
        self.size_kw={}
        if dark_mode:
            self.hue_kw.update(palette='yellow magenta cyan'.split(), edgecolor=None)
        else:
            self.hue_kw.update(palette='green red blue'.split())   
    
    def __repr__(self):
        return f"""MLfit applied to 4FGL-{dataset.upper()} \n\n{super().__repr__()}
        """
    
    def load_data(self, fgl):
        """Extract subset of 4FGL information relevant to ML pulsar-like selection
        """

        cols = 'glat glon significance r95 variability class1'.split()
        fgl = Fermi4FGL(fgl)
        
        # remove sources without variability or r95
        df = fgl.loc[:,cols][(fgl.variability>0) & (fgl.r95>0) &(fgl.significance>4)].copy()
        print(f"""Remove {len(fgl)-len(df)} without valid r95 or variability or significance>4
            -> {len(df)} remain""")

        # extract LP spectral function, get its parameters
        df['lp_spec'] = [fgl.get_specfunc(name, 'LP') for name in df.index]
        sed = df['lp_spec']
        df['Ep'] = 10**sed.apply(lambda f: f.epeak)
        df['Fp'] = 10**sed.apply(lambda f: f.fpeak)
        df['d'] = sed.apply(lambda f: f.curvature()).clip(-0.1,2)

        # lower-case class1, combine 'unk' and ''  associations to 'unid'
        def reclassify(class1):            
            cl = class1.lower()
            return 'unid' if cl in ('unk', '') else cl
     
        df['association'] = df.class1.apply(reclassify)
        # append columns with logs needed later
        df['log_var'] = np.log10(df.variability)
        df['log_epeak'] = np.log10(df.Ep)
        df['log_fpeak'] = np.log10(df.Fp)  

        return df,len(fgl)
        
    def prediction_association_table(self):
        """The number of sources classified as each of the  targets, according to the 
        association type.
        """
        df = self.df.copy()
        
        # combine a bunch of the class1 guys into "other"
        def make_other(s):
            if s in 'bll fsrq psr msp bcu spp glc unid'.split():
                return s
            return 'other'
            
        df.loc[:,'association'] = df.association.apply(make_other).values
        def simple_pivot(df, x='prediction', y= 'association'):        
            ret =df.groupby([x,y]).size().reset_index().pivot(
                columns=x, index=y, values=0)
            return ret.reindex(index='bll fsrq psr msp glc bcu spp other unid'.split())
            
        t=simple_pivot(df)
        t[np.isnan(t)]=0
        return t.astype(int)
    
    def confusion_matrix(self):
        """return a confusion matrix"""
        from sklearn import metrics
        # get true and predicted for the training data
        df = self.df
        tp = df.loc[~pd.isna(df[self.target_field]), 'target prediction'.split()].to_numpy().T
        
 
        labels=self.target_names
        cm = metrics.confusion_matrix(tp[0], tp[1],)
        n = self.target_counts['pulsar']
        purity = 1- cm[0,1]/n
        efficiency = cm[1,1]/n
        
        cmdf=pd.DataFrame(cm,
                        columns=pd.Series(labels,name='prediction'), 
                        index=pd.Series(labels,name='associations'), )
        return cmdf
    
    def predict_prob(self, query='association=="unid"'):
        """Return DF with fit probabilities
        """
        mdl = self.classifier
        assert mdl.probability, 'Fit must be with probability  True' 
        dfq = self.df.query(query) if query is not None else self.df
        X = dfq.loc[:, self.features]
        return pd.DataFrame(mdl.predict_proba(X), index=dfq.index,
                            columns=['p_'+ n for n in self.target_names])
    
    def plot_prediction_association(self, table):
        """Bar chart showing the prediction counts for each association type, according 
        to the predicted target class. 
        """
    
        fig, ax = plt.subplots(figsize=(10,5))
        ax = table.plot.barh(stacked=True, ax=ax, color=self.palette)
        ax.invert_yaxis()
        ax.set(xlabel='Prediction counts', ylabel='Association type')
        ax.legend(bbox_to_anchor=(0.78,0.75), loc='lower left', frameon=True,
                title='Target class')
        return fig
    
    def pairplot(self, query='', **kwargs):  
        """Corner plots, showing KDE distributions of each of the target populations for each of
        the feature parameters.
        """      
        if query=='':
            # default: only targets
            df = self.df.loc[~ pd.isna(self.df[self.target_field])]
        else:
            df = self.df.query(query)
        kw = dict(kind='kde', hue=self.target_field, hue_order=self.target_names, height=2, corner=True)
        kw.update(**kwargs)
        g = sns.pairplot(df, vars=self.features,  palette=self.palette[:len(self.target_names)], **kw,)
        return g.figure
    
    def ep_vs_d(self, df=None):
        # from pylib.tools import epeak_kw
        if df is None: df=self.df
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 5),sharex=True,sharey=True,
                                    gridspec_kw=dict(wspace=0.05))
        kw = dict( x='log_epeak', y='d',  palette=self.palette[:len(self.targets)],
                  hue_order=self.target_names)
        
        sns.kdeplot(df,ax=ax1, hue=self.target_field, **kw)
        sns.kdeplot(df.query('association=="unid"'),ax=ax2, hue='prediction',**kw)

        ax1.set(**epeak_kw(),ylabel='curvature $d$', xlim=(-1.5,1.5)); ax2.set(**epeak_kw())
        return fig

    def pulsar_prob_hists(self):
        """Histograms of the classifier pulsar probability. 
        Upper three plots: each of the labeled target clases; lower plots: stacked histograms
        for the classifier assigments, colors corresponding to the plots above.
        """
        probs= self.predict_prob(query=None)
        df = pd.concat([self.df, probs], axis=1)
        titles = ['psr+msp','bll','fsrq','unid']
        non_targets = ['unid', 'bcu']
        titles = list(self.target_names) + non_targets
        fig, axx = plt.subplots(nrows=len(titles), figsize=(8,10),sharex=True,
                                    gridspec_kw=dict(hspace=0.05))
        kw = dict( x='p_pulsar',  bins=np.arange(0,1.01, 0.025), log_scale=(False,True),
                    element='step', multiple='stack', legend=False)

        for ax, hue_order, color in zip(axx, self.target_names, self.palette):        
            sns.histplot(df, ax=ax,  hue='target',hue_order=[hue_order],
                        palette=[color],**kw)            

        for i,nt in enumerate(non_targets):
            sns.histplot(df.query(f'association=="{nt}"'), ax=axx[i+3], hue='prediction', 
                    hue_order=self.target_names,  palette=self.palette,**kw);
        
        axx[-1].set(xlabel='Pulsar probability')
        for ax, title in zip(axx, titles):
            ax.text( 0.5, 0.85, title, ha='center', transform=ax.transAxes, fontsize=14)
        return fig
    
    def plot_spectral(self, df):
        """Spectral shape scatter plots: curvature $d$ vs $E_p$.
        """

        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,6), sharex=True, sharey=True,
                                    gridspec_kw=dict(wspace=0.08))
        kw =dict(x='log_epeak', y='d', s=10, hue='subset', edgecolor='none' )
        sns.scatterplot(df, ax=ax1, hue_order=self.target_names,  **kw,
                        palette=self.palette, );
        ax1.set(**epeak_kw(), yticks=np.arange(0.5,2.1,0.5), ylabel='Curvature ${d}$',
            xlim=(-1,1),ylim=(0.1,2.2));
        sns.scatterplot(df, ax=ax2, **kw, hue_order=['unid'], palette=['0.5'], )
        
        sns.kdeplot(df, ax=ax2, **kw, hue_order=self.target_names, palette=self.palette, alpha=0.4,legend=False );
        ax2.set(**epeak_kw(),)# yticks=np.arange(0.5,2.1,0.5));
        return fig

    def plot_flux_shape(self, df):
        """Scatter plots of peak flux $Fp$ vs. diffuse flux.
        """
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,6), sharex=True, sharey=True,
                                    gridspec_kw=dict(wspace=0.08))
        kw =dict(y='log_fpeak', x='diffuse', s=10, hue='subset', edgecolor='none' )
        sns.scatterplot(df, ax=ax1, hue_order=self.target_names,  **kw,
                        palette=self.palette, );
        sns.scatterplot(df, ax=ax2, **kw, hue_order=['unid'], palette=['0.5'], )
        
        sns.kdeplot(df, ax=ax2, **kw, hue_order=self.target_names, palette=self.palette, alpha=0.4,legend=False );
        ax1.set(ylim=(-2.75,4), **fpeak_kw('y'), xlim=(-1,2.5))
        return fig
    
    
    def write_summary(self, overwrite=False):

        summary_file=f'files/{dataset}_{len(self.target_names)}_class_classification.csv'  

        if Path(summary_file).is_file() and not overwrite:
            print(f'File `{summary_file}` exists--not overwriting.')
            return
        
        def set_pulsar_type(s):
            if s.association in self.psr_names: return s.association
            if s.prediction=='pulsar': return s.association+'-pulsar'
            return np.nan
        def set_source_type(s):
            return s.association if s.association in self.targets['pulsar'] else \
                s.association+'-'+s.prediction

        def get_diffuse(df):
            from astropy.coordinates import SkyCoord
            from pylib.diffuse import Diffuse
            diff = Diffuse()
            sdirs = SkyCoord(df.glon, df.glat, unit='deg', frame='galactic')
            return  diff.get_values_at(sdirs)
            
        df = self.df.copy()

        # add the three predicted probabilities
        # df['p_pulsar'] = self.predict_prob(query=None).loc[:,'p_pulsar']
        df = pd.concat([df,self.predict_prob(query=None)], axis=1)
        
        # set source_type for pulsars and pulsar-predictions
        df['source_type'] = df.apply(set_source_type, axis=1)
        # df = df[df.source_type.notna()]
            

        # add a diffuse column
        df['diffuse'] = get_diffuse(df) #self.get_diffuses(df)

        cols= 'source_type glat glon significance r95 Ep Fp d diffuse p_pulsar'.split()
        df.loc[:, cols].to_csv(summary_file, float_format='%.3f') 
        print(f'Wrote {len(df)}-record summary, using model {self.model}, to `{summary_file}` \n  columns: {cols}')





    #Takes a tuple of classifiers and a tuple of their names
    # Axis object needs working on
    def pltAvgPrecRec(classifiers, names, X, y, Axis=None):
    
        import seaborn as sns
        import sklearn
        from sklearn.metrics import PrecisionRecallDisplay
        import matplotlib.pyplot as plt
    
        def getPrecRec(theX, they, clf, ax=None):
            X_train, X_test, y_train, y_test = train_test_split(theX, they, test_size=0.333)
    
            classifier = clf.fit(X_train, y_train)
    
            #Get precision and recall values
            tem = PrecisionRecallDisplay.from_estimator(
                classifier, X_test, y_test, name=name, ax=ax, plot_chance_level=True
            )
            tem.figure_.clear()
    
            return tem
    
        #set up
        thePrec = np.empty(0)
        theRecall = np.empty(0)
        theSet = np.empty(0)
        
        
        #Loop through classifiers
        for name, clf in zip(names, classifiers):
        
            they = (y=='psr')
    
            pr = getPrecRec(X, they, clf, Axis)
    
            count = 0
            prec = pr.precision
            recall = pr.recall
    
    
            while((count:=count+1) <= 20):
    
                pr = getPrecRec(X, they, clf, Axis)
    
                if prec.size < pr.precision.size:
                    prec += pr.precision[:prec.size]
                    recall += pr.recall[:recall.size]
                else:
                    p = np.ones(prec.size)
                    r = np.ones(recall.size)
    
                    p = prec/count
                    r = recall/count
    
                    p[:pr.precision.size] = pr.precision
                    r[:pr.recall.size] = pr.recall
                    prec += p
                    recall += r
    
    
    
            theSet = np.concatenate((theSet, np.full((prec.size), name)))
            thePrec = np.concatenate((thePrec, prec/count))
            theRecall = np.concatenate((theRecall, recall/count))
    
        prdf = pd.DataFrame(data={"prec": thePrec, 
                                  "recall": theRecall, 
                                  "group": theSet})
    
        plot = sns.lineplot(data=prdf, x="recall", y="prec", hue='group')
        
        return plot
        

def doc(nc=2, np=2, kde=False, ):
    from pylib.tools import FigNum, show_date

    def targets(nc, np=2):
        if np==1: 
            a = dict(psr=('psr',), msp=('msp',))
        else:
            a = dict( pulsar = ('psr', 'msp',) if np==2 else ('psr','msp','glc')) 
        b = dict( blazar=('bll', 'fsrq')) if nc==2 else \
            dict(bll=('bll',), fsrq=('fsrq',))
        return dict(**a,**b)
    
    skprop = dict(
        features= ('log_var', 'log_fpeak', 'log_epeak', 'd'),
        targets = targets(nc,np),
        model_name = 'SVC',
        truth_field='association',
        # will be set to a targets key
        target_field = 'target',
        )
    fn = FigNum(n=1, dn=0.1)
    if kde:
        show(f"""# KDE approach  ({dataset.upper()})""")
        show_date()
        show("""Not applying ML, so no fit to targets to generate prediction model. Instead we compute KDE probability density distributions
        for the ML targets, which we then apply to the unid and bcu associations.
        """)
    else:
        show(f"""# ML {nc}-class Classification  ({dataset.upper()})""")

    with capture_show('Setup:') as imp:
        self = MLfit(skprop)
    show(imp)
    if kde: 
        return self
    show(str(self))
    show(f"""## Feature distributions """)
    show_fig(self.pairplot, fignum=fn.next)

    show(f"""## Train then apply prediction """)
    self.train_predict()
    df=self.df

    show(f"""### The confusion matrix""")
    cmdf = self.confusion_matrix()
    show(cmdf)
    
    N = self.target_counts['pulsar']
    TP = cmdf.iloc[-1,-1]
    FP = cmdf.sum(axis=1)[-1] - TP
    FN = cmdf.sum(axis=0)[-1] - TP
    efficiency, purity = TP/N, TP/(FN+TP)
    show(f'\n{purity=:.2f}, {efficiency=:.2f}' )

    show(f"""### All predictions""")
    table = self.prediction_association_table()
    show(table)
    show_fig(self.plot_prediction_association,table, fignum=fn.next)
    show(f"""#### Write summary file""")
    self.write_summary()
    return self

if 'doc' in sys.argv:
    self = doc()

def kde_setup(kde_vars = 'd log_epeak log_fpeak diffuse'.split()):
    self = doc(nc=2, np=1, kde=True)
    def make_group(self):

        def groupit(s):
            if s.association=='unid': return 'unid'
            if s.association=='bcu': return 'bcu'
            if ~ pd.isna(s.target): return s.target
            return np.nan
        df = self.df
        df['subset'] = df.apply(groupit, axis=1)
    make_group(self)
    cut = '0.15<Ep<4 & d>0.2 & variability<25'
    show(f'### Data selection cut: "{cut}"')
    dfc = self.df.query(cut).copy()
    all = pd.Series(self.df.groupby('subset').size(), name='all')
    sel = pd.Series(dfc.groupby('subset').size(), name='selected')
    pct = pd.Series((100*sel/all).round(0).astype(int), name='%')
    t =pd.DataFrame([all,sel,pct])['blazar psr msp unid bcu'.split()]; 
    t.index.name='counts'
    show(t)

    
    # apply diffuse
    df3 = pd.read_csv(f'files/{dataset}_2_class_classification.csv', index_col=0)
    dfc['diffuse'] = df3.diffuse
    def apply_kde(self, df=None, features=None):
        from pylib.kde import Gaussian_kde
        if df is None: df = self.df.copy() 
        if features is None: features=self.features
        for name, sdf in df.groupby('subset'):
            gde = Gaussian_kde(sdf,  features)
            u = gde(df)
            df[name+'_kde'] = u
        return df
    
    show(f"""## Create KDE functions 
    Using variables {kde_vars} for KDE analysis""")
    apply_kde(self, dfc, kde_vars)
    return self, dfc

def apply_kde(self, df=None, features=None):
    from pylib.kde import Gaussian_kde
    if df is None: df = self.df.copy() 
    if features is None: features=self.features
    for name, sdf in df.groupby('subset'):
        gde = Gaussian_kde(sdf,  features)
        u = gde(df)
        df[name+'_kde'] = u
    return df

if 'kde' in sys.argv:
    import warnings
    warnings.filterwarnings("ignore")

    self = doc(nc=2, np=1, kde=True)
    def intro():
        show("""\
            ## Introduction
            The classification schemes that we have been using are misleading for the Unid's. 
            The following features are problematic for classifying the Unids.
            * Flux, or $Fp$: the Unids are mostly low flux, a threshold effect for known sources but apparently 
            a feature of the possible new component.
            * Curvature: Unid curvatures are much higher, either science, or as Jean believes, a systematic 
            for curved weak sources (which option can be resolved with a gtobssim study.)
            * Softness, or $Ep$: the Unids have a large soft component, not reflected in the target populations
            The fact there is an Unid component not in the targets violates the basic ML assumption, 
            as is evident in the pulsar probability plots that I posted above. Using the pulsar prediction, 
            assuming that it was also selecting this new population, underestimates this component.
            Thus we use a simpler approach: calculate KDE probabilities for the three targets using 
            only the two spectral shape parameters curvature and Ep  (after removing those with significant 
            variability) . I've been applying this to the "pulsar" selection to isolate 
            the `msp` and `psr` subsets.
            This only presumes that the shape is independent of the flux.
            """)
    intro()
    def make_group(self):
        df = self.df
        def groupit(s):
            if s.association=='unid': return 'unid'
            if ~ pd.isna(s.target): return s.target
            return np.nan
        df = self.df
        df['subset'] = df.apply(groupit, axis=1)
    make_group(self)
    cut = '0.15<Ep<4 & d>0.2 & variability<25'
    show(f'### Data selection cut: "{cut}"')
    dfc = self.df.query(cut)
    all = pd.Series(self.df.groupby('subset').size(), name='all')
    sel = pd.Series(dfc.groupby('subset').size(), name='selected')
    show(pd.DataFrame([all,sel]))
    # apply diffuse
    df3 = pd.read_csv(f'files/dr3_2_class_classification.csv', index_col=0)
    dfc['diffuse'] = df3.diffuse
    
    show(f"""### Spectral shape plots""")    
    show_fig(self.plot_spectral, dfc)

    show(f"""### Flux plots""")    
    show_fig(self.plot_flux_shape, dfc)

def plot_kde_density(self,df, **kwargs):

    fig, axx = plt.subplots(ncols=2, nrows=2,  figsize=(15,12),  
                                    gridspec_kw=dict(wspace=0.3,hspace=0.05))
    other_kw   =  dict(hue_order=['unid'],          palette=['0.5'],)
    other_kw.update(kwargs)
    kde_kw   = dict(hue_order=self.target_names, palette=self.palette,  alpha=0.4,legend=False)
    target_kw= dict(hue_order=self.target_names, palette=self.palette,)

    spectral_kw = dict(data=df, x='log_epeak', y='d', s=10, hue='subset', edgecolor='none' )
    flux_kw = dict(data=df, y='log_fpeak', x='diffuse', s=10, hue='subset', edgecolor='none' )
    
    def spectral( ax1, ax2 ):
        """Spectral shape scatter plots: curvature $d$ vs $E_p$.
        """            
        sns.scatterplot(ax=ax1, **spectral_kw, **target_kw)        
        sns.scatterplot(ax=ax2, **spectral_kw, **other_kw)
        sns.kdeplot(    ax=ax2, **spectral_kw, **kde_kw) 

        for ax in (ax1,ax2):
            ax.set(**epeak_kw(), xlim=(-1,1), ylim=(0.1,2.2),
                   yticks=np.arange(0.5,2.1,0.5), ylabel='Curvature ${d}$' )
        ax1.set(xticklabels=[])

    def flux( ax1, ax2):
        """Scatter plots of peak flux $Fp$ vs. diffuse flux.
        """
        sns.scatterplot(ax=ax1, **flux_kw, **target_kw)        
        sns.scatterplot(ax=ax2, **flux_kw, **other_kw)
        sns.kdeplot(    ax=ax2, **flux_kw, **kde_kw)

        for ax in (ax1,ax2):
            ax.set(ylim=(-2.75,4), **fpeak_kw('y'), xlim=(-1,2.5))
        ax1.set(xticklabels=[])

    spectral( *axx[:,0])
    flux(     *axx[:,1])
    return fig

if 'bcu' in sys.argv:
    import warnings
    warnings.filterwarnings("ignore")
    self, dfc = kde_setup()
