import sys
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.catalogs import Fermi4FGL
from pylib.tools import epeak_kw, fpeak_kw, set_theme
from pylib.scikit_learn import SKlearn

dark_mode = set_theme(sys.argv)
if 'show' in sys.argv:
    from pylib.ipynb_docgen import show, show_fig, capture_show

dataset='dr3' if 'dr3' in sys.argv else 'dr4'

def lp_pars(fgl):
    """ extract LP spectral functions from 4FGL catalog object get its parameters
    Return DataFrame with EP,FP,d,d_unc
    """

    df = pd.DataFrame(index=fgl.index)
    df['lp_spec'] = [fgl.get_specfunc(name, 'LP') for name in df.index]
    sed = df['lp_spec']
    df['Ep'] = 10**sed.apply(lambda f: f.epeak)
    df['Fp'] = 10**sed.apply(lambda f: f.fpeak)
    df['d'] = sed.apply(lambda f: f.curvature()).clip(-0.1,2)
    df['d_unc'] = 2*fgl.field('unc_LP_beta')
    return df.drop(columns=['lp_spec'])

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

          # create association with lower-case class1, combine 'unk' and '' to 'unid'
        def reclassify(class1):            
            cl = class1.lower()
            return 'unid' if cl in ('unk', '') else cl
   
        df['association'] = df.class1.apply(reclassify)

        # append LP columns from  catalog
        df = pd.concat([df, lp_pars(fgl).loc[df.index]], axis=1)
        if dataset=='dr4':
            # replace with common DR3 if DR4
            dr3 = Fermi4FGL('dr3')
            lp3 = lp_pars(dr3)
            df['hasdr3'] =np.isin(df.index, dr3.index)
            common = df[df.hasdr3].index
            print(f'Apply DR3 LP values for all but {sum(~df.hasdr3)} sources.')
            lpcols='Ep Fp d d_unc'.split()
            df.loc[common,lpcols]=  lp3.loc[common,lpcols]

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

title = sys.argv[-1] if 'title' in sys.argv else None


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



def doc(nc=2, np=2, kde=False, bcu=False ):
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
        show(f"""# KDE approach  ({dataset.upper()})""" if title is None else '# '+title)
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
    
    
    #import 'default' classifiers
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    theNames=('Neural Net',
              'RBF SVM',
              'Random Forest')

    theClassifiers=(MLPClassifier(alpha=1, max_iter=1000),
                 SVC(gamma=2, C=1,probability=True), 
                 RandomForestClassifier(n_estimators=100, max_features=2))
    
    ax = plt.axes()
    
    prp = self.pltAvgPrecRec(theClassifiers, theNames, Axis=ax)
    show(prp)    
    
    self.write_summary()
    return self

if 'doc' in sys.argv:
    self = doc()

def kde_setup(kde_vars = 'd log_epeak diffuse'.split(), nc=2, bcu=False):
    self = doc(nc=nc, np=1, kde=True, bcu=bcu)
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
    all = pd.Series(self.df.groupby('subset').size(), name='total')
    sel = pd.Series(dfc.groupby('subset').size(), name='selected')
    pct = pd.Series((100*sel/all).round(0).astype(int), name='%')
    classes = 'blazar psr msp unid'.split()
    if bcu: classes = classes + ['bcu']
    t =pd.DataFrame([all,sel,pct])[classes]; 
    # t.index.name='counts'
    show(t)
    # apply diffuse
    df3 = pd.read_csv(f'files/{dataset}_{nc}_class_classification.csv', index_col=0)
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
    
    show(f"""## Create KDE functions  in lieu of ML training
    * Features: {kde_vars} 
    * Targets: {self.targets.keys()}
    """)
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

def hist_kde(self,df, ):
    """Histograms of the blazar KDE values...
    """
    data = df.copy()
    def hist_blazar_kde( ax):
        data['blazar class']= data.association
        sns.histplot(data, ax=ax, x='blazar_kde', hue='blazar class',
                     hue_order='bll fsrq bcu'.split(), palette=self.palette,
                     element='step')
        ax.set(xticks=np.arange(0,1.51, 0.5), xlabel='blazar KDE')
     
    def hist_pulsar_kde( ax):
        data['pulsar class']= data.association
        sns.histplot(data, ax=ax, x='blazar_kde', hue='pulsar class',
                     hue_order='psr msp'.split(), palette=self.palette,
                     element='step')
        ax.set(xticks=np.arange(0,1.51, 0.5), xlabel='blazar KDE')
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,4)) 
    hist_blazar_kde( ax=ax1)
    hist_pulsar_kde( ax=ax2)
    return fig

if 'kde' in sys.argv:
    self, dfc = kde_setup()

if 'kdedoc' in sys.argv:
    import warnings
    warnings.filterwarnings("ignore")
    self, dfc = kde_setup()
    
  
    show(f"""### Blazar KDE for blazar and pulsar types
    The KDE for blazars is determined from the `bll` and `fsrq` classes. Here we 
    examine distributions
    of it for the blazar types, including also `bcu`, and the pulsar types.
    """)    
    show_fig(hist_kde, self, dfc, )
    show(f"""Note that the `bcu` has a component that does not correspond to that
    expected for the known blazar types. The pulsars show show small 
    values of the blazar KDE, but there is a little mixing for `msp`. 
    """)

    show(f"""### Spectral shape plots""")    
    show_fig(self.plot_spectral, dfc)

    show(f"""### Flux plots""")    
    show_fig(self.plot_flux_shape, dfc)

def plot_kde_density(self,df, s=10, **kwargs):
    """Scatter plots showing the associated sources in the upper row, and the UnID selection in lower row.
    Left column: Curvature $d$ vs $E_p$; Right column: Source peak flux $F_p$ vs. diffuse background flux.
    """

    fig, axx = plt.subplots(ncols=2, nrows=2,  figsize=(15,12),  
                                    gridspec_kw=dict(wspace=0.3,hspace=0.05))
    other_kw   =  dict(hue_order=['unid'],    palette=['0.5'],)
    other_kw.update(kwargs)
    kde_kw   = dict(hue_order=self.target_names, palette=self.palette,  alpha=0.4,legend=False)
    target_kw= dict(hue_order=self.target_names, palette=self.palette,)

    spectral_kw = dict(data=df, x='log_epeak', y='d',  hue='subset', s=s, edgecolor='none' )
    flux_kw = dict(data=df, y='log_fpeak', x='diffuse',  hue='subset',s=s, edgecolor='none' )
    def fix_for_kde(kw): # so not to trigger warning
        kw.pop('s'); kw.pop('edgecolor')
        return kw
    def spectral( ax1, ax2 ):
        """Spectral shape scatter plots: curvature $d$ vs $E_p$.
        """            
        sns.scatterplot(ax=ax1, **spectral_kw, **target_kw)        
        sns.scatterplot(ax=ax2, **spectral_kw, **other_kw)
        sns.kdeplot(    ax=ax2, **fix_for_kde(spectral_kw), **kde_kw) 

        for ax in (ax1,ax2):
            ax.set(**epeak_kw(), xlim=(-1,1), ylim=(0.1,2.2),
                   yticks=np.arange(0.5,2.1,0.5), ylabel='Curvature ${d}$' )
        ax1.set(xticklabels=[])

    def flux( ax1, ax2):
        """Scatter plots of peak flux $Fp$ vs. diffuse flux.
        """
        sns.scatterplot(ax=ax1, **flux_kw, **target_kw)        
        sns.scatterplot(ax=ax2, **flux_kw, **other_kw)
        sns.kdeplot(    ax=ax2, **fix_for_kde(flux_kw), **kde_kw)

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

