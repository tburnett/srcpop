import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.catalogs import Fermi4FGL
from pylib.tools import epeak_kw, fpeak_kw, set_theme, diffuse_kw
from pylib.scikit_learn import SKlearn

dark_mode = set_theme(sys.argv)
if 'show' in sys.argv:
    from pylib.ipynb_docgen import show, show_fig, capture_show, capture_hide

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

#  create association with lower-case class1, combine 'unk','bcu', and '' to 'unid'
def reclassify(class1):            
    cl = class1.lower()
    return 'unid' if cl in ('unk','bcu', '') else cl

#=======================================================================
class MLfit(SKlearn):

    def __init__(self, skprop:dict, 
                 fgl:str=dataset,
    ):
        self.df, self.cat_len = self.load_data(fgl)
        super().__init__(self.df, skprop)
        self.palette =['cyan', 'magenta', 'yellow'] if dark_mode else 'green red blue'.split()
        self.cache_file = f'files/{dataset}_{len(self.trainer_names)}_class_classification.csv'
        
        self.hue_kw={}
        self.size_kw={}
        if dark_mode:
            self.hue_kw.update(palette='yellow magenta cyan'.split(), edgecolor=None)
        else:
            self.hue_kw.update(palette='violet blue red'.split())   
    
    def __repr__(self):
        return f"""MLfit applied to 4FGL-{dataset.upper()} \n\n{super().__repr__()}
        """
    
    def load_data(self, fgl):
        """Extract subset of 4FGL information relevant to ML pulsar-like selection
        """

        cols = 'glat glon significance r95 variability class1'.split()
        fgl = Fermi4FGL(fgl)
        
        # remove sources without variability or r95 but keep SNR
        df = fgl.loc[:,cols][  (fgl.variability>0) 
                             & ( (fgl.r95>0) | (fgl.class1=='SNR')) 
                             & (fgl.significance>4)
                             ].copy()
        print(f"""Remove {len(fgl)-len(df)} without valid r95 or variability or significance>4
            -> {len(df)} remain""")

        # zero the SNR r95 values
        df['r95'] = df.r95.apply(lambda x: 0 if pd.isna(x) else x)
 
        ### move out
        # # create association with lower-case class1, combine 'unk' and '' to 'unid'
        # def reclassify(class1):            
        #     cl = class1.lower()
        #     return 'unid' if cl in ('unk', '') else cl
   
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

        # add sqrt_d
        df['sqrt_d'] = np.sqrt(df.d.clip(0,2))

        # append columns with logs needed later
        df['log_var'] = np.log10(df.variability)
        df['log_epeak'] = np.log10(df.Ep)
        df['log_fpeak'] = np.log10(df.Fp)  

        return df,len(fgl)
        
    def prediction_association_table(self, df=None):
        """The number of sources classified as each of the  trainers, according to the 
        association type.
        """
        if df is None: df = self.df.copy()
        
        # combine a bunch of the class1 guys into "other"
        def make_other(s):
            if s in 'bll fsrq psr msp spp snr rdg glc unid'.split(): # remobe bcu
                return s
            return 'other'
            
        df.loc[:,'association'] = df.association.apply(make_other).values
        def simple_pivot(df, x='prediction', y= 'association'):        
            ret =df.groupby([x,y]).size().reset_index().pivot(
                columns=x, index=y, values=0)
            return ret.reindex(index='bll fsrq psr msp glc spp snr rdg other unid'.split()) ## remove bcu
            
        t=simple_pivot(df)
        t[np.isnan(t)]=0
        return t.astype(int)
    
    def confusion_matrix(self):
        """return a confusion matrix"""
        from sklearn import metrics
        # get true and predicted for the training data
        df = self.df
        tp = df.loc[~pd.isna(df[self.trainer_field]), 'trainer prediction'.split()].to_numpy().T
        
 
        labels=self.trainer_names
        cm = metrics.confusion_matrix(tp[0], tp[1],)
        n = self.trainer_counts['pulsar']
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
                            columns=['p_'+ n for n in self.trainer_names])
    
    def plot_prediction_association(self, table, ax=None):
        """Bar chart showing the prediction counts for each association type, according 
        to the predicted training class.  The association types above the horizontal dashed line
        were combined to form training classes, those below are targets. 
        """

        fig, ax = plt.subplots(figsize=(10,5)) if ax is None else (ax.figure, ax)
        ax = table.plot.barh(stacked=True, ax=ax, color=self.palette)
        ax.invert_yaxis()
        ax.set(xlabel='Prediction counts', ylabel='Association type')
        ax.legend( loc='upper right', #bbox_to_anchor=(0.78,0.75), loc='lower left',
                frameon=True,   title='Training class')
        ax.axhline(3.5, ls='--', color='0.9' if dark_mode else '0.1')
        ax.text(1200, 2.5, 'Used for training',ha='center')
        ax.text(1200, 5.5, 'Targets', ha='center')
        return fig
    
    def pairplot(self, query='', **kwargs):  
        """Corner plots, showing KDE distributions of each of the trainer populations for each of
        the feature parameters.
        """      
        if query=='':
            # default: only trainers
            df = self.df.loc[~ pd.isna(self.df[self.trainer_field])]
        else:
            df = self.df.query(query)
        kw = dict(kind='kde', hue=self.trainer_field, hue_order=self.trainer_names, height=2, corner=True)
        kw.update(**kwargs)
        g = sns.pairplot(df, vars=self.features,  palette=self.palette[:len(self.trainer_names)], **kw,)
        return g.figure
    
    def ep_vs_d(self, df=None):
        # from pylib.tools import epeak_kw
        if df is None: df=self.df
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 5),sharex=True,sharey=True,
                                    gridspec_kw=dict(wspace=0.05))
        kw = dict( x='log_epeak', y='d',  palette=self.palette[:len(self.trainers)],
                  hue_order=self.trainer_names)
        
        sns.kdeplot(df,ax=ax1, hue=self.trainer_field, **kw)
        sns.kdeplot(df.query('association=="unid"'),ax=ax2, hue='prediction',**kw)

        ax1.set(**epeak_kw(),ylabel='curvature $d$', xlim=(-1.5,1.5)); ax2.set(**epeak_kw())
        return fig

    def pulsar_prob_hists(self):
        """Histograms of the classifier pulsar probability. 
        Upper two plots: each of the labeled training classes; lower plot: histogram
        for the classifier assignments, colors corresponding to the plots above.
        """
        probs= self.predict_prob(query=None)
        df = pd.concat([self.df, probs], axis=1)

        non_trainers = ['unid'] #, 'bcu'] 
        titles = list(self.trainer_names) + non_trainers
        fig, axx = plt.subplots(nrows=len(titles), figsize=(8,8),sharex=True, sharey=True,
                                    gridspec_kw=dict(hspace=0.1))
        kw = dict( x='p_pulsar',  bins=np.arange(0,1.01, 0.025), log_scale=(False,True),
                    element='step', multiple='stack', legend=False)

        for ax, hue_order, color in zip(axx, self.trainer_names, self.palette):        
            sns.histplot(df, ax=ax,  hue='trainer',hue_order=[hue_order],
                        palette=[color],**kw)            

        for i,nt in enumerate(non_trainers):
            ntr = len(self.trainers)
            sns.histplot(df.query(f'association=="{nt}"'), ax=axx[i+ntr], hue='prediction', 
                    hue_order=self.trainer_names,  palette=self.palette[:ntr],**kw);
        
        axx[-1].set(xlabel='Pulsar probability')
        for ax, title in zip(axx, titles):
            ax.text( 0.5, 0.75, title, ha='center', transform=ax.transAxes, fontsize='large')
        return fig
    
    def plot_spectral(self, df):
        """Spectral shape scatter plots: curvature $d$ vs $E_p$.
        """

        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,6), sharex=True, sharey=True,
                                    gridspec_kw=dict(wspace=0.08))
        kw =dict(x='log_epeak', y='d', s=10, hue='subset', edgecolor='none' )
        sns.scatterplot(df, ax=ax1, hue_order=self.trainer_names,  **kw,
                        palette=self.palette, );
        ax1.set(**epeak_kw(), yticks=np.arange(0.5,2.1,0.5), ylabel='Curvature ${d}$',
            xlim=(-1,1),ylim=(0.1,2.2));
        sns.scatterplot(df, ax=ax2, **kw, hue_order=['unid'], palette=['0.5'], )
        
        sns.kdeplot(df, ax=ax2, **kw, hue_order=self.trainer_names, palette=self.palette, alpha=0.4,legend=False );
        ax2.set(**epeak_kw(),)# yticks=np.arange(0.5,2.1,0.5));
        return fig

    def plot_flux_shape(self, df):
        """Scatter plots of peak flux $Fp$ vs. diffuse flux.
        """
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,6), sharex=True, sharey=True,
                                    gridspec_kw=dict(wspace=0.08))
        kw =dict(y='log_fpeak', x='diffuse', s=10, hue='subset', edgecolor='none' )
        sns.scatterplot(df, ax=ax1, hue_order=self.trainer_names,  **kw,
                        palette=self.palette, );
        sns.scatterplot(df, ax=ax2, **kw, hue_order=['unid'], palette=['0.5'], )
        
        sns.kdeplot(df, ax=ax2, **kw, hue_order=self.trainer_names, palette=self.palette, alpha=0.4,legend=False );
        ax1.set(ylim=(-2.75,4), **fpeak_kw('y'), xlim=(-1,2.5))
        return fig
    
    def write_summary(self, overwrite=False):

        summary_file=f'files/{dataset}_{len(self.trainer_names)}_class_classification.csv'  

        if Path(summary_file).is_file() and not overwrite:
            print(f'File `{summary_file}` exists--not overwriting.')
            return
        
        def set_pulsar_type(s):
            if s.association in self.psr_names: return s.association
            if s.prediction=='pulsar': return s.association+'-pulsar'
            return np.nan
        def set_source_type(s):
            return s.association if s.association in self.trainers['pulsar'] else \
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

        
        
    def precision_recall(self, names=None, ax=None, test_size=0.33, ncount=10, ):
        """Plot of the precision, or purity, vs the recall, or efficiency,
          for the binary pulsar selection. 
        """
        from pylib.scikit_learn import get_model 

        fig, ax = plt.subplots(figsize=(5,5) ) if ax is None else (ax.figure, ax)
        if names is None:
            names = {self.model_name : str(self.model)}
            
        X,y = self.getXy()
        
        def getPrecRec(theX, they, theclf, ax=None):
            from sklearn.metrics import PrecisionRecallDisplay
            from sklearn.model_selection import train_test_split 
            ty = (they=='pulsar')
            X_train, X_test, y_train, y_test = train_test_split(theX, ty, test_size=test_size)
            classifier = theclf.fit(X_train, y_train)
            #Get precision and recall values
            tem = PrecisionRecallDisplay.from_estimator(
                classifier, X_test, y_test, name=name, ax=None, # plot_chance_level=True
            )
            plt.close(tem.figure_)    
            return tem
        
        #set up
        thePrec = np.empty(0)
        theRecall = np.empty(0)
        theSet = np.empty(0)

        #Loop through classifiers
        for name, label in names.items():
            clf = get_model(name)
            pr = getPrecRec(X, y, clf, ax)
            count = 0
            prec = pr.precision
            recall = pr.recall
        
            while((count:=count+1) <= 10):

                pr = getPrecRec(X, y, clf, ax)
        
                if prec.size < pr.precision.size:
                    prec += pr.precision[:prec.size]
                    recall += pr.recall[:recall.size]
                else:
                    p = prec/count
                    r = recall/count
        
                    p[:pr.precision.size] = pr.precision
                    r[:pr.recall.size] = pr.recall
                    prec += p
                    recall += r
        
            theSet = np.concatenate((theSet, np.full((prec.size), label)))
            thePrec = np.concatenate((thePrec, prec/count))
            theRecall = np.concatenate((theRecall, recall/count))
        
        prdf = pd.DataFrame(data={"precision": thePrec, 
                                "recall": theRecall, 
                                "Classification model": theSet})
        
        sns.lineplot(data=prdf, ax=ax, x="recall", y="precision", hue='Classification model',
                    palette=self.palette[:len(names)],lw=2)
        ax.set(xlim=(0,1), ylim=(0,1))
        return fig


class Doc(MLfit):

    def zone_vs_class(self, query='variability<25', *, zone_def=[0.1,0.95], 
                    var_name='log_epeak', bins=np.arange(-1,2.2,0.1)):
        """Table of histograms of the {var_name} with pulsar prediction zone columns 
        and association class rows. The number of sources are shown on the histogram plots, which are log-log scale.
        The energy bin above 100 GeV contains all entries above that. The vertical scale shows three decades to 1000.
        """
        # append columns with predection probability
        probs = self.predict_prob(query=None)
        dfc = pd.concat([self.df, probs], axis=1).copy()
        if query is not None:
            df=dfc.query(query).copy()
        else:
            df = dfc.copy()
        df.log_epeak = df.log_epeak.clip(-1,2)

        # set zone, class name, make a 2-d group
        df['zone'] = np.digitize(df.p_pulsar, zone_def)
        zone_names = 'blazar undetermined pulsar'.split()
        
        def set_class(x):
            if x in 'bll fsrq bcu unid'.split(): return x
            if x in 'psr msp'.split(): return 'pulsar' 
            return np.nan
        df['class_name'] = df.association.apply(set_class)
        class_names = 'bll fsrq pulsar unid bcu'.split()
        
        g = df.groupby(['class_name','zone'])
        
        fig = plt.figure(figsize=(12,12), layout="constrained")
        axd = fig.subplot_mosaic(
                    [ ['.', '.',  't',    't',  't'  ],
                        ['s', '.',  'z0',   'z1', 'z2' ],
                        ['s', 'c0', 'h00', 'h01', 'h02'],
                        ['s', 'c1', 'h10', 'h11', 'h12'],
                        ['s', 'c2', 'h20', 'h21', 'h22'],
                        ['s',  '.', '.',    '.',  '.'  ],
                        ['s', 'c3', 'h30', 'h31', 'h32'],
                        ['s', 'c4', 'h40', 'h41', 'h42'],
                        ['.', '.',  'b',   'b',   'b'  ], ],
                width_ratios = [0.5,0.75,3,3,3], 
                height_ratios=[0.5,0.5,3,3,3,0.5,3,3,1], )

        def make_label(ax,text, fontsize='large',  **kwargs):
            ax.text(0.5,0.5, text, ha='center', va='center', fontsize=fontsize,  **kwargs)
            ax.set_axis_off()
            
        for label, ax in axd.items():
            if label=='t':  make_label(ax, 'Pulsar prediction zone',fontsize=22); continue
            if label=='s':  make_label(ax, 'Association class', rotation='vertical',fontsize=22); continue
            if label=='b':  make_label(ax, 'Peak energy [GeV]' if var_name=='log_epeak' else var_name, fontsize=None,); continue
            if label[0]=='z': make_label(ax, zone_names[int(label[1])]); continue;
            if label[0]=='c': make_label(ax, class_names[int(label[1])]); continue;
        
            c,z = int(label[1]), int(label[2])
            try:
                cdf = g.get_group((class_names[c], z))
                sns.histplot( ax=ax, x=cdf[var_name], log_scale=(False, True), element='step', 
                color='maroon', bins=bins, edgecolor='w' )
            except KeyError: #if empty
                cdf = []
            ax.set(xlabel='', ylabel='', yticklabels=[], # xlim=(-1,2.1),
                ylim=(0.8,1e3))
            if  var_name=='log_epeak':
                ax.set(xticks=[-1,0,1,2], xticklabels=[] if c<4 else '0.1 1 10 100'.split())
            elif c<4:
                ax.set(xticklabels=[])
                
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.text(0.6,0.85, f'{len(cdf)}', transform=ax.transAxes, ha='right', fontsize='small')
        return fig
    

        
title = sys.argv[-1] if 'title' in sys.argv else None

def doc(nc=2, np=2, nf=4, kde=False, bcu=False ):
    from pylib.tools import FigNum, show_date

    def trainers():
        if np==1: 
            a = dict(psr=('psr',), msp=('msp',))
        else:
            a = dict( pulsar = ('psr', 'msp',) if np==2 else ('psr','msp','glc')) 
        b = dict( blazar=('bll', 'fsrq')) if nc==2 else \
            dict(bll=('bll',), fsrq=('fsrq',))
        return dict(**a,**b)
    
    skprop = dict(
        features= ('log_var', 'log_fpeak', 'log_epeak', 'd') if nf==4 else \
                  ('log_var', 'log_epeak', 'd'),
        clips = [(None,None), (None,None),(-1,3), (-0.1,2) ],
        trainers = trainers(),
        model_name = 'SVC',
        truth_field='association',
        # will be set to a trainers key
        trainer_field = 'trainer',
        )
    fn = FigNum(n=1, dn=0.1)
    if kde:
        show(f"""# KDE approach  ({dataset.upper()})""" if title is None else '# '+title)
        show_date()
        show("""Not applying ML, so no class fits to generate prediction model. Instead we compute KDE probability density distributions
        for the ML trainers, which we then apply to the unid and bcu associations.
        """)
    else:
        show(f"""# ML {nc}-class Classification """)

    with capture_show('Setup:') as imp:
        self = Doc(skprop)
    show(imp)
    if kde: 
        return self
    
    show(str(self))
    show(f"""## Feature distributions """)
    show_fig(self.pairplot, fignum=fn.next)

    show(f"""## Train then apply prediction """)
    self.train_predict()

    show(f"""### Precision-recall graph
         These evaluations shows how well the pulsars are distinguished from blazars.
         Comparison of our SVC model with the neural network alternative.""")
    fig, ax = plt.subplots(figsize=(6,6))
    ax.set(xlim=(0,1), ylim=(0,1))
    show_fig(self.precision_recall, dict(SVC='Support Vector Machine', NN='Neural network'), fignum=fn.next)


    # show(f"""### The confusion matrix""")
    # cmdf = self.confusion_matrix()
    # show(cmdf)
    # N = self.trainer_counts['pulsar']
    # TP = cmdf.iloc[-1,-1]
    # FP = cmdf.sum(axis=1)[-1] - TP
    # FN = cmdf.sum(axis=0)[-1] - TP
    # efficiency, purity = TP/N, TP/(FN+TP)
    # show(f'\n{purity=:.2f}, {efficiency=:.2f}' )

    show(f"""### All predictions""")
    table = self.prediction_association_table()
    show(table)
    show_fig(self.plot_prediction_association,table, fignum=fn.next)

    show(f"""## The issue with this
         In figure {fn.next} we show the pulsar probability distributions for the training classes, and
         the unid target. The large number of unid sources not near 0 or 1 demonstrates the 
         existence of a component not in the training set, violating the basic ML assumption.
         """)
    show_fig(self.pulsar_prob_hists, fignum=fn.current)

    # show(f"""#### Write summary file, adding diffuse correlation""")
    self.write_summary()
    return self

if 'doc' in sys.argv:
    self = doc()
    plt.close() # don't know why this is needed

def apply_diffuse(df, nc=2):
    df3 = pd.read_csv(f'files/{dataset}_{nc}_class_classification.csv', index_col=0)
    df['diffuse'] = df3.diffuse

include_bcu = 'include_bcu' in sys.argv

def kde_setup(kde_vars = 'sqrt_d log_epeak diffuse'.split(), nc=2, bcu=False,
           cut = '0.15<Ep<4 & d>0.2 & variability<25'   ):
    self = doc(nc=nc, np=1, kde=True, bcu=bcu)

    def make_group(self):

        def groupit(s):
            if 'include_bcu' in sys.argv:
                if s.association in 'unid bcu'.split(): return 'unid' # special
            if s.association in 'unid bcu spp'.split(): return s.association
            # if s.association=='bcu': return 'bcu'
            if ~ pd.isna(s.trainer): return s.trainer
            return np.nan
        df = self.df
        df['subset'] = df.apply(groupit, axis=1)
    make_group(self)
    
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
    # # apply diffuse
    # df3 = pd.read_csv(f'files/{dataset}_{nc}_class_classification.csv', index_col=0)
    # dfc['diffuse'] = df3.diffuse
    apply_diffuse(dfc)

    
    show(f"""## Create KDE functions instead of ML training
    * Classes: {', '.join(self.trainers.keys())}
    * Features: {', '.join(kde_vars)} 
    
    Apply to unids {'+ bcus' if include_bcu else ''}
    """)
    apply_kde(self, dfc, kde_vars)
    return self, dfc

def apply_kde(self, df=None, features=None):
    from pylib.kde import Gaussian_kde
    if df is None: df = self.df.copy() 
    if features is None: features=self.features
    for name, sdf in df.groupby('subset'):
        try:
            gde = Gaussian_kde(sdf,  features)
        except Exception as msg:
            print(msg, file=sys.stderr)
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

if 'kde_no_dcut' in sys.argv:
    self, data = kde_setup(cut='0.15<Ep<10 & variability<25')
    unid = data[data.subset=='unid']

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
    kde_kw   = dict(hue_order=self.trainer_names, palette=self.palette,  alpha=0.4,legend=False)
    target_kw= dict(hue_order=self.trainer_names, palette=self.palette,)

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
            ax.set(#ylim=(-2.75,4), 
                   **fpeak_kw('y'), xlim=(-1,2.5))
        ax1.set(xticklabels=[])

    spectral( *axx[:,0])
    flux(     *axx[:,1])
    return fig

if 'bcu' in sys.argv:
    import warnings
    warnings.filterwarnings("ignore")
    self, dfc = kde_setup()


