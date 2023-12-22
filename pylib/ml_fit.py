import sys
from pathlib import Path
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.catalogs import Fermi4FGL
from pylib.tools import FigNum, show_date, update_legend, epeak_kw
from pylib.sklearn import SKlearn

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
                 fgl:str=dataset):
        self.df, self.cat_len = self.load_data(fgl)
        super().__init__(self.df, skprop)
        self.palette =['cyan', 'magenta', 'yellow'] if dark_mode else 'green red blue'.split()
        
    def __repr__(self):
        return f"""MLfit applied to 4FGL-{dataset.upper()} \n\nSKlearn: {super().__repr__()}
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
            tsel = self.df[self.target_field].apply(lambda c: c in self.target_names)\
            if self.target_names is not None else slice(None)
            df = self.df.loc[tsel]
        else:
            df = self.df.query(query)
        kw = dict(kind='kde', hue=self.target_field, hue_order=self.target_names, height=2, corner=True)
        kw.update(**kwargs)
        g = sns.pairplot(df, vars=self.features,  palette=self.palette[:len(self.target_names)], **kw,)
        return g.figure
    
    def ep_vs_d(self):
        # from pylib.tools import epeak_kw
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12, 5),sharex=True,sharey=True,
                                    gridspec_kw=dict(wspace=0.05))
        kw = dict( x='log_epeak', y='d',  palette=self.palette[:len(self.targets)],
                  hue_order=self.target_names)
        
        sns.kdeplot(self.df,ax=ax1, hue=self.target_field, **kw)
        sns.kdeplot(self.df.query('association=="unid"'),ax=ax2, hue='prediction',**kw)

        ax1.set(**epeak_kw(),ylabel='curvature $d$', xlim=(-1.5,1.5)); ax2.set(**epeak_kw())
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

def doc(nc=2, np=2):
    from pylib.tools import FigNum, show_date

    def targets(nc, np=2):
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
    show(f"""# ML {nc}-class Classification  ({dataset.upper()})""")
    show_date()
    with capture_show('Setup:') as imp:
        self = MLfit(skprop)
    show(imp)
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

