import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.catalogs import Fermi4FGL
from pylib.tools import FigNum, show_date, update_legend

if 'show' in sys.argv:
    from pylib.ipynb_docgen import show, show_fig

sns.set_theme('notebook' if 'talk' not in sys.argv else 'talk', font_scale=1.25) 
if 'dark' in sys.argv:
    plt.style.use('dark_background') #s
    plt.rcParams['grid.color']='0.5'
    dark_mode=True
else:
    dark_mode=False
fontsize = plt.rcParams["font.size"] # needed to be persistent??

dataset='dr3' if 'dr3' in sys.argv else 'dr4'

@dataclass
class MLspec:

    features: tuple = tuple("""log_var log_fpeak log_epeak d """.split())
    target : str ='target'
    target_names:tuple = tuple('bll fsrq pulsar'.split())
    psr_names: tuple = tuple('psr msp'.split()) # no glc

    palette =['yellow', 'magenta', 'cyan'] if dark_mode else 'green red blue'.split()
    model_name : str = 'SVC'

    def __repr__(self):
        return f"""
        Scikit-learn specifications: 
        * features: {self.features}
        * target names: {self.target_names}
        * pulsar names: {self.psr_names}
        * model_name : {self.model_name}"""

    def get_model(self):
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
        F,kw = cdict[self.model_name]
        return F(**kw)

def lp_pars(fgl):
    # extract LP spectral function from 4FGL catalog object get its parameters
    df = pd.DataFrame(index=fgl.index)
    df['lp_spec'] = [fgl.get_specfunc(name, 'LP') for name in df.index]
    sed = df['lp_spec']
    df['Ep'] = 10**sed.apply(lambda f: f.epeak)
    df['Fp'] = 10**sed.apply(lambda f: f.fpeak)
    df['d'] = sed.apply(lambda f: f.curvature()).clip(-0.1,2)
    df['d_unc'] = 2*fgl.field('unc_LP_beta')
    return df.drop(columns=['lp_spec'])

class MLfitter(MLspec):

    def __init__(self, fgl=dataset):
        self.df, self.cat_len = self.load_data(fgl)

        self.model = self.get_model() 
        self.setup_target()

        print(f"""Targets: {dict(self.df.groupby('target').size())}    """)
        print(f"""pulsar: {dict(self.df.groupby('association').size()[list(self.psr_names)])} """)

    def __repr__(self):
        return f"""MLfitter applied to 4FGL-{dataset.upper()} \n{super().__repr__()}
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
        # df['lp_spec'] = [fgl.get_specfunc(name, 'LP') for name in df.index]
        # sed = df['lp_spec']
        # df['Ep'] = 10**sed.apply(lambda f: f.epeak)
        # df['Fp'] = 10**sed.apply(lambda f: f.fpeak)
        # df['d'] = sed.apply(lambda f: f.curvature()).clip(-0.1,2)


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

    def setup_target(self):
        df = self.df
        def target_namer(name):
            name = name.lower()
            if name in 'bll fsrq'.split(): return name
            if name in self.psr_names: return 'pulsar'
            return None
        df['target'] = df.class1.apply(target_namer)

    def getXy(self,) : #, features, target, target_names=None):
        """Return an X,y pair for ML training
        """

        tsel = self.df[self.target].apply(lambda c: c in self.target_names)\
            if self.target_names is not None else slice(None)
        assert sum(tsel)>0, 'No data selected for training.'
        return self.df.loc[tsel, self.features], self.df.loc[tsel, self.target]

    def fit(self, model):
        """
        """
        X,y = self.getXy() 
        return model.fit(X,y)
    
    def predict(self, query=None):
        """Return a "prediction" vector using the classifier, required to be a trained model

        - query -- optional query string
        return a Series 
        """
        # the feature names used for the classification -- expect all to be in the dataframe
        assert hasattr(self, 'classifier'), 'Model was not fit'
        fnames = getattr(self.classifier, 'feature_names_in_', [])
        assert np.all(np.isin(fnames, self.df.columns)), f'classifier not set properly'
        dfq = self.df if query is None else self.df.query(query) 
        assert len(dfq)>0, 'No data selected'
        ypred = self.classifier.predict(dfq.loc[:,fnames])
        return pd.Series(ypred, index=dfq.index, name='prediction')

    def train_predict(self):
        
        model = self.model #get_model(model_name)
        try:
            X,y = self.getXy() 
            model.probability=True # needed to get probabilities
            self.classifier =  model.fit(X,y)
            # self.classifier = self.fit( model )
            # self.probs = self.classifier.predict_proba(y)
        except ValueError as err:
            print(f"""Bad data? {err}""")
            print(self.df.loc[:,self.features].describe())
            return

        self.df['prediction'] = self.predict()

        # global references to the data sets for plots below.
        df = self.df
        self.train_df = df[df.association.apply(lambda x: x in self.target_names)]
        self.unid_df = df[df.association=='unid']

    def predict_prob(self, query='association=="unid"'):
        """Return DF with fit probabilities
        """
        mdl = self.classifier
        assert mdl.probability, 'Fit must be with probability  True' 
        dfq = self.df.query(query) if query is not None else self.df
        X = dfq.loc[:, self.features]
        return pd.DataFrame(mdl.predict_proba(X), index=dfq.index,
                            columns=['p_'+ n for n in self.target_names])

    def confusion_display(self,  test_size=0.25, **kwargs):
        """Confusion analysis plots.
        """
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay 

        Xy = self.getXy()
        split_kw = dict(test_size=test_size, shuffle=True)
        split_kw.update(kwargs)
        Xtrain, Xtest, ytrain, ytest  =  train_test_split(*Xy, **split_kw)

        model = self.model #                       # 2. instantiate model
        classifier = model.fit(Xtrain, ytrain)     # 3. fit model to data
        y_model = model.predict(Xtest)             # 4. predict on new data

        print (f"""Confusion analysis, test size = {100*test_size:.0f}%
        * Model: {str(model)}
        * Features: {list(Xy[0].columns)}
        Accuracy: {100*accuracy_score(ytest, y_model):.0f}%
        """)

        fig, axx = plt.subplots(ncols=2, figsize=(12,5),
                               gridspec_kw=dict(wspace=0.4))
        # Plot non-normalized confusion matrix
        titles_options = [
            ("Unnormalized", None),
            ("Normalized", "true"),
        ]
        for ax, (title, normalize)  in zip(axx.flatten(), titles_options):
            disp = ConfusionMatrixDisplay.from_estimator(
                    classifier, Xtest, ytest,
                    display_labels=self.target_names,
                    cmap=plt.cm.Blues,
                    normalize=normalize,
                    ax=ax,
                )

            ax.set_title(title)
        return fig 
   
    def pairplot(self, query='', **kwargs):  
        """Corner plots.
        """      
        if query=='':
            tsel = self.df[self.target].apply(lambda c: c in self.target_names)\
            if self.target_names is not None else slice(None)
            df = self.df.loc[tsel]
        else:
            df = self.df.query(query)
        kw = dict(kind='kde', hue=self.target, hue_order=self.target_names, height=2, corner=True)
        kw.update(**kwargs)
        g = sns.pairplot(df, vars=self.features,  palette=self.palette, **kw,)
        return g.figure

    def plot_pulsar_prob(self):
        """Predicted pulsar probability distributions for unid sources. 
        Upper plot: Stacked histograms, according to which target class was selected
        by the prediction. Lower plot: empirical cumulative distributions.
        """

        pp = self.predict_prob()
        pp['prediction'] = self.df.prediction
        hue_kw = dict(  hue = 'prediction',
                        hue_order=self.target_names,
                        palette=self.palette)
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(6,5), sharex=True,
                                    gridspec_kw=dict(hspace=0.1))
        sns.histplot(pp, ax=ax1, x='p_pulsar', element='step',bins=np.arange(0,1.01,0.025), 
                multiple='stack', **hue_kw,
                        # hue = 'prediction',
                        # hue_order=self.target_names,
                        # palette=self.palette,
                        edgecolor= '0.9',   alpha=0.8,  )
        update_legend(ax1, pp, 'prediction', loc='upper center', fontsize=12)

        sns.ecdfplot(pp, **hue_kw, ax=ax2, x='p_pulsar', legend=False);
        return fig

    def prediction_association_table(self):
        """The number of sources classified as each of the three targets, according to the 
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

    def plot_prediction_association(self, table):
        """Bar chart showing the prediction counts for each association type, according 
        to the predicted target class. 
        """
    
        fig, ax = plt.subplots()
        ax = table.plot.barh(stacked=True, ax=ax, color=self.palette)
        ax.invert_yaxis()
        ax.set(title='Prediction counts', ylabel='Association type')
        ax.legend(bbox_to_anchor=(0.78,0.75), loc='lower left', frameon=True ,
                title='Target class')
        # curly bracket to denote pulsar associations
        ax.text(0.07, 0.6, '}', fontsize=80, va='center', transform=ax.transAxes) 
        ax.text(0.22, 0.6, 'pulsar target', va='center', 
                color='white' if dark_mode else 'k', transform=ax.transAxes)
        return fig

   
    def write_summary(self, 
                summary_file=f'files/{dataset}_classification.csv', 
                overwrite=True):
            
            if Path(summary_file).is_file() and not overwrite:
                print(f'File `{summary_file}` exists--not overwriting.')
                return
            
            def set_pulsar_type(s):
                if s.association in self.psr_names: return s.association
                if s.prediction=='pulsar': return s.association+'-pulsar'
                return np.nan
            def set_source_type(s):
                return s.association if s.association in self.psr_names else \
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

            cols= 'source_type glat glon significance r95 Ep Fp d hasdr3 diffuse p_bll p_fsrq p_pulsar'.split()
            df.loc[:, cols].to_csv(summary_file, float_format='%.3f') 
            print(f'Wrote {len(df)}-record summary, using model {self.model}, to `{summary_file}` \n  columns: {cols}')


if 'show' in sys.argv:
    from pylib.ipynb_docgen import show, capture_hide,capture_show, show_fig
    
if 'doc' in sys.argv:
    from pylib.ipynb_docgen import show, capture_hide,capture_show, show_fig
    fn = FigNum(n=1, dn=0.1)

    show(f"""# Machine Learning classification for {dataset.upper()}
    Output for the [confluence page](https://confluence.slac.stanford.edu/display/SCIGRPS/The+UW+Machine+Learning+classification) """)
    show_date()
    with capture_show('setup printout') as setup_print:
        self = MLfitter()
        show(setup_print)

    show(f"""

    ### Data preparation
    We select {len(self.df)} persistent point sources from the 4FGL-{dataset.upper()} catalog. We create
    an `association` property based on the `CLASS1` field. This takes its lower-case value with `unk` combined with 
    completely unassociated to `unid`.

    ### Scikit-learn parameters

    The context is the scikit-learn package
    
    * __Targets__: We observe that 90% of the associated sources are pulsars (MSP (`msp`) or young (`psr`) or 
    blazars, BL Lac (`bll`) or FSRQ (`fsrq`), including `bcu`. Thus we choose three classes for training:
    pulsar=`psr+msp`,
    combined to represent pulsars,  `bll`, and `fsrq`. Note that this does not include the blazar `bcu` class, as 
    it is a mixture and we choose to distinguish `bll` from `fsrq`.

    * __Features__: We assume that the separation needs only variability and spectral information. For the 
    spectral information we use the three parameters of the log parabola spectral function: the energy and
    energy flux at the peak of the SED, $E_p$  and $F_p$ and the curvature $d$, (We examine spatial
    information later to check that sources predicted to be "pulsars" are  indeed Galactic.)
    * __Classification model__: We use the SVC model.
    """)

                
    show(f"""### The features for the target names
    This plot shows the distributions of the features for each of the three targets. For the variability,
    Ep  and Fp we use the log base 10 values log_var, log_fpeak, and log_epeak.
    """)
    show_fig(self.pairplot, fignum=fn.next,
             caption="""Corner plot of the training sources showing the four features for each of 
             the three targets. """) 
    self.train_predict()
    show("""Note especially the roles of the variability and the correlation of curvature $d$ and $E_p$
         in distinguishing pulsars from blazars. """)

    show(f"""## Fitting
    This means training the model with the target classes, then using it to predict the most likely
    identity of other sources. We are most interested in the unid sources, but apply it to all.
    """)
    show(f"""### Confusion analysis
    The following is the output of the "confusion analysis", which trains with a randomly selected 75%
    of the targets, then checks the result of predictions for the remaining 25%.
    """)
    with capture_show('') as confusion_print:
        fig = self.confusion_display()
    show(confusion_print)
    show(fig, fignum=fn.next, caption='')
    show("""We are only interested in how well the pulsar category is selected, so the mixing between the
    two blazar categories is not an issue. So ~10% of real pulsars are lost, with <5% contamination of
    FSRQs. This means ~90% efficiency and >95% purity. 
    """)

    show(f"""### Prediction statistics
    This shows the category assignment for all the association types, including the targets.
    """)
    table = self.prediction_association_table()

    show_fig(self.plot_prediction_association, table, fignum=fn.next, )
    show(f"""#### Table with values
         """)
    show(table)
    
    show(f"""## Unid prediction probabilities
    The prediction procedure estimates the three probabilities. In Figure {fn.next} we show the distribution of the pulsar
    probability, tagged with the prediction category, and the cumulative distributions.
    """)
    show_fig(self.plot_pulsar_prob, fignum=fn, )


    show("""## Unid features
    Finally, here are the distributions of the features for the unid class, for each prediction.
    """)
    show_fig(self.pairplot, "association=='unid'", hue='prediction', fignum=fn.next,
             caption=f"""Corner plot of the `unid` sources showing the four features for each of 
             the three predicted target classes.""" )
    
    show(f"""## Write summary table if it doesn't exist.""")
    self.write_summary( overwrite=False)


