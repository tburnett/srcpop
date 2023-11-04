import sys
from pathlib import Path
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from pylib.catalogs import Fermi4FGL


sns.set_theme('notebook' if 'talk' not in sys.argv else 'talk', font_scale=1.25) 
if 'dark' in sys.argv:
    plt.style.use('dark_background') #s
    plt.rcParams['grid.color']='0.5'
    dark_mode=True
else:
    dark_mode=False
fontsize = plt.rcParams["font.size"] # needed to be persistent??


        # if dark_mode:
        #     self.hue_kw.update(palette='yellow magenta cyan'.split(), edgecolor=None)
        # else:
        #     self.hue_kw.update( palette='green red blue'.split())

dataset='dr3' if 'dr3' in sys.argv else 'dr4'

def show_date():
    from pylib.ipynb_docgen import show
    import datetime
    date=str(datetime.datetime.now())[:16]
    show(f"""<h5 style="text-align:right; margin-right:15px"> {date}</h5>""")

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

def update_legend(ax, data, hue, **kwargs):
    """ seaborn companion to insert counts in legend,
    perhaps change location or font.

    """
    if len(kwargs)>0:
        ax.legend(**kwargs)
    gs = data.groupby(hue).size()
    leg = ax.get_legend()
    for tobj in leg.get_texts():
        text = tobj.get_text()
        if text in gs.index:
            tobj.set_text(f'({gs[text]}) {text}')


@dataclass
class MLspec:

    features: tuple = tuple("""log_var log_fpeak log_epeak d """.split())
    target : str ='target'
    target_names:tuple = tuple('bll fsrq pulsar'.split())
    psr_names: tuple = tuple('psr msp glc'.split()) #spp

    palette =['yellow', 'magenta', 'cyan'] if dark_mode else 'green red blue'.split()
    model_name : str = 'SVC'

    def __repr__(self):
        return f"""
        Scikit-learn specifications: 
        * features: {self.features}
        * target: {self.target}
        * target_names: {self.target_names}
        * psr_names: {self.psr_names}
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

class MLfitter(MLspec):

    def __init__(self, fgl=dataset):
        self.df, self.cat_len = self.load_data(fgl)

        self.model = self.get_model() 
        self.setup_target()

        print(f"""Targets: {dict(self.df.groupby('target').size())}    """)
        print(f"""pulsar: {dict(self.df.groupby('association').size()[list(self.psr_names)])} """)

    def load_data(self, fgl):
        """Extract subset of 4FGL information relevant to ML pulsar-like selection
        """

        cols = 'glat glon significance r95 variability class1'.split()
        fgl = Fermi4FGL(fgl)

        # fgl.loc[:, 'beta'] = fgl.field('LP_beta').astype(np.float64)
        
        # remove soruces without variability or r95
        df = fgl.loc[:,cols][(fgl.variability>0) & (fgl.r95>0) &(fgl.significance>4)].copy()
        print(f"""Remove {len(fgl)-len(df)} without valid r95 or variability or significance>4
            -> {len(df)} remain""")

        df['lp_spec'] = [fgl.get_specfunc(name, 'LP') for name in df.index]
        sed = df['lp_spec']
        df['Ep'] = 10**sed.apply(lambda f: f.epeak)
        df['Fp'] = 10**sed.apply(lambda f: f.fpeak)
        df['d'] = sed.apply(lambda f: f.curvature()).clip(-0.1,2)

        # now remvoe any bad SED fits
        # n = len(df)
        # df = df[df.d>-0.2]
        # if len(df)<n:
        #     print(f'Remove {n-len(df)} without good LP spectral fit -> {len(df)} remain')

        # reclassify associations
        def reclassify(class1):
            cl = class1.lower()
            if cl in 'bll fsrq psr msp bcu unk spp glc unid'.split():
                return cl
            # if cl in 'psr msp'.split(): return 'psr'
            if cl=='': return 'unid'
            return 'other'
        df['association'] = df.class1.apply(reclassify)
        # append columns woth logs needed later
        df['log_var'] = np.log10(df.variability)
        df['log_epeak'] = np.log10(df.Ep)
        df['log_fpeak'] = np.log10(df.Fp)  

        return df, len(fgl)

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
        """
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

    def plot_predict_prob(self):

        pp = self.predict_prob()
        pp['prediction'] = self.df.prediction
        fig, ax = plt.subplots(figsize=(5,3))
        sns.histplot(pp, ax=ax, x='p_pulsar', element='step',bins=np.arange(0,1.01,0.025), 
                multiple='stack',
                        hue = 'prediction',
                        hue_order=self.target_names,
                        palette=self.palette,
                        edgecolor= '0.9',   alpha=0.8,  )
        return fig

    def plot_prediction_association(self, ):

        def simple_pivot(df, x='prediction', y= 'association'):        
            ret =df.groupby([x,y]).size().reset_index().pivot(
                columns=x, index=y, values=0)
            return ret.reindex(index='bll fsrq psr msp spp glc bcu other unk unid'.split())
             
        df_plot = simple_pivot(self.df)
        fig, ax = plt.subplots()
        ax = df_plot.plot.barh(stacked=True, ax=ax, color=self.palette)
        ax.invert_yaxis()
        ax.set(title='Prediction counts', ylabel='Association type')
        ax.legend(bbox_to_anchor=(0.78,0.75), loc='lower left', frameon=True )
        return fig

    def write_summary(self, 
            summary_file=f'files/{dataset}_pulsar_summary.csv', 
            overwrite=True):
        
        if Path(summary_file).is_file() and not overwrite:
            print(f'File `{summary_file}` exists--not overwriting.')
            return
        
        def set_pulsar_type(s):
            if s.association in self.psr_names: return s.association
            if s.prediction=='pulsar': return s.association+'-pulsar'
            return np.nan

        def get_diffuse(df):
            from astropy.coordinates import SkyCoord
            from pylib.diffuse import Diffuse
            diff = Diffuse()
            sdirs = SkyCoord(df.glon, df.glat, unit='deg', frame='galactic')
            return  diff.get_values_at(sdirs)
            
        df = self.df.copy()
        
        df['p_pulsar'] = self.predict_prob(query=None).loc[:,'p_pulsar']
        
        # set source_type for pulsars and pulsar-pred;ctions
        df['source_type'] = df.apply(set_pulsar_type, axis=1)
        df = df[df.source_type.notna()]
            

        # add a diffuse column
        df['diffuse'] = get_diffuse(df) #self.get_diffuses(df)

        cols= 'source_type glat glon significance r95 Ep Fp d p_pulsar diffuse'.split()
        df.loc[:, cols].to_csv(summary_file, float_format='%.3f') 
        print(f'Wrote {len(df)}-record summary, using model {self.model}, to `{summary_file}` \n  columns: {cols}')



if 'show' in sys.argv:
    from pylib.ipynb_docgen import show, capture_hide,capture_show
    
if 'doc' in sys.argv:
    from pylib.ipynb_docgen import show, capture_hide,capture_show

    show(f"""# The UW Machine Learning classification for {dataset.upper()}
    Output for the [confluence page](https://confluence.slac.stanford.edu/display/SCIGRPS/The+UW+Machine+Learning+classification) """)
    show_date()


    show(f"""
    ### Scikit-learn parameters

    The context is the scikit-learn package
    
    * Targets: We observe that 90% of the associated sources are pulsars (MSP (msp) or young (psr)) or 
    blazars, BL Lac (bll) or FSRQ (fsrq). Thus we choose three classes for training:  pulsar=psr+msp+glc,
    combined to represent pulsars,  bll, and fsrq. Not that this does not include the blazar bcu class, as 
    it is a mixture.
    * Features: We assume that the separation needs only variability and spectral information. For the 
    spectral information we use the three parameters of the log parabola spectral function: the energy and
    energy flux at the peak of the SED, $E_p$  and $F_p$ and the curvature $d$, (We examine spatial
    information later to check that "pulsars" are  indeed Galactic.)
    * Classification model: We use the SVC model.
    """)

    with capture_show('setup printout') as setup_print:
        self = MLfitter()
        show(setup_print)

    show(f"""
    ### Data selection
    We start with the {self.cat_len} sources in the 4FGL-{dataset.upper()} catalog, and discard those
    without an R95 value (extended sources) or with a significance < 4, with {len(self.df)} remaining. 
    """)
            
    show(f"""### The features for the target names
    This plot shows the distributions of the features for each of the three targets. For the variability,
    Ep  and Fp we use the log base 10 values log_var, log_fpeak, and log_epeak.
    """)
    show(self.pairplot(), fignum=1, caption='');
    self.train_predict()

    show(f"""##Fitting
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
    show(fig, fignum=2, caption='')
    show("""We are only interested in how well the pulsar category is selected, so the mixing between the
    two blazar categories is not an issue. So ~10% of real pulsars are lost, with <5% contamination of
    FSRQs. This means ~90% efficiency and >95% purity. 
    """)

    show(f"""### Prediction statistics
    This shows the category assignment for all the association types, including the targets.
    """)
    show(self.plot_prediction_association(), fignum=3, caption='')

    
    show("""## Unid probabilities
    The prediction procedure estimates the three probabilities. Here is the distribution of the pulsar
    probability, tagged with the prediction category.
    """)
    show(self.plot_predict_prob(), fignum=4, caption='')

    show("""## Unid features
    Finally, here are the distributions of the features for the unid class, for each prediction.
    """)
    show(self.pairplot("association=='unid'", hue='prediction'), fignum=5, caption='')
    
    self.write_summary( overwrite=False)
