from pylib.fermi_sources import *

class ML(FermiSources):

    def __init__(self,  *pars, 
                 title="""Application of machine learning to the Fermi unassociated sources
                """,
                 **kwargs):

        show('# '+title)
        show_date()
        super().__init__(*pars, **kwargs)
        return
        fcat = self.fermicat 
        self.df.loc[:,'sgu'] = fcat['flags'].apply(lambda f: f[14])
        self.df.loc[:,'fcat_epeak'] = fcat.specfunc.apply(lambda f: f.sedfun.peak)
        self.df.loc[:,'fcat_curvature']= 2 * fcat.specfunc.apply(lambda f: f.curvature())

       
    def outline(self):
        show(f"""
            ## Outline
            **Goal: use predictive artificial intelligence to classify source types of the unid's**
            
            Procedure:
            * Choose the standard `scikit-learn` ML implementation 
            * Choose "features"
            * Evaluate classifier options, select one
            * Validate, perhaps adjust feature set
            * Apply to the unid's (including "SGU"s)
            """)
        
    def show_prediction_association(self, fignum=None, caption=''):
        show(f"""## Predictions
        """)

        def simple_pivot(df, x='prediction', y= 'association'):        
            return df.groupby([x,y]).size().reset_index().pivot(
                columns=x, index=y, values=0)
            
        df_plot = simple_pivot(self.df)
        fig, ax = plt.subplots()
        ax = df_plot.plot.barh(stacked=True, ax=ax)
        ax.invert_yaxis()
        ax.set(title='Prediction counts', ylabel='Association type')
        ax.legend(bbox_to_anchor=(0.8,0.8), loc='lower left', frameon=True )
        show(fig, fignum=None, caption=caption)
        show(f"""Notes:
        * The target is the unids, but applied to all
        * BCUs mostly blazars, a check
        * BLL, FSRQ, Pulsar look OK (a little redundant), a check
        """)

    def compare_variabilities(self, fignum=None):
        show(f"""### Variability measures: nbb vs variability
        """)
        fig, ax = plt.subplots(figsize=(8,8))
        df = self.df.set_index('association').loc[('bll fsrq'.split())]
        sns.scatterplot(df, x=np.log10(df['var']), y=np.log10(df.nbb) , hue='association',
                        size='log_eflux', sizes=(5,100), ax=ax);
        ax.set(xlabel='log(variabilitiy)', ylabel='log(nbb)');
        ax.axvline(np.log10(26), color='0.6', ls='--')
        ax.axhline(np.log10(1.5), color='0.6', ls='--')
        show(fig, fignum=fignum,
             caption='Log nbb vs log variability. Dashed lines show thresholds.')
        show(f"""**➜** Choose `nbb` since it detects BL Lac variability missed by 4FGL
        """)

    def scatter_train_predict(self,  x,y, fignum=None, caption='', target='unid', **kwargs):
        
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,6),
                                    sharex=True,sharey=True,
                                    gridspec_kw=dict(wspace=0.1))

        size_kw = dict(size='log_eflux', sizes=(20,150),size_norm=(-12,-10))
        hue_kw = lambda what: dict(hue_order=self.mlspec.target_names, hue =what)
        df = self.df

        ax1.set(**kwargs)
        ax1.set_title('Training')
        ax2.set_title(f'{target} prediction')
        sns.scatterplot(self.train_df, x=x, y=y,  **hue_kw('association'), **size_kw,  ax=ax1)
        ax1.legend(loc='upper right', fontsize=12)

        target_df = df.query(f'association=="{target}"')
        assert len(target_df)>0, f'Failed to find target {target}'
        sns.scatterplot(target_df, x=x, y=y,  **hue_kw('prediction'), **size_kw,   ax=ax2)
        ax2.legend(loc='upper right',fontsize=12)
        ax2.set(xlabel=ax1.get_xlabel())
        
        fig.text(0.514, 0.5, '⇨', fontsize=50, ha='center')
        show(fig, fignum=fignum, caption=caption)
        
    def pairplot(self, fignum=None):
        show(f"""## Examine correlations among selected features
        """)
        show(super().pairplot(), fignum=fignum, caption="""A KDE "pairplot" of the chosen features
        """)

    # def train_predict(self, fignum=None):
    #     show(f"""## Train, predict, assess results
    #     """)
    #     super().train_predict(show_confusion=True, hide=True)

    def show_sgus(self, fignum=None, caption=''):
        dfq  = self.df.query('sgu==True')
        show(f"""## What about the  {len(dfq)} SGU sources?""")
        
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(9,2.5),
                gridspec_kw=dict(wspace=0.3))
        sns.countplot(dfq, y ='prediction', hue='prediction',
                      hue_order=self.mlspec.target_names,
                      ax=ax2)
        ax2.get_legend().remove()
        sns.countplot(dfq, y ='association', ax=ax1)
        show(fig, caption="""Left: initial association category; 
        Right: prediction assignments.
        """)
        show(f"""Conclusion: mostly pulsars!""")
    
        target_names = self.mlspec.target_names
        # df = self.df.query('sgu==True')
        fig, ax = plt.subplots(figsize=(6,6))
        sns.scatterplot(dfq, x='log_epeak', y='curvature', 
                        hue_order=target_names, hue='prediction',
                        size='log_eflux', sizes=(10,200),size_norm=(-12,-10),
                        ax=ax )
        ax.set(**epeak_kw('x'), #xticks = [-1,0,1,2,3], 
               title='SGU assignments')
        ax.legend(loc='upper right', fontsize=12);
        show(fig, fignum=fignum, caption=caption)
        return 
        
    def classifier_evaluation(self):
        cnames =  [
            "Nearest Neighbors",
            "Linear SVM",
            "RBF SVM",
            #"Gaussian Process",
            "Decision Tree",
            "Random Forest",
            "Neural Net",
            "AdaBoost",
            "Naive Bayes",
            "QDA",
        ]
        show(f"""
        ## Classifiers
        This is an effort led by my student Timothy Tomter
        The full list that was considered was
        
        {cnames}
    
        Scores for the two best are shown here:
            """)    
        show('images/classifier_comparison.png')
    
        show("""We chose the first, a Support Vector Classifier""")
    
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
    
    def show_notes(self):
        show("""
            ## Notes, todos:
            * Reexamine feature set using random forest importance measures
            * Check sky positions--are they consistent with presumed counterpart catalog detection efficiencies and
            expected source distributions? Perhaps include it in the training after accounting for efficiency<br>
            * Perhaps expand the "other" category, e.g. SNRs
            * Check some of the individual ones brought up here
        """)

def main():

    fn = FigNum()
    # sns.set_context('talk')
    self =  ML(mlspec=MLspec(
        features='log_var log_fpeak log_epeak curvature'.split()))
    self.outline()
    self.show_data()
    self.compare_variabilities(fignum=fn.next)
    self.pairplot(fignum=fn.next)
    self.classifier_evaluation()
    self.train_predict(show_confusion=True)
    self.show_prediction_association(fignum=fn.next)
    self.curvature_epeak_flux(fignum=fn.next); fn.next
    # self.show_sgus(fignum=fn.next)
    self.show_notes()

import sys
if len(sys.argv)>1:
    if sys.argv[1]=='main':
        main()
    elif sys.argv[1]=='setup':
        self =  ML(title='ML doc development')
        self.train_predict()

    
