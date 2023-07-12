from pylib.fermi_sources import *

class ML(FermiSources):

    def __init__(self, *pars, **kwargs):
        from utilities.catalogs import Fermi4FGL
        show("""# Appplication of machine learning to the Fermi unassociated sources
        """)
        super().__init__(*pars, **kwargs)
        fcat = Fermi4FGL()
        self.df.loc[:,'sgu'] = fcat['flags'].apply(lambda f: f[14])
        self.df.loc[:,'fcat_epeak'] = fcat.specfunc.apply(lambda f: f.sedfun.peak)
        self.df.loc[:,'fcat_curvature']= fcat.specfunc.apply(lambda f: f.curvature())

        sns.set_context('notebook')
        
    def outline(self):
        show(f"""
            ## Outline
            **Goal: use predictive artificial inteliegence to classify source types of the unid's**
            
            Procedure:
            * Choose the standard `scikit-learn` ML implementation 
            * Choose "featues"
            * Evaluate classifier options, select one
            * Validate, perhaps adjust feature set
            * Apply to the unid's the SGU's
            """)

        
    def show_prediction_association(self):
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
        show(fig)
        show(f"""Notes:
        * The target is the unids, but applied to all
        * BCUs mostly blazars, a check
        * BLL, FSRQ, Pulsar look OK (a little redundant), a check
        """)

    def compare_variabilities(self):
        show(f"""### Variability measures: nbb vs variability
        """)
        fig, ax = plt.subplots(figsize=(8,8))
        df = self.df.set_index('association').loc[('bll fsrq'.split())]
        sns.scatterplot(df, x=np.log10(df['var']), y=np.log10(df.nbb) , hue='association',
                        size='log_eflux', sizes=(5,100), ax=ax);
        ax.set(xlabel='log(variabilitiy)', ylabel='log(nbb)');
        ax.axvline(np.log10(26), color='0.6', ls='--')
        ax.axhline(np.log10(1.5), color='0.6', ls='--')
        show(fig, caption='Log nbb vs log variability. Dashed lines show thresholds.')
        show(f"""**➜** Choose `nbb` since it detects BL Lac variability missed by 4FGL
        """)
    def scatter_train_predict(self,  x,y, caption='', target='unid', **kwargs):
        
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,6),
                                    sharex=True,sharey=True,
                                    gridspec_kw=dict(wspace=0.2))

        target_names = self.mlspec.target_names
        df = self.df

        kw = dict()
        kw.update(kwargs)
        ax1.set(**kw)
        ax1.set_title('Training')
        ax2.set_title(f'{target} prediction')
        sns.scatterplot(self.train_df, x=x, y=y, 
                        hue_order=target_names, hue='association',
                        size='log_eflux', sizes=(20,150),size_norm=(-12,-10),
                        ax=ax1)
        ax1.legend(loc='upper right')

        target_df = df.query(f'association=="{target}"')
        assert len(target_df)>0, f'Failed to find target {target}'
        sns.scatterplot(target_df, x=x, y=y,  
                        hue_order=target_names, hue='prediction',
                        size='log_eflux', sizes=(20,150),size_norm=(-12,-10),
                        ax=ax2)
        ax2.legend(loc='upper right')
        
        fig.text(0.51, 0.5, '⇨', fontsize=50, ha='center')
        show(fig, caption=caption)
        
    def pairplot(self):
        show(f"""## Examine correlations among selected features
        """)
        show(super().pairplot(), caption="""A KDE "pairplot" of the chosen features
        """)

    def train_predict(self):
        show(f"""## Train, predict, assess results
        """)
        super().train_predict(show_confusion=True, hide=True)

    def show_sgus(self):
        dfq  = self.df.query('sgu==True')
        show(f"""## What about the  {len(dfq)} SGU sources?""")
        
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(9,4),
                gridspec_kw=dict(wspace=0.3))
        sns.countplot(dfq, y ='prediction', hue_order=self.mlspec.target_names,
                      ax=ax2)
        sns.countplot(dfq, y ='association', ax=ax1)
        show(fig, caption="""Left: initial association category; 
        Right: prediction assignments.
        """)
        show(f"""Conclusion: mostly pulsars!""")
    
        target_names = self.mlspec.target_names
        df = self.df.query('sgu==True')
        fig, ax = plt.subplots(figsize=(6,6))
        sns.scatterplot(df, x='log_epeak', y='curvature', 
                        hue_order=target_names, hue='prediction',
                        size='log_eflux', sizes=(10,200),size_norm=(-12,-10),
                        ax=ax )
        ax.set(xticks = [-1,0,1,2,3], title='SGU assignments')
        ax.legend(loc='upper right', fontsize=12);
        show(fig)
    
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

def main():
    self =  ML()
    self.outline()
    self.show_data()
    self.compare_variabilities()
    self.pairplot()
    self.classifier_evaluation()
    self.train_predict()
    self.show_prediction_association()
    self.curvature_epeak_flux()
    self.show_sgus()
        
main()