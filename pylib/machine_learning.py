from pylib.fermi_sources import *
sns.set_theme('notebook', font_scale = 1.5)
if 'dark' in sys.argv:
    plt.style.use('dark_background') #s
    plt.rcParams['grid.color']='0.5'
    dark_mode=True

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
        self.df.loc[:,'fcat_d']= 2 * fcat.specfunc.apply(lambda f: f.d())

       
    def outline(self):
        show(f"""
            ## Outline
            **Goal: use predictive artificial intelligence to classify source types of the unid's**
            
            Procedure:
            * Choose the standard `scikit-learn` ML implementation 
            * Choose "features"
            * Evaluate classifier options, select one
            * Validate, perhaps adjust feature set
            * Apply to the UNID's 
            """)
        
    def show_prediction_association(self, fignum=None, caption=''):
        show(f"""## Predictions
        """)

        def simple_pivot(df, x='prediction', y= 'association'):        
            ret =df.groupby([x,y]).size().reset_index().pivot(
                columns=x, index=y, values=0)
            return ret.reindex(index='bll fsrq psr bcu unk spp other unid'.split())
            
            
        df_plot = simple_pivot(self.df)
        fig, ax = plt.subplots()
        ax = df_plot.plot.barh(stacked=True, ax=ax, color=self.palette)
        ax.invert_yaxis()
        ax.set(title='Prediction counts', ylabel='Association type')
        ax.legend(bbox_to_anchor=(0.78,0.75), loc='lower left', frameon=True )
        show(fig, fignum=fignum, caption=caption)
        show(f"""Notes:
        * The target is the UNIDs, but applied to all
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
        sns.scatterplot(dfq, x='log_epeak', y='d', 
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
    
    def d_epeak_flux(self, fignum=None):
        show(f"""### Curvature vs $E_p$: compare training and unid sets""")


        self.scatter_train_predict(x='log_epeak', y='d', fignum=fignum,
                caption=f"""Curvature vs $E_p$ for the training set on the
            left, the unid on the right.""",
                            **epeak_kw('x'),
                            yticks=[0,0.5,1,1.5,2]
                            )
        show(f"""Note that the d distribution is shifted to higher values for the unid 
        data.
        """)

        show(f"""### Curvature vs. $F_p$
            Check the dependence of the curvarure $d$ on the peak flux.
            """)

        self.scatter_train_predict( x='log_fpeak', y='d',fignum=fignum+1 if fignum is not None else None,
                caption=f"""Curvature vs $F_p$ for associated sources on the
            left, the unid on the right.""",
                        **fpeak_kw('x'),
                          yticks=[0,0.5,1,1.5,2])
    
   
    def show_notes(self):
        show("""
            ## Notes, todos:
            * Reexamine feature set using random forest importance measures
            * Perhaps expand the "other" category, e.g. SNRs
            * Check some of the individual ones brought up here
        """)

def main():

    fn = FigNum()
    # sns.set_context('talk')
    self =  ML(mlspec=MLspec(
        features='log_var log_fpeak log_epeak d'.split()))
    self.outline()
    self.show_data()
    # self.compare_variabilities(fignum=fn.next)
    self.pairplot(fignum=fn.next)
    self.classifier_evaluation()
    self.train_predict(show_confusion=True)
    self.show_prediction_association(fignum=fn.next)
    self.d_epeak_flux(fignum=fn.next); fn.next
    # self.show_sgus(fignum=fn.next)
    self.show_notes()

import sys
if len(sys.argv)>1:
    if sys.argv[1]=='main':
        main()
    elif sys.argv[1]=='setup':
        self =  ML(title='ML doc development')
        self.train_predict()

    
