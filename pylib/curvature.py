from pylib.machine_learning import *

class Curvature(ML):
    def __init__(self, title='Spectral Curvature'):
        from utilities.catalogs import UWcat, Fermi4FGL
        super().__init__(title=title)
        self.fermicat = Fermi4FGL(); 
        self.uwcat = UWcat().set_index('jname')
        self.train_predict(show_confusion=False)

    def distributions(self, fignum=1):
        data = self.df
        show(r"""## Curvature distributions 
        The spectral curvature is defined as the negative of the second derivative of the SED distribution
        in log-log space. For the common log-parabola (LP) spectral function it is $2\beta$. Pulsars are fit
        with a power-law exponential form (PLEC). For these we evaluate the curvature at the reference energy. 
        """)
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(10,8), sharex=True, sharey=True,
                                    gridspec_kw=dict(hspace=0.1))
        ax1.set(ylim=(1,400))
        for ax in (ax1,ax2): ax.axvline(4/3, ls='--', color='0.5')
        hist_kw = dict(x='curvature', bins = np.linspace(-0.2,2, 42), element='step',
                    log_scale=(False, True), kde=True)
        sns.histplot(data[np.isin(data.association.values, self.mlspec.target_names) ], ax=ax1 ,
                    hue='association', **hist_kw)
        sns.histplot(data[data.association=='unid'], ax=ax2 , **hist_kw,  hue='prediction',)
        
        show(fig, fignum=fignum, caption="""Distribution of the spectral curvature, with KDE overlays. Upper panel:
            Associated sourcs, according to association type. Lower panel: ML predictions for the  unassociated
            sources. The dashed line, at ccurvature=4/3, represents the theoretical upper limit for
            synchroton curvature radiation.""")
        show(f"""An initial assumption might be that unassociated sources are one of the three major types, 
            which account for 95% of all associated sources. However, the shapes of the predicted distributions 
            are not consistent with a simple linear combination of the associated counterparts. This is 
            dramatically true for the pulsars, which we expect to be limited by the physical bound of 4/3.
            """)
        
    def measurments(self, fignum=None):
        show(f"""## Curvature measurment details
        ### Role of measurement error
        THere are two issues to consider:
        1. The unassociated sources tend to be weaker, unassociated perhaps because they were harder to detect 
        in other wavelengths--this means that statistical errors for spectral details are larger. For pulsars, larger error 
        circles make radio searches more difficult.
        2. Higher curvatures have intrinsically larger errors
        """)
        self.uw_measurements()
        self.fgl_measurements()
        self.compare_curvature()

    def uw_measurements(self, fignum=None):
        show(f"""### UW measurments
        """)
        
        df = self.df.copy().set_index('uw_name')
        df['d_unc'] = self.uwcat.errs.apply(lambda s: np.array(s[1:-1].split(), float)[2])
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(df,ax=ax, x='curvature', y=df.d_unc.clip(0,1));
        show(fig, fignum=fignum, caption='')
        y = df.d_unc
        show(f""" The UW fit procedure optimized curvatures independently for each source, limiting the value to 2. 
        There were some sources for which the fit was not done, or apparently failed.
        Total: {len(y)}; Bad errors: 0: {sum(y==0)}, >1: {sum(y>1.)}
        """)

    def fgl_measurements(self, fignum=None):
        show(r"""### 4FGL curvature errors
        Examine the log-parabola fit parameter $\beta$ for all sources, catalog fields `LP_Beta` and `unc_LP_Beta`. 
        Display the spectral curvature, $d=2\beta$.
        """)
        
        # list(filter(lambda s: 'LP_' in s , fcat.fitscols.names))
        fcat = self.fermicat.copy()
        
        fcat['d']     = 2*np.array([x for x in self.fermicat.field('LP_Beta')])
        fcat['d_unc'] = 2*np.array([x for x in self.fermicat.field('Unc_LP_Beta')])
        fig, ax = plt.subplots(figsize=(8,8))
        sns.scatterplot(fcat, ax=ax, x='d', y=fcat.d_unc.clip(0,1))
        show(fig, fignum=fignum, caption='')

    def fgl_pulsar_curvature(self, fignum=None):
        show(r"""### Comapare $\beta$ and $d$ for pulsars
        Since pulsars are fit to both PLEC and LP spectra, 
        """)

        fcat = self.fermicat.copy()
        size_kw = dict(size='log_eflux',sizes=(10,300))
        name='Exp_Index' #ExpFactorS'
        fcat[name]     = 2*np.array([x for x in self.fermicat.field('PLEC_'+name)])
        fcat[name+'_unc'] = 2*np.array([x for x in self.fermicat.field('Unc_PLEC_'+name)])
        fcat['log_eflux'] = np.log10(fcat.eflux)
        fcat['beta'] = np.array([x for x in self.fermicat.field('LP_Beta')])
        df = fcat[fcat.specfunc.apply(lambda x: x.__class__.__name__=="PLSuperExpCutoff4")]
        nfixed = sum(pd.isna(df.Exp_Index_unc))
        show(f"""Note: only {len(df)-nfixed} out of {len(df)} 
            pulsar fits have Exp_Index free: otherwise it is fixed to 4/3.""")
                    
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(10,10), sharex=True)
        for ax in (ax1,ax2):
            ax.axvline(4/3, ls='--', color='0.3')
        sns.scatterplot(df, ax=ax1, x=name, y=name+'_unc', **size_kw)
        sns.scatterplot(df, ax=ax2, x=name, y='beta', **size_kw)
        ax2.plot([0.4, 2.0], [0.2,1.0], ls='--', color='red')
        show(fig, fignum=fignum, caption="""Correlations of the PLEC curvature parameter `Exp_Index` with respect to
        its uncertainty (top), and the LP curvature `LP_Beta` (bottom). The special value 4/3 is shown on the x-axis, 
        and the expected correspondence to the LP curvature as a red dash line on the bottom.
        """)

    def compare_curvature(self, fignum=None):

        show(r"""## Compare UW and 4FGL curvatures
        For this comparison we determed the `fgl_comparison` parameter using a numerical second derivative
        of its preferred spectral function, evaluated at the reference energy. 
        This puts LP and PLEC on the same basis. (The UW curvature was determined the same way.)
        """)

        df = self.df.copy().set_index('uw_name')
        
        fig, axx = plt.subplots(ncols=2, figsize=(15,8), sharex=True, sharey=True,
                            gridspec_kw=dict(wspace=0.05))
        dfs = [df[np.isin(df.association.values, self.mlspec.target_names) ],
                df[df.association=='unid']]
        ticks = np.arange(0,1.6, 0.5)
        info=[]; lim=(-0.2,2)
        for df, ax in zip(dfs, axx):
            sns.kdeplot(df, ax=ax, x='curvature', y='fcat_curvature', hue='association');
            sns.scatterplot(df, ax=ax, x='curvature', y='fcat_curvature', hue='association', alpha=0.5);
            ax.set(ylim=(-0.2,1.6), xlim=(-0.2,2.1), xlabel='uw_curvature', xticks=ticks, yticks=ticks)
            ax.plot([0,1.5], [0,1.5], 'r')
            ax.axhline(4/3, ls='--', color='0.3')
            info.append((len(df), sum(df.fcat_curvature<0.01)))
            
        show(fig, fignum=fignum, caption="""Comparison between UW and 4FGL curvature values. Left panel: 
        the associated blazar or pulsar sources. Right panel: unassociated sources. The horizontal dashed line
        is at 4/3, the value used by the 4FGL beta-limiting prior. The red line corresponds to equality for the
        two measurements.
        """)
        show(f"""The points along the y-axis are the cases in which the preferred 4FGL spectrum is a power-law.
        this happens ({info[0][1]} / {info[0][0]},  {info[1][1]} / {info[1][0]}) of the time for the twe cases.
        For "ordinary" souces, left panel, the correspondence is reasonable, but for many unassociated sources the 4/3 
        4FGL "4/3" prior distorts the relationship.
            """)
