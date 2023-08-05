from pylib.machine_learning import *

class Curvature(ML):
    def __init__(self, title='Evidence for a new gamma-ray source class'):
        from utilities.catalogs import UWcat, Fermi4FGL

        # with capture_hide('setup output') as setup_output:
        super().__init__(title=title)
            # self.fermicat = Fermi4FGL(); 
            # self.uwcat = UWcat().set_index('jname')
        self.train_predict(show_confusion=False)

    def intro(self):
        show(f"""
            It is clear from the [ML studies](machine_learning.ipynb) that the spectral curvature plays an important role.
             In this section we study this variable in detail. We have been using the UW determination, so here we look at
             how it was measured for the UW all-sky analysis as well as the public 4FGL-DR4. 
             """)

    def distributions(self, fignum=1):
        data = self.df.copy()
        show(r"""## Curvature distributions 
        The spectral curvature is defined as the negative of the second derivative of the SED distribution
        in log-log space. For the common log-parabola (LP) spectral function it is $2\beta$. Pulsars are fit
        with a power-law exponential cutoff form (PLEC). For these we evaluate the curvature at the reference energy. 
        """)
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(10,8), sharex=True, sharey=True,
                                    gridspec_kw=dict(hspace=0.1))
        ax1.set(ylim=(1,400))
        for ax in (ax1,ax2): ax.axvline(4/3, ls='--', color='0.5')
        hist_kw = dict(x='curvature', bins = np.linspace(-0.2,2, 42), element='step',
                    log_scale=(False, True), kde=True)
        hue_kw = dict(hue='association', hue_order='bll fsrq psr'.split())
        sns.histplot(data[np.isin(data.association.values, self.mlspec.target_names) ], ax=ax1 ,
                    **hue_kw, **hist_kw)
        hue_kw.update(hue='prediction')
        sns.histplot(data[data.association=='unid'], ax=ax2 , **hist_kw,  **hue_kw)
        
        show(fig, fignum=fignum, caption="""Distribution of the spectral curvature, with KDE overlays. Upper panel:
            Associated sources, according to association type. Lower panel: ML predictions for the  unassociated
            sources. The dashed line, at curvature=4/3, represents the theoretical upper limit for
            synchrotron curvature radiation.""")
        show(f"""An assumption might be that unassociated sources are one of the three major types, 
            which account for 95% of all associated sources. However, the shapes of the predicted distributions 
            are not consistent with a simple linear combination of the associated counterparts. This is 
            dramatically true for the pulsars, which we expect to be limited by the physical bound of 4/3.
            The presence of a new source type calls the assignment into question. In the following map, we show 
            all unid sources with curvature>4/3. 
            """)
        self.high_curv_pos(fignum=fignum+1 if fignum is not None else None )
        show(f"""The galactic positions and strong correlation of flux with latitude makes it clear that these are mostly 
             of galactic origin. 
             """)
        
    def high_curv_pos(self, fignum=None):  
        df = self.df.query('curvature>1.33333 &  nbb<4')
        unid = df.association=='unid' 
        psr = df.association=='psr'                  
        show(f"""### Positions of {sum(unid)} curvature>4/3 unassociated sources.
        Since only {sum(psr)} of the actual pulsars satisfy this, there should be few undetected pulsars.
        """)
        self.show_positions(df[unid], fignum=fignum, 
                            caption="""Aitoff display of the selected source positions.
        """)

    def measurements(self, fignum=None):
        show(f"""## Curvature measurement details
        ### Role of measurement error
        There are two issues to consider:
        1. The unassociated sources tend to be weaker, unassociated perhaps because they were harder to detect 
        in other wavelengths--this means that statistical errors for spectral details are larger. For pulsars, larger error 
        circles make radio searches more difficult.
        2. Higher curvatures have intrinsically larger errors
        """)

        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(14,7), sharex=True, sharey=True,
                                    gridspec_kw=dict(wspace=0.05))
        show(f""" ### The UW and 4FGL measurement uncertainties""") 
        self.uw_measurements(ax=ax1)
        self.fgl_measurements(ax=ax2)
        ax1.set(ylabel='Curvature uncertainty')

        show(fig, fignum=fignum,
             caption="""Scatter plots of the curvature and its statistical error, as measured in the uw1410 dataset on 
        the left, and 4FGL-DR4 on the right.
            """)
        self.compare_curvature(fignum=None if fignum is None else fignum+1)
        self.fgl_pulsar_curvature(fignum=None if fignum is None else fignum+2)

    def uw_measurements(self, ax=None, fignum=None):
        
        df = self.df.copy().set_index('uw_name')
        df['d_unc'] = 2*self.uwcat.errs.apply(lambda s: np.array(s[1:-1].split(), float)[2])
        fig, ax1 = plt.subplots(figsize=(8,6)) if ax is None else (ax.figure, ax)
        df['log TS'] = np.log10(df.ts)
        sns.scatterplot(df,ax=ax1, x='curvature', y=df.d_unc.clip(0,1.5),
                    size='log TS', sizes=(10,200));
        if ax1 is None: show(fig, fignum=fignum, caption='')
        y = df.d_unc
        show(f""" * UW: The fit procedure optimized curvatures individually for each source, limiting the value to 2. 
        There were some sources for which the fit was not done, or apparently failed.
        Of the {len(y)} sources there were {sum(y==0)} with the error set to zero and  {sum(y>1.)} set to 2. 
        """)
    
    def fgl_measurements(self, ax=None, fignum=None):
        show(r"""* 4FGL: The log-parabola curvature parameter $\beta$ (which is 1/2 the spectral curvature) 
        was optimized along with the other spectral parameters, but, new for DR4, the likelihood included a
        prior Gaussian with mean 0.1 and width 0.3. So two sigmas is at 0.7, or curvature=1.4, suppressing 
        high curvatures as designed.
        See Section 3.4 of the [4FGL-DR4 paper]](https://arxiv.org/pdf/2307.12546.pdf))

        """)        
        fcat = self.fermicat.copy()
        fcat['curvature']     = 2*np.array([x for x in self.fermicat.field('LP_Beta')])
        fcat['d_unc'] = 2*np.array([x for x in self.fermicat.field('Unc_LP_Beta')])
        fig, ax1 = plt.subplots(figsize=(8,8)) if ax is None else (ax.figure, ax)
        sns.scatterplot(fcat, ax=ax1, x='curvature', y=fcat.d_unc.clip(0,1))
        if ax is None: show(fig, fignum=fignum, caption='')

    def fgl_pulsar_curvature(self, fignum=None):
        show(r"""### Compare $\beta$ and $d$ for pulsars
        Since pulsars are fit to both PLEC and LP spectra, this is a check on
        the correspondence of the respective curvature parameters.
        """)
        fcat = self.fermicat.copy()

        cvar = lambda a: self.fermicat.field('PLEC_'+a).astype(float)
        cvar_unc = lambda a: self.fermicat.field('Unc_PLEC_'+a).astype(float)
        plec_names = 'Flux_Density IndexS ExpfactorS Exp_Index'.split() # for N0, gamma, d, b
        N0, gamma, d, b = [cvar(fn) for fn in plec_names]
        fcat['d']=d
        fcat['d_unc'] = cvar_unc(plec_names[2])
        fcat['dp'] = d + b*(2-gamma) # 3PC EQ. (22)
        fcat['beta'] = self.fermicat.field('LP_Beta').astype(float)

        df = fcat[fcat.specfunc.apply(lambda x: x.__class__.__name__=="PLSuperExpCutoff4")]
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=(10,10), sharex=True,
                            gridspec_kw=dict(hspace=0.05))

        size_kw = dict(size='log_eflux',sizes=(10,300))
        sns.scatterplot(df, ax=ax1, x='d', y='d_unc',)# **size_kw)
        sns.scatterplot(df, ax=ax2, x='d',  y='beta', )#**size_kw)
        ax2.plot([0, 2.0], [0,1.0], ls='--', color='red')
        ax2.set(xlim=(0, 1.6), ylim=(0,0.8),yticks=np.arange(0,0.9, 0.2))

        for ax in (ax1,ax2): ax.axvline(4/3, ls='--', color="0.6")
        show(fig, fignum=fignum, caption=r"""Correlations of the PLEC curvature parameter $d$
              (`PLEC_ExpFactorS` in the FITS file) with respect to
        its uncertainty (top), and the LP curvature beta (`LP_Beta`) (bottom). The expected correspondence to the 
        LP curvature $\beta$ is shown as a red dash line on the bottom, and the theoretical upper limit
        for synchrotron curvature radiation at 4/3 as a vertical dashed line.
        """)

    def compare_curvature(self, fignum=None):

        show(r"""### Compare UW and 4FGL curvatures
        For this comparison we determined the `fgl_comparison` parameter using a numerical second derivative
        of its preferred spectral function, evaluated at the reference energy. 
        This puts LP and PLEC on the same basis. (The UW curvature was determined the same way.)
        """)

        df = self.df.copy().set_index('uw_name')
        
        fig, axx = plt.subplots(ncols=2, figsize=(14,8), sharex=True, sharey=True,
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
            ax.axvline(4/3, ls='--', color='0.3')
            info.append((len(df), sum(df.fcat_curvature<0.01)))
            
        show(fig, fignum=fignum, caption="""Comparison between UW and 4FGL curvature values. Left panel: 
        the associated blazar or pulsar sources. Right panel: unassociated sources.
        The dashed lines mark the curvature radiation limit, and 
        the red line corresponds to equality for the two measurements.
        """)
        show(f"""The points along the y-axis are the cases in which the preferred 4FGL spectrum is a power-law.
        this happens ({info[0][1]} / {info[0][0]},  {info[1][1]} / {info[1][0]}) of the time for the two cases.
        For "ordinary" sources, left panel, the correspondence is reasonable, but for many unassociated sources the 4/3 
        4FGL $0.2 \\pm 0.6$ prior distorts the relationship.
            """)

    def latitude_estimate(self, fignum=None):

        show("""## Planar/Polar unid estimate

        The galactic component of the UNIDs revealed by the high-curvature analysis suggests the presence of many hundreds of
        sources of a previously unknown type.

        Here we make a crude estimate of how many there might be by using latitudes, comparing the 
             planar ($|b|<30^\circ$) and polar populations.
        """)
        data = self.df.copy()
        data['latitudes'] = data.abs_sin_b.apply(
            lambda x: 'planar' if x<0.5  else 'polar')
        def srctype(src):
            if src.association in 'bll fsrq bcu'.split(): return 'blazar'
            if src.association =='psr': return 'psr'
            return 'unid'
        data['source_type'] = data.apply(srctype, axis=1)
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(data, x='abs_sin_b', ax=ax, bins=40, hue='source_type', element='step')
        ax.set(xlabel=r'$|\sin(b)|$', xlim=(0,1))
        show(fig, fignum=fignum, caption="""Histograms of $|\sin(b)|$ showing the major source components blazar, pulsar, and
        unassociated.""")
        
        show(f"""Assuming that all high-latitude UNID's are blazars, estimate the galactic component.
        """)
        gb = data.groupby(['source_type', 'latitudes']).size(); gb.name='count'
        show(gb)
        a,b = gb.loc['blazar']
        c,d = gb.loc['unid']
        show(f"""From the table, we estimate that the number of undetected blazars in the planar region is {b-a}.
        Then, assuming that the distribution of blazars in the unid is uniform, the net galactic 
        unids in planar region is {c-d -(b-a)}.
        This would also contain undetected pulsars.""")

    def sed_plots(self, fignum=None, ncols=15, height=0.5):
        show(f"""## "Golden" HCU SED plots, sorted by TS""")
        df = self.df.query('curvature>1.33333 & nbb<4 & association=="unid"').copy()
        df.index.name = '4FGL-DR4'
        df['Fp'] = 10**df.log_fpeak
        df['Ep'] = 10**df.log_epeak
        cols = 'ts r95 glat glon Fp Ep curvature nbb sgu uw_name'.split()
        dfx = df[cols].sort_values('ts',ascending=False)
        sedplotgrid(dfx[cols], fignum=None, ncols=ncols, height=height)
        show(f"""There are {sum(df.sgu==True)} SGUs in this list.""")

    def to_csv(self, filename='files/hcu_candidates.csv'):
        df = self.df.query('curvature>1.33333 &  nbb<4 & association=="unid"')
        df.to_csv(filename, float_format='%.3f')
        show(f"""Wrote (golden) HCU sources, sorted by TS, to `{filename}` with {len(df)} entries.""")

    def venn(self,ax=None, ts_cut=None):
        """ Plot shown in talk."""
        from matplotlib_venn import venn3
        fig, ax =plt.subplots(figsize=(6,6)) if ax is None else (ax.figure, ax)
        q = 'nbb<4'
        if ts_cut is not None: q += ' & ts>ts_cut'  
        all_df = self.df.query(q)
        all=set(all_df.index)
        unid = set(all_df.query('association=="unid"').index)
        hcu = set(all_df.query('curvature>1.333').index)
        venn3([all, unid, hcu], (f'TS>{ts_cut}' if ts_cut is not None else 'Constant', 'UNID', 'High Curvature'),
            ax=ax, alpha=0.6, set_colors='g r violet'.split());
        show(fig)

    def summary(self):
        show("""## Summary
            There are two strong indicators that the Fermi-LAT data contain many hundreds of an unknown 
            new galactic source of gamma-rays, easily more than the ~300 gamma-ray pulsars
            * Figure 1 shows that ordinary sources are consistent with the monoenergetic curvature radiation spectrum
            limit of 4/3, while many of the unid sources do not
            * Figure 2 makes if very clear that there is a correlation between high curvature and galactic position.

            The study of the actual measurements of curvature, Figures 3 and 4, show that the fit procedures for the UW spectral fits 
            used here differ in a dramatic way from what was used for the 4FGL-DR4 catalog: The latter fits were engineered 
            to suppress this effect.
             
            A reality check, the latitude estimate accompanying Figure 6 of 743 galactic unids appears consistent. 

            Further steps to validate and refine this:
            * Careful gtlike LP fits to a set of these, without the catalog's curvature constraint of course
            * Referring to Figure 1, there appear to be very many down to lower curvatures. With the ML predictions and the source 
            positions as a guide, it should be possible to recover them, and more generally, account for all unids at least statistically.
            
            """)
        
import sys
if len(sys.argv)>1:
    if sys.argv[1]=='main':    
            self = Curvature()
            self.intro()
            self.distributions(fignum=1)
            self.measurements(fignum=3)
            self.latitude_estimate(fignum=6)
            self.summary()
    elif sys.argv[1]=='setup':
        self = Curvature()