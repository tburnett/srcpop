from pylib.curvature import *
from pylib.kde import KDE, Gaussian_kde

   
def spectral_cut(df, elim=(0.4,4), fpmax=20, cmin=0.5):
    show("""
         """)
    return df[ (df.curvature>cmin)  
            & (df.log_epeak > np.log10(elim[0]) )
            & (df.log_epeak < np.log10(elim[1]) )
            & (df.log_fpeak < np.log10(fpmax)   ) 
            ]
   
  
class Paper(Curvature):
    def __init__(self,title='Observation of a new gamma-ray source class', **kwargs):
        super().__init__(title=title,  **kwargs)

        df = self.df#[self.df.nbb<4].copy()
        df['log TS'] = np.log10(df.ts)

        def plike(rec):
            class1 = rec.class1.lower() if not pd.isna(rec.class1) else np.nan
            if not pd.isna(class1) and class1 in ('msp','psr'): 
                return dict(msp='MSP', psr='young')[class1]
            if rec.association=='unid': return 'UNID-'+ rec.prediction.upper()
            if class1=='glc': return 'glc'
            return rec.association
        df['source type']= df.apply(plike, axis=1)
        self.df=df

    def abstract(self):
        show(r"""## Abstract
            The most recent source catalog from the Large Area Telescope (LAT) instrument on
              _Fermi_, covering 14 years, contains 7190 non-transient point sources,
              of which 2371 have no co-location association with objects observed in other 
             wavelengths to provide confirmation of their identities. This has been a feature 
             of all catalogs since the first 8-month release in which 161 out of 630 were 
             unassociated. The unassociated sources (UNID) break down into roughly equal 
             Galactic and extra-Galactic populations. The origin of the latter is  accounted 
             for by the non-uniform coverage and sensitivity limit of the counterpart catalogs. 
             To examine the Galactic component of the UNID we apply a 
             predictive artificial intelligence technique using as features the 
             parameters of the peak in the spectral energy distribution, and the 
             variability measure. The UNID sources predicted to be pulsars have 
             significant spectral curvature, with peak energies in the range seen 
             in the pulsars used for training, and are clearly Galactic in spatial distribution. 
             But the distribution of curvatures and spectral peak energies is quite unlike that 
             seen for pulsars, with curvatures extending well above those expected for pulsars.
             We define a selection in the range
              of spectral parameters which almost all of these satisfy,  resulting in 617 total.
            Applying the same selection to each of the associated source classes we see that none of the 
             resulting spatial distributions are consistent, the closest being 
             millisecond pulsars (94) and Galactic clusters (19). 
             While there may be some undetected pulsars in this set, we conclude that the
             majority must represent a new class of gamma-ray emitting sources.
        """)

    def examine_cuts(self, elim=(0.4,4), fpmax=20, cmin=0.5):
        all = self.df 
        spcut = spectral_cut(all, elim=elim, fpmax=fpmax)
        # low_b = spcut[spcut.log_fpeak<1] 
        c1 = all.groupby('source type').size(); c1.name='All'
        c2= spcut.groupby('source type').size(); c2.name='Spectral cut'

        t = pd.DataFrame([c1,c2]).reindex(
            columns='MSP young bll fsrq bcu glc other UNID-PSR UNID-BLL UNID-FSRQ'.split())
        
        t['Total'] = np.sum(t.to_numpy(), axis=1)
        return t
    
    def d_vs_ep(self, ax=None, hue_order='MSP young UNID-PSR'.split(), legend='auto', plot_limit=False):
        df = self.df.query('log_epeak>-0.75 & curvature>0.2' )
        df = df[df['source type'].apply(lambda st: st in hue_order)]
        hue_kw = dict(hue='source type', hue_order=hue_order)
        hue_order='bll fsrq psr'.split()
        size_kw =dict(size='log TS', sizes=(10,400))
        # uv = self.hcu_cut
        fig, ax = plt.subplots(figsize=(8,6)) if ax is None else (ax.figure, ax)
        size_kw =dict(size='log TS', sizes=(10,200))
        data_kw= dict(data=df, x='log_epeak', y=np.log10(df.curvature))
        sns.scatterplot(**data_kw, ax=ax, #x='log_epeak', y=np.log10(df.curvature), legend=legend,
                    style=hue_kw['hue'],
                    markers={'UNID-PSR':'o', 'MSP':'s', 'glc':'d'},
                    palette={'UNID-PSR':'cornflowerblue', 'MSP':'red', 'glc':'seagreen'},
                    **hue_kw, #hue='pulsar type', hue_order='MSP predicted'.split(),
                    **size_kw)
        hue_kw.update(hue_order=['MSP'])
        sns.kdeplot(**data_kw, **hue_kw,                
                    palette={'UNID-PSR':'cornflowerblue', 'MSP':'red', 'glc':'seagreen'},)
        
        xticks = [0.25,0.5,1,2,4]
        ax.set(xticks=np.log10(xticks), xticklabels=[f'{x}' for x in xticks] , 
            xlabel='$E_p$ (GeV)', xlim=np.log10((0.2,8)), 
            ylabel='$\log(d_p)$', ylim=(-.70,0.35), # yticks=np.arange(0,2.1,0.5),
            )
        if plot_limit:
            ax.axhline(4/3, ls='--', color='red')
        # loge_pts = [-0.77,0.6]
        # ax.plot(loge_pts, uv(loge_pts),  ls=':', color='k')
        if legend=='auto': plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.88,0.4));
        return fig

    def show_dp_vs_ep(self, fignum):

        df = self.df.copy()
        # hc = HCUcut()
        # df['perp'] = df.apply(lambda row: HCUcut().perp((row.log_epeak,row.curvature)), axis=1)
        
        df['log TS'] = np.log10(df.ts)
        show(f"""## Analysis of curvature vs. peak energy
        For MSPs, the spectral curvature is  correlated with the
        peak energy, a consequence of $\dot{{E}}$ evolution. 
        Specifically it was shown that $E_p = 1.1\ \mathrm{{GeV}} (d_p/0.46)^{{1.33}}$ within 30%. [3PC paper]. 

        In Figure {fignum} we show such a plot for MSPs, with the UNID-PSR and glc subsets overlayed
        for comparison. 
        """)
        fig, ax =plt.subplots(figsize=(10,6))
        fig = self.d_vs_ep( ax=ax, hue_order='UNID-PSR MSP glc'.split(),)
        # show(fig, hue_order='UNID-PSR MSP glc'.split(), ax=ax), 
        #      fignum=1, caption="""Curvature $d$ vs.$E_p$. The dotted line
        #      is an arbitrary reference to separate MSPs.
        #      """)
        f = lambda x:  0.75 * ( x - np.log10(1.1)) + np.log10(0.46) 
        xx = np.array([-1,1])
        ax.plot(xx, f(xx), 'k'); #ax.plot( np.log10(1.1), np.log10(0.46), 'd', markersize=30);
        ax.axhline(np.log10(4/3), ls='--', color='k');
        show(fig, fignum=fignum, caption=f"""Dependence of the spectral curvature on peak energy.
        The horizontal dashed line is at 4/3, the value for the curvature radiation from monoenergetic 
        electrons, while the inclined solid line corresponds to a study of correlation of $d_p$ and $E_p$ 
        described in the text. The contours for a KDE estimation of the MSP density are also shown.
        """)

        # show(f"""
        # #### Using KDE to estimate MSP content of UNID-PSR
        # Using the KDE function derived from the $E_p$ vs. $d_p$, distributions, consider 
        # its distribution over the MSP and UNID-PSR subsets. Assuming that a component
        # of the UNID-PSR sources are undetected MSPs which would have a similar distribution,
        # we see from Figure {fignum+1} that the largest possible size is about four times the number currently
        # detected well under the latitude estimate of about the same number.
        # """)

        # data= self.df[(df.log_epeak>-0.5) & (df.curvature>0.2)].copy()
        # source_type = data['source type']
        # data['log_d'] = np.log10(data.curvature.clip(1e-3,10))
        # msp_data = data[source_type=='MSP']  
        # x,y = 'log_epeak log_d'.split()
        # msp_kde = KDE(msp_data,  x=x, y=y, )

        
        # # msp_kde.plot()
        
        # msp_cdf = msp_kde(msp_kde.dfxy.to_numpy().T)
        # unid_data = data[source_type=='UNID-PSR' ]  
        # unid_cdf = msp_kde(unid_data.loc[:,(x,y)].to_numpy().T)
        # hkw = dict(bins=np.linspace(0,1,11), histtype='step', density=False, lw=2 )
        # fig, ax = plt.subplots(figsize=(6,4))
        # ax.hist(msp_cdf, **hkw,  label='MSP');
        # ax.hist(unid_cdf, **hkw,   label='UNID-PSR')
        # ax.set(ylabel='Counts', xlabel = 'KDE probability', 
        #     xlim=(0,1), xticks=np.arange(0,1.1,0.25))
        # ax.legend(); 
        # show(fig, fignum=fignum+1, caption=f"""Histogram of the KDE probability distribution 
        # derived from MSPs applied to MSPs and UNID-PSRs.
        # """)

    def peak_position(self):
        hue_order='bll fsrq psr'.split()
        size_kw =dict(size='log TS', sizes=(10,200))
        hcu = df.query('association=="unid" & curvature>0.4').copy(); len(hcu)
        
        show(f"""### Peak position""")
        fig, (ax1,ax2) = plt.subplots(ncols=2,figsize=(14,8), sharex=True,sharey=True,
                                    gridspec_kw=dict(wspace=0.05))
        
        known= df[df.association.apply(lambda x: x in self.mlspec.target_names)]
        xy = dict( x='log_epeak', y='log_fpeak')
        sns.scatterplot(known, ax=ax1, **xy,  
                        hue='association', hue_order=hue_order, **size_kw)
        sns.scatterplot(hcu,ax=ax2, **xy,
                        hue='prediction', hue_order=hue_order,  **size_kw);
        show(fig)

    def fp_vs_sinb(self, ax=None, cut='HCU cut',  hue_order='MSP young UNID-PSR'.split(), legend='auto'):
    
        df = self.df if cut is None else self.df[self.df[cut]]

        hue_kw = dict(hue='source type', hue_order=hue_order)
        size_kw =dict(size='log_ts', sizes=(10,100))
        fig, ax = plt.subplots(figsize=(8,5)) if ax is None else (ax.figure, ax)

        sns.scatterplot(df, ax=ax, x='abs_sin_b', y='log_fpeak', legend=legend,
                        **hue_kw,**size_kw)
        ax.set(xlim=(0,1.), ylim=(-1.5,1.5), ylabel='$F_p$ (eV s-1 cm-2)',
            yticks=[-1,0,1], yticklabels='0.1 1 10'.split(),
            xlabel= r'$|\sin(b)|$')
        if legend=='auto': plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.88,0.4));
        return fig
        
    def show_fp_vs_sinb(self, fignum=100, cut='HCU cut', hue_order='MSP young UNID-PSR'.split()):
        
        hcu = self.df[self.df['HCU cut']]
        hcu_cut=hcu[hcu['source type']=='UNID-PSR']
        show(f""" ### $F_p$ vs. $|\\sin(b)|$              
            Selecting sources above the black line in Figure {fignum-1},
            results in {len(hcu_cut)} HCU sources. In Figure {fignum} we see 
            the resulting distribution in peak flux $F_p$ vs. $|\\sin(b)|$ for known
            pulsars and HCU candidates.
        """)

        fig = self.fp_vs_sinb( cut=cut, hue_order=hue_order)
        show(fig, fignum=fignum, caption="""Scatter plot peak flux vs latitude for 
            known pulsars and pulsar-like UNIDS.
        """)
        show("""
            """)
            
    def show_venn(self):

        all = self.df.query('nbb<4')
        unid = all.query('association=="unid"')
        hcu = unid.query('curvature>1.333')
        len(hcu)
        
        def venn(self,ax=None, ts_cut=None):
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
        fig, ax = plt.subplots()
        venn(self,ax, None)
        show(fig)

    def source_table(self):
        all = self.df 
        hcu = all[all['HCU cut']];
        c1 = all.groupby('source type').size(); c1.name='All'
        c2 = hcu.groupby('source type').size(); c2.name='HCU cut'
        t = pd.DataFrame([c1,c2]).reindex(
            columns='MSP young bll fsrq bcu other UNID-PSR UNID-BLL UNID-FSRQ '.split())
        t['Total'] = np.sum(t.to_numpy(), axis=1)
        show(t)
            
    def show_ecdf_abs_sin_b(self, fignum=None):
        show(f"""## Empirical cumulative distribution functions

             
        """)
        fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(12,5), sharex=True, sharey=True,
                                    gridspec_kw=dict(wspace=0.05))
        sns.ecdfplot(self.df, ax=ax1, x='abs_sin_b', hue = 'source type',
                    hue_order='MSP young bll fsrq bcu other'.split());

        sns.ecdfplot(self.df, ax=ax2, x='abs_sin_b', hue = 'source type',
                    hue_order='UNID-BLL  UNID-FSRQ UNID-PSR'.split());
        ax1.set(xlim=(0,1), ylim=(0,1), aspect=1)
        for ax in (ax1,ax2):
            ax.set(xlabel=f'$|\sin(b)|$')
            ax.plot((0,1), (0,1), ':b')
        show(fig, fignum=fignum, caption=f"""Empirical cumulative distribution functions
        for $|\\sin(b)|$. Left panel: associated sources; Right panel: unassociated sources.
        The dashed line corresponds to an isotropic distribution
        """)

    def show_ait_plots(self):
        show(f"""## AIT plots for each source type""")
        df = self.df
        stl = df.groupby('source type').size()
        for name, cnt in stl.items():
            xdf = df[df['source type']==name]
        self.show_positions(xdf, figsize=(8,8), title=f'{name} ({cnt})')
    
    def to_csv(self, dfx, filename = 'files/hcu_candidates.csv'):
        dfx.index.name = '4FGL-DR4'
        dfx.to_csv(filename, float_format='%.3f')
        show(f"""Wrote HCU  sources, sorted by TS, to `{filename}` with {len(dfx)} entries.""")

    def fp_vs_abs_b(self,  binsize=2.5, ax=None):
        data = spectral_cut(
            self.df[self.df['source type']=='UNID-PSR'] ).query('log_fpeak<1.5')
        
        size_kw =dict(size='log_ts', sizes=(10,100))
        fig, ax = plt.subplots(figsize=(8,5)) if ax is None else (ax.figure, ax)
        x=np.abs(data.glat); y=data.log_fpeak
        sns.scatterplot(data, ax=ax, x=x, y=y, c='cornflowerblue' ) #x=np.abs(data.glat), y='log_fpeak'  )
        # have to offset x to plot in center of bin 
        bs = binsize
        sns.regplot(x=x+bs/2, y=y, x_bins=np.arange(bs/2,26,bs), 
                fit_reg=None, scatter=True,marker='s', color='maroon',
            line_kws=dict(markersize=50,ls='--'));
        
        ax.set(xlim=(0,0.4), ylim=(-1.5,1.5), 
            ylabel=r'$F_p\ \mathrm{(eV\ s^{-1}\ cm^{-2})}$',
            yticks=[-1,0,1], yticklabels='0.1 1 10'.split(),
            xticks = np.arange(0,30, 5),
            xlabel= r'$|b|$ (deg)');
        return fig

    def show_sin_b(self,fignum):
        show(rf"""### Distributions in $|\sin(b)|$
        WE look at distributions in $|\sin(b)|$ to estimate the contribution of undetected MSPs to 
        UNID-PSR population.
        """)
        hue_kw = dict( hue='source type',
                    hue_order='UNID-PSR MSP glc '.split())
        size_kw = dict(size='log TS', sizes=(10,200))
        data_kw = dict(data=self.df, x='abs_sin_b', )
        fig, ax = plt.subplots(figsize=(8,4))
        sns.histplot(**data_kw,**hue_kw, ax=ax, element='step',
                    bins=np.arange(0,1.1,0.1), log_scale=(False,True))
        ax.set(xlim=(0,1), xlabel=r'$|\sin(b)|$', xticks = np.arange(0,1.1,0.25))
        # kde= sns.kdeplot(**data_kw,**hue_kw,ax=ax)
        # ax.set(ylim=(-1,1), xlim=(0,1), xlabel=r'$|\sin(b)|$',
        #       yticks=np.arange(-1,1.1,0.5))
        # plt.legend(fontsize=12, bbox_to_anchor=(0.85,0.6) );
        show(fig, fignum=fignum, caption="""Histogram of $|\sin(b)|$.
        """)
        show(f"""As seen in Figure {fignum}, the number of undetected MSPs must be a large
            fraction of 
            all UNID-PSR sources above about ${np.degrees(np.arcsin(0.4)):.0f}^\circ$, in number equal to the
            number of detected ones.
        """)
    
    def ml_summary(self):
        show(f"""## ML step summary
        * Training set: pulsars (MSP + young), BL Lacs, FSRQs
        * Features:  spectral peak parameters ( energy, flux, curvature), variability, energy flux
        * predictions: UNID -> (UNID-PSR, UNID-BLL, UNID-FSRQ)
        """)
    
    def show_examine_cuts(self, elim=(0.4,4), fpmax=20, cmin=0.5):
        show(f"""### Application of the "spectral cut" to the source classes
            We subdivide the sources into the following ten subsets called "source types"
            below<br>
            
            Associated: 
            * pulsars: MSP, young
            * blazars: bll, fsrq, and bcu
            * Galactic clusters: glc
            * others: Everything else<br>
            
            Unassociated: UNID-PSR, UNID-FSRQ and UNID-BLL according to ML prediction.
        
            Preliminary spectral selection cuts are: 
            * $d_p$ >{cmin}
            * {elim[0]}< $E_p$ < {elim[1]} GeV  
            * $F_p$ < {fpmax} eV s-1 cm-2

            The resulting counts:
            """)
        show(self.examine_cuts(elim, fpmax, cmin))

    def show_prelim_dp_vs_ep(self, fignum=1):
        show(f"""## Curvature vs. $E_p$ for UNID
        The ML training depended strongly on the two spectral variables
        peak curvature $d_p$ and energy $E_p$. Here we look at these 
        variables for the predictions of the UNID sources.    
        """)
        fig, ax = plt.subplots(figsize=(10,8))
        sns.scatterplot(self.df, ax=ax,x='log_epeak', y='curvature', hue='source type', 
                        hue_order='UNID-BLL UNID-FSRQ UNID-PSR'.split(),
                    size='log TS', sizes=(20,200),)# alpha=0.4)
        x = np.log10([0.4, 0.4, 4, 4])
        y = [2,0.5,0.5,2]
        ax.plot(x, y, ls=':', color='k');
        ax.set(xlabel='$E_p$ (GeV)',xticks=[-1,0,1,2], 
                xticklabels='0.1 1 10 100'.split(), xlim=(-1.2,2.5),
                ylabel='$d_p$',)
        plt.legend(fontsize=12)
        show(fig, fignum=fignum, caption="""Curvature vs. peak energy. 
            Colors identify the ML prediction.
            The dotted line encompasses most of the PSR prediction, defines the "spectral cut".
        """)
        self.show_examine_cuts()

    def show_skymaps(self,fignum=2):
        show(f""" ## Aitoff Skymaps
        Here we show three sets of skymaps. First, for the three UNID subsets 
        to assess the Galactic content of each; then the same three following the 
        spectral cut, and finally four subsets of the associated sources
        to look for candidates to account the Galactic sources accounting for 
        the Galactic unassociated.

        """)
        show(f"""#### UNID """)
        show( ait_plots(self.df, 
                        hue='source type', hue_order='UNID-FSRQ UNID-BLL UNID-PSR'.split(),
                    )  ,fignum=fignum, caption='Aitoff plots')
        show(f"""#### UNID after spectral cut""")
        show( ait_plots(spectral_cut(self.df), 
                        hue='source type', hue_order='UNID-FSRQ UNID-BLL UNID-PSR'.split(),
                    )  ,fignum=fignum+1, caption='Aitoff plots')

        show(f"""#### Associated after spectral cut """)
        df = self.df.copy()
        def sort_ided(st):
            if st in 'MSP glc'.split(): return st
            if st in 'bcu bll fsrq'.split(): return 'blazar'
            return 'the rest'
        df['st'] = df['source type'].apply(sort_ided)
        show(
            ait_plots(spectral_cut(df.query('association!="unid"')), 
                hue='st', hue_order=('MSP', 'glc', 'blazar','the rest') ),
            fignum=fignum+2, caption='Aitoff plots')
        show(f"""### Conclusions:
        * Figure {fignum}: The UNID-PSR were selected as such according to 
        the spectral and variability only: Clearly they are dominantly Galactic.
        * Figure {fignum+1}: The specific spectral cut removes most of the blazar
        types, including an apparent Galactic component while hardly changing the
        UNID-PSR sources.
        * Figure {fignum+2}: Of the known Galactic classes with a multi-degree 
        scale height, only MSP and glc are candidates. 
        """)    

    def show_fp_vs_abs_b(self, fignum=7):    

        show(f"""## Flux vs latitude for selected UNID-PSR sources
        Here we select the selected UNID-PSR sources seen in Figure 3 
        and look at the peak flux.
        """)
        show(self.fp_vs_abs_b(), fignum=fignum, caption="""Scatter plot of peak
        flux vs. the absolute galactic latitude. The points with error bars are
        the mean of the log flux.
        """)
        show("""This is a puzzle, which cannot be ascribed to threshold selection effect
        near the Galactic plane.""")

    def show_kde_comparison(self, fignum=99):
        
        def kde_comparison(data, hue, vars, ax=None ):
            
            assert np.all(np.isin(vars, data.columns)), f'{vars} not all in data'
            source_type = data[hue]
            msp_data = data[source_type=='MSP']  
            unid_data = data[source_type=='UNID-PSR' ] 
            
            # create KDE from the MSP data
            kde = Gaussian_kde(msp_data, vars)
        
            # and apply it to each
            msp_cdf = kde(msp_data) 
            unid_cdf = kde(unid_data)
        
            #compare probability distributions
            hkw = dict(bins=np.linspace(0, max(msp_cdf), 10), histtype='step', density=False, lw=2 )
            ax.hist(msp_cdf, **hkw,  label='MSP');
            ax.hist(unid_cdf, **hkw,   label='UNID-PSR')
        
            ## needs a msp_cdf defined for all data
            # sns.histplot(data, x='msp_cdf', hue=hue, hue_order='MSP UNID-PSR'.split(),
            #             element='step', hde=True, ax=ax)
            ax.set(ylabel='Counts', xlabel = 'KDE probability', 
                # xlim=(0,1), xticks=np.arange(0,1.1,0.25)
                )
            ax.legend()
            return kde

        show(f"""## KDE comparisons
        The UNID-PSR data set surely contains undetected MSPs. Assuming that
        they follow the same distributions in spectral characteristics and
        locations, this comparison should allow an estimate of the fraction.
        """)
        data = self.df.query('-0.5<log_epeak<0.7 & curvature>0.2').copy()
        data['log_dp'] = np.log10(data.curvature.clip(1e-3,12))
        hue = 'source type'
        
        fig, ax = plt.subplots()
        kde = kde_comparison(data, hue=hue, 
                                vars='log_epeak log_dp glat glon'.split(),
                                ax=ax)# 'log_epeak log_dp '.split())
        update_legend(ax, data, hue)
        show(f"""Run {kde} with MSP data""")
        show(fig,fignum=fignum,
            caption="""The MSP KDE probability distribution applied back to the
        MSP data, and to the overlapping UNID-PSR data set.
        """)
    
    def show_msp_estimate(self, fignum, vars='log_epeak log_dp glat glon'.split(),
                 spectral_cut='-0.5<log_epeak<0.7 & curvature>0.2', pmin=0.6):

        def do_kde(data, hue='source type', hue_order='MSP UNID-PSR'.split(), vars=vars ):
            from pylib.kde import apply_kde
            data['log_dp'] = np.log10(data.curvature.clip(1e-3,12))
            data['msp_cdf'] = apply_kde(data,hue=hue, hue_value='MSP',vars=vars)
            fig, ax = plt.subplots(figsize=(7,5))
            sns.histplot(data, x='msp_cdf', hue=hue, hue_order=hue_order,
                        element='step', bins=np.arange(0,1.1,0.1),kde=True,log_scale=(False, True), ax=ax)
            update_legend(ax, data, hue,)
            ax.set(ylabel='Counts', xlabel = 'MSP KDE probability', 
                xlim=(0,1), xticks=np.arange(0,1.1,0.25), ylim=(0.5,None),
                )
            return fig
        show(f"""## KDE analysis -- how many MSPs?
        Here we applly the spectral cut "{spectral_cut}" and then
        calculate the KDE proability distribution using the variables {vars} with the MSP subset.
        Then we plot below histograms of this distribution for the MSP and UNID_PSR subsets 
        satisfying the spectral cut.
        """)
        cut_data = self.df.query(spectral_cut).copy()
        show(
            do_kde(data=cut_data),
            fignum=fignum,
            caption="""Histograms of the MSP KDE probablilty, with KDE interpolations, for the 
            MSP and UNID-PSR data sets'
            """)
        show(f"""
        """)
        
        all = cut_data.groupby('source type').size()
        a,b = all['UNID-PSR'], all['MSP']
        cut = cut_data.query(f'msp_cdf>{pmin}').groupby('source type').size()
        c,d = cut['UNID-PSR'], cut['MSP']
        show(f"""Let's assume that all UNID-PSR sources with a MSP probability>{100*pmin:.0f}% are actually
        MSPs, since the two curves maintain a constant ratio above that. We make an estimate of the undetected MSP content in UNID-PSR with 
        the ratio of the number of UNID-PSR sources above {100*pmin:.0f}% MSP probability to the
        number of such MSP sources: {c}/{d} = {c/d:.1f}. This implies that {c/d * b:.0f} of the
        {a} UNID-MSP  sources are MSPs, leaving {a-c/d * b:.0f}.
        """)

    def forward(self):
        show(f"""
        ## The way forward
        * Comments/suggestions on turning the above into a paper!
        * Refit with 4FGL-DR4 only, (maybe Jean can produce a new version without the curvature prior? There is always DR3)
        * A section on previous ML results (Elizabeth)
        * A section on efforts to find associations (Kent)
        * Hopefully speculation on the science behind curved sources narrower than pulsars (Kent? Matthew?)
        """)


def main():
    self = Paper()
    show(self.setup_output)
    self.abstract()
    self.ml_summary()
    self.show_prelim_dp_vs_ep(fignum=1)
    self.show_skymaps(fignum=2)
    self.show_sin_b(fignum=5)
    self.show_dp_vs_ep( fignum=6)
    self.show_kde_comparison(fignum=7)
    self.show_fp_vs_abs_b(fignum=8)
    self.forward()
    return self
