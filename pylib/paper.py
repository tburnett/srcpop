from pylib.curvature import*

class HCUcut:
    def __init__(self, a=(-0.7,0.4), b=(0.7,1.5)):
        """ a,b -- 2-tuples for two points in log10(epeak) - curvature space
        defining a line
        """
        from numpy import linalg
        self.a = np.array(a)
        d = np.array(b)-self.a
        self.ref = d/linalg.norm(d)
    def perp(self, c):
        "Return signed perpendicular distance to the line"
        return float(np.cross(self.ref, np.array(c)-self.a))    
    def __call__(self, x):
        """Function of log10(Ep) returning a curvature defining the line"""
        return self.a[1] + (x-self.a[0]) * self.ref[1]/self.ref[0]
    def above(self, c):
        """Return True if c is above the line"""
        return c[1]>self(c[0])
    
def spectral_cut(df, elim=(0.4,4), fpmax=20):
    return df[ (df.curvature>0.5)  
            & (df.log_epeak > np.log10(elim[0]) )
            & (df.log_epeak < np.log10(elim[1]) )
            & (df.log_fpeak < np.log10(fpmax)   ) 
            ]
    
  
class Paper(Curvature):
    def __init__(self):
        super().__init__(title='Observation of a new gamma-ray source class')

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
        self.hcu_cut = HCUcut()
        df['HCU cut'] = df.apply(lambda row: self.hcu_cut.above((row.log_epeak, row.curvature)),axis=1)

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
              But the range of curvatures exceeds that observed for pulsars which are limited 
             by the monoenergetic curvature radiation value. We define a selection in the range
              of spectral parameters which almost all of these satisfy,  resulting in 617 total.
            Applying the same selection to each of the associated source classes we see that none of the 
             resulting spatial distributions are consistent, the closest being 
             millisecond pulsars (94) and Galactic clusters (19). 
             While there may be some undetected pulsars in this set, we conclude that the
             majority must represent a new class of gamma-ray emitting sources.
        """)


    def examine_cuts(self, elim=(0.4,4), fpmax=20):
        all = self.df 
        spcut = spectral_cut(all, elim=elim, fpmax=fpmax)
        low_b = spcut[spcut.log_fpeak<1] 
        c1 = all.groupby('source type').size(); c1.name='All'
        c2= spcut.groupby('source type').size(); c2.name='Spectral cut'
        c3 = low_b.groupby('source type').size(); c3.name='Weak'
        t = pd.DataFrame([c1,c2]).reindex(
            columns='MSP young bll fsrq bcu glc other UNID-PSR UNID-BLL UNID-FSRQ'.split())
        
        t['Total'] = np.sum(t.to_numpy(), axis=1)
        return t
    
    def d_vs_ep(self, ax=None, hue_order='MSP young UNID-PSR'.split(), legend='auto'):
        df = self.df
        hue_kw = dict(hue='source type', hue_order=hue_order)
        hue_order='bll fsrq psr'.split()
        size_kw =dict(size='log TS', sizes=(10,400))
        uv = self.hcu_cut
        fig, ax = plt.subplots(figsize=(8,6)) if ax is None else (ax.figure, ax)
        size_kw =dict(size='log TS', sizes=(10,200))
        sns.scatterplot(df.query('log_epeak>-0.75' ), ax=ax,
                        x='log_epeak', y='curvature', legend=legend,
                    **hue_kw, #hue='pulsar type', hue_order='MSP predicted'.split(),
                    **size_kw)
        xticks = [0.25,0.5,1,2,4]
        ax.set(xticks=np.log10(xticks), xticklabels=[f'{x}' for x in xticks] , 
            xlabel='$E_p$ (GeV)', xlim=(-0.9,0.9), ylim=(0,2.1))
        ax.set(yticks=np.arange(0,2.1,0.5))
        ax.axhline(4/3, ls='--', color='red')
        loge_pts = [-0.77,0.6]
        ax.plot(loge_pts, uv(loge_pts),  ls='--', color='k')
        if legend=='auto': plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.88,0.5));
        return fig

    def show_d_vs_ep(self, fignum=None, ):
        show(f"""## Separating HCUs and pulsars
    
            Here we look at the distribution in curvature and $E_p$.
            In Figure {fignum} below we show the curvature vs. $E_p$ for the known pulsars and 
            the curved UNID sources. The red dashed line at curvature 4/3 represents
            the upper limit for pulsars, and indeed the HCU candidate distribution 
            shows no indication of such a limit. However, there must be pulsar
            contributions. To estimate that, and determine a selection that concentrates
            the HCU contribution, we will select those above the inclined dashed black line. 
            """)
        fig = self.d_vs_ep()
        
        show(fig, fignum=fignum, caption="""Scatter plot of curvature vs. $E_p$ showing known pulsars and
        UNIDs predicted to be pulsars. The horizontal dashed red line is at 4/3,
        the synchrotron curvature radiation maximum. The inclined black dashed
        line represents an empirical separation to isolate most of the MSPs
        and generate an enhanced HCU sample for those above it.""")

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

    def ml_summary(self):
        show(f"""## ML step summary
        * Training set: pulsars (MSP + young), BL Lacs, FSRQs
        * Features:  spectral peak parameters ( energy, flux, curvature), variability, energy flux
        * predictions: UNID -> (UNID-PSR, UNID-BLL, UNID-FSRQ)
        """)

    def forward(self):
        show(f"""
        ## The way forward
        * Invite checks!
        * Refit with 4FGL-DR4 only, (maybe Jean can produce a new version without the curvature prior? There is always DR3)
        * Estimate the fraction of MSPs, using Careful comparison of the curvature distributions
        * A section on previous ML results (Elizabeth)
        * A section on efforts to find associations (Kent)
        * Hopefully speculation on narrow curved sources
        """)


def main():

    self = Paper()
    show(self.setup_output)
    self.abstract()
    self.ml_summary()
    
    show(f"""## Curvature vs. $E_p$ for UNID
    """)
    fig, ax = plt.subplots(figsize=(10,8))
    sns.scatterplot(self.df, ax=ax,x='log_epeak', y='curvature', hue='source type', 
                    hue_order='UNID-BLL UNID-FSRQ UNID-PSR'.split(),
                size='log_ts', sizes=(20,200),)# alpha=0.4)
    x = np.log10([0.4, 0.4, 4, 4])
    y = [2,0.5,0.5,2]
    ax.plot(x, y, ls=':', color='k');
    ax.set(xlabel='$E_p$ (GeV)',xticks=[-1,0,1,2], 
        xticklabels='0.1 1 10 100'.split(), xlim=(-1.2,2.5))
    plt.legend(fontsize=12)

    show(fig, fignum=1, caption="""Curvature vs. $E_p$. Colors identify the ML prediction.
    The dotted line encompasses most of the PSR prediction, defines the "spectral cut".
    """)

    elim=(0.4,4); fpmax=20; cmin=0.5
    show(f"""### Application of the "spectral cut" to the source classes
        Selection cuts: 
        * curvature>{cmin}
        * {elim[0]}< $E_p$ < {elim[1]} GeV  
        * $F_p$ < {fpmax} eV s-1 cm-2
        """)
    show(self.examine_cuts(elim, fpmax))

    show(f"""### Skymaps: UNID """)
    show( ait_plots(self.df, 
                    hue='source type', hue_order='UNID-FSRQ UNID-BLL UNID-PSR'.split(),
                )  ,fignum=2, caption='Aitoff plots')
    show(f"""### Skymaps: UNID after spectral cut""")
    show( ait_plots(spectral_cut(self.df), 
                    hue='source type', hue_order='UNID-FSRQ UNID-BLL UNID-PSR'.split(),
                )  ,fignum=3, caption='Aitoff plots')

    show(f"""### Skymaps: Associated after spectral cut """)
    df = self.df.copy()
    def sort_ided(st):
        if st in 'MSP glc'.split(): return st
        if st in 'bcu bll fsrq'.split(): return 'blazar'
        return 'the rest'
    df['st'] = df['source type'].apply(sort_ided)
    show(
        ait_plots(spectral_cut(df.query('association!="unid"')), 
            hue='st', hue_order=('MSP', 'glc', 'blazar','the rest') ),
        fignum=4, caption='Aitoff plots')

    show(""" ### Peak flux vs. $|b|$ """ )
    show(self.fp_vs_abs_b(), fignum=5, caption="""$F_p$ vs. $|b|$.""")

    self.forward()
    return self
