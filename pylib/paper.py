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
    
class Paper(Curvature):
    def __init__(self):
        super().__init__(title='Observation of a new gamma-ray source class')

        df = self.df#[self.df.nbb<4].copy()
        df['log TS'] = np.log10(df.ts)

        def plike(rec):
            class1 = rec.class1
            if not pd.isna(class1) and class1.lower() in ('msp','psr'): 
                return dict(msp='MSP', psr='young')[rec.class1.lower()]
            if rec.association=='unid': return 'UNID-'+rec.prediction.upper()
            return rec.association
        df['source type']= df.apply(plike, axis=1)
        self.hcu_cut = HCUcut()
        df['HCU cut'] = df.apply(lambda row: self.hcu_cut.above((row.log_epeak, row.curvature)),axis=1)

        self.df=df

    def d_vs_ep(self, hue_order='MSP young UNID-PSR'.split()):
        df = self.df
        hue_kw = dict(hue='source type', hue_order=hue_order)
        hue_order='bll fsrq psr'.split()
        size_kw =dict(size='log TS', sizes=(10,400))
        uv = self.hcu_cut
        fig, ax = plt.subplots(figsize=(8,6))
        size_kw =dict(size='log TS', sizes=(10,200))
        sns.scatterplot(df.query('log_epeak>-0.75' ), ax=ax,
                        x='log_epeak', y='curvature',
                    **hue_kw, #hue='pulsar type', hue_order='MSP predicted'.split(),
                    **size_kw)
        xticks = [0.25,0.5,1,2,4]
        ax.set(xticks=np.log10(xticks), xticklabels=[f'{x}' for x in xticks] , 
            xlabel='$E_p$ (GeV)', xlim=(-0.9,0.9), ylim=(0,2.1))
        ax.set(yticks=np.arange(0,2.1,0.5))
        ax.axhline(4/3, ls='--', color='red')
        loge_pts = [-0.77,0.6]
        ax.plot(loge_pts, uv(loge_pts),  ls='--', color='k')
        plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.88,0.5));
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

    # def fp_vs_sinb(self, fignum=100, hue_order='MSP young UNID-PSR'.split()):
        
    #     df = self.df
    #     df_hcu = df[df['HCU cut']]
    #     gold_hcu = df_hcu[df_hcu['source type']=='UNID-PSR']
    #     show(f""" ### $F_p$ vs. $|\\sin(b)|$              
    #         Selecting sources above the black line in Figure {fignum-1},
    #         results in {len(gold_hcu)} HCU sources. In Figure {fignum} we see 
    #         the resulting distribution in peak flux $F_p$ vs. $|\\sin(b)|$ for known
    #         pulsars and HCU candidates.
    #     """)

    #     hue_kw = dict(hue='source type', hue_order=hue_order)
    #     size_kw =dict(size='log_ts', sizes=(10,100))
    #     fig, ax = plt.subplots(figsize=(8,6))

    #     sns.scatterplot(df_hcu, ax=ax, x='abs_sin_b', y='log_fpeak', 
    #                     **hue_kw,**size_kw)
    #     ax.set(xlim=(0,1.), ylim=(-1.5,1.5), ylabel='$F_p$ (eV s-1 cm-2)',
    #         yticks=[-1,0,1], yticklabels='0.1 1 10'.split(),
    #         xlabel= r'$|\sin(b)|$')
    #     plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.88,0.5));
    #     show(fig, fignum=fignum, caption="""Scatter plot peak flux vs latitude for 
    #         known pulsars and pulsar-like UNIDS.
    #     """)
    #     show("""
    #         """)
    def fp_vs_sinb(self, fignum=2, cut='HCU cut',  hue_order='MSP young UNID-PSR'.split()):
    
        df = self.df if cut is None else self.df[self.df[cut]]

        hue_kw = dict(hue='source type', hue_order=hue_order)
        size_kw =dict(size='log_ts', sizes=(10,100))
        fig, ax = plt.subplots(figsize=(8,5))

        sns.scatterplot(df, ax=ax, x='abs_sin_b', y='log_fpeak', 
                        **hue_kw,**size_kw)
        ax.set(xlim=(0,1.), ylim=(-1.5,1.5), ylabel='$F_p$ (eV s-1 cm-2)',
            yticks=[-1,0,1], yticklabels='0.1 1 10'.split(),
            xlabel= r'$|\sin(b)|$')
        plt.legend(fontsize=12, loc='lower left', bbox_to_anchor=(0.88,0.4));
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

        fig = self.fp_vs_sinb( fignum, cut, hue_order)
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

    def stats(self, filename='files/hcu_candidates.csv'):
        show("""#### Apply HCU spectral cut and write out file""")
        hcu_cut = HCUcut()     
        
        df = self.df[self.df.nbb<4].copy()   
        hcu = df.apply(lambda row: hcu_cut.above((row.log_epeak, row.curvature)),axis=1)

        c1 = df.groupby('source type').size(); c1.name='All'
        c2 = df[hcu].groupby('source type').size(); c2.name='HDU cut'
        show(pd.DataFrame([c1,c2]))
        if filename is not None:
            self.df_hdu =df[hcu & (df['source type']=='UNID')]
            df = self.df_hdu.copy()
            df.index.name = '4FGL-DR4'
            df['Fp'] = 10**df.log_fpeak
            df['Ep'] = 10**df.log_epeak
            cols = 'ts r95 glat glon Fp Ep curvature sgu uw_name'.split()

            df.loc[:,cols].to_csv(filename, float_format='%.3f')
            show(f"""Write {len(df)} HCU candidates to `{filename}`""")
            