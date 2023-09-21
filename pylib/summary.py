from paper import *
from pathlib import Path


def pulsar_curvature(logE_p):
    r""" Return log of average curvature $\log(d_p)$ for a given $\log(E_p)$, based on 3PC measurement of each
    with respect to $\dot{E}$.<br>
    Inverts $E_p = 1.1\ \mathrm{{GeV}}\ (d_p/0.46)^{{1.33}}$
    """
    return ( logE_p - np.log10(1.1))/1.33 + np.log10(0.46) 



class Summary(Paper):

    def __init__(self, summary_file='files/summary.csv', *pars, **kwargs):
        if summary_file is not None and Path(summary_file).exists():
            show(f'# {kwargs.pop("title", "")}')
            show_date()
            self.df = pd.read_csv(summary_file, index_col=0)
            # make log10 columns
            log_list = dict(Ep='log_epeak',Fp='log_fpeak', ts='log TS')
            for k,v in log_list.items():
                self.df[v]= np.log10(self.df[k])      
            self.df['abs_sin_b']= np.abs(np.sin(np.radians(self.df.glat)))                      
            show(f'* Read file `{summary_file}` with {len(self.df)} sources.')
        else:
            super().__init__(*pars, **kwargs)
            self.spectral_selection()
        self.hue_kw =dict(hue='source type', hue_order='UNID-PSR MSP young'.split(),
                 palette='green red blue'.split())
        

    def save(self, summary_file):
        show(f'## Write summary to `{summary_file}`')
        df = self.df.copy()
        df['Ep'] = np.power(10, df.log_epeak)
        df['Fp'] = np.power(10, df.log_fpeak)
        cols='glon glat ts r95 curvature Fp Ep'.split()+['source type']
        rows = np.isin(df['source type'].values, 'UNID-PSR young MSP'.split())
        df.loc[rows, cols].to_csv(summary_file, float_format='%.3f' )

    def spectral_selection(self, ecut = (0.25,6), dp_min= 0.15):
        log_ecut = np.log10(ecut)

        df= self.df.query(f'curvature>{dp_min} & {log_ecut[1]} >log_epeak>{log_ecut[0]}').copy()
        df['log_dp'] = np.log10(df.curvature)
        df['xi'] = pulsar_curvature(df.log_epeak)
        show(f"""* Select subset for following: {ecut[0]} < $E_p$ < {ecut[1]} and $d_p$>{dp_min} """)
        self.df=df

    def dp_vs_ep(self,  df=None,ax=None,):
        if df is None: df = self.df
        fig, ax = plt.subplots(figsize=(10,6)) if ax is None else (ax.figure, ax)
        
        data_kw= dict(data=df, x='log_epeak', y='log_dp' )
        size_kw = dict(size='log_fpeak', sizes=(20,200) )
        sns.scatterplot(**data_kw,  **self.hue_kw,  **size_kw ,)
        
        xx = np.log10((0.25,6)) 
        ax.plot(xx, pulsar_curvature(xx), 'k', ls=':');
        axis_kw= lambda a, label, v: {f'{a}label':label,f'{a}ticks':np.log10(v), f'{a}ticklabels':v }
        
        ax.set(**axis_kw('x','$E_p$ (GeV)', [0.25,0.5,1,2,4]),
            **axis_kw('y','$d_p$', [0.25,0.5,1,2]) )
        update_legend(ax, df, hue='source type',  fontsize=10,
                    bbox_to_anchor=(0.85,0.55))
        return fig

    def show_dp_vs_ep(self, fignum, df=None, **kwargs):
        fig, ax = plt.subplots(figsize=(10,6))
        if df is None: df = self.df
        fig = self.dp_vs_ep(ax=ax,df=df)
        
        show(f"""## Plot $d_p$ vs. $E_p$ on log scales
        The dotted line is the measured dependence of the mean for pulsars,
        $E_p = 1.1\ \mathrm{{GeV}}\ (d_p/0.46)^{{1.33}}$
        """)
        show(fig, fignum=fignum, caption=r"""Scatter plot of $d_p$ vs $E_p$ for UNID_PSR and MSP sources.
        """)
        
        show(f"""### Plot the difference""")
        xi = pulsar_curvature(df.log_epeak)
        
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df, ax=ax, x=df.log_dp-xi, **self.hue_kw, 
                    kde='True', element='step');
        ax.set(xlabel=r'$\log(d_p-\xi)$');
        ax.axvline(0, ls='--', color='0.6')
        update_legend(ax, df, hue='source type');
        show(fig, fignum=fignum+1, 
            caption=r"""Histogram of the log curvature difference, where $\xi$ is the pulsar mean,
            $d_p(E_p)$. """)
        
    def fp_vs_sinb(self, fignum, df=None):
        show(f"""## $F_p$ vs. $|\sin(b)|$""")
        if df is None: df=self.df
        fig, ax = plt.subplots()
        sns.scatterplot(df, ax=ax,x='log_fpeak', y='abs_sin_b', **self.hue_kw)\
            .set(xlim=(-2,3.5));
        update_legend(ax, df,hue='source type', loc='upper right', fontsize=12);
        fpcut = lambda x: 0.55*(1.5-x)/2.5
        xx = np.array((-1.2,1.5))
        ax.plot(xx, fpcut(xx), '--k') #[-1,1.5], [0.55,0], '--k')
        show(fig, fignum=fignum, caption=' ')
        show(f"""Now select those below the dashed line... """)
        dfx = df.query('-1.2< log_fpeak<1.5 ').copy()
        return df[
            (df.log_fpeak.apply(fpcut)>df.abs_sin_b)
            & (df.log_fpeak>-1.2)
            ]
    
    def skyplot(self,fignum,  df=None,  **kwargs):
        fig,ax = plt.subplots(figsize=(12,6))
        if df is None: df = self.df
        glon = df.glon.values
        glon[glon>180]-=360
        sns.scatterplot(df, ax=ax, x=glon, y=df.glat, **self.hue_kw,
                    size='log_fpeak',sizes=(5,100))
        xticks=kwargs.get('xticks', np.arange(180,-181, -90))
        kw = dict(xlabel='Galactic longitude', xlim=(180,-180), 
            xticks=xticks, xticklabels=np.mod(xticks,360),
                ylabel='Galactic latitude')
        kw.update(kwargs)
        ax.set( **kw,)#.update(xlim=(-45, -90), ylim=(-10,10))) )
        ax.axhline(0, color='0.6')
        ax.axvline(0, color='0.6')
        update_legend(ax, df, hue=self.hue_kw['hue'], loc='upper left',fontsize=12)
        show(fig, fignum=fignum, caption="""Positions. """)

