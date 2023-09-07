from paper import *



def pulsar_curvature(E_p):
    r""" Return average curvature $d_p$ for a given $E_p$, based on 3PC measurement of each
    with respect to $\dot{E}$.<br>
    Invert $E_p = 1.1\ \mathrm{{GeV}}\ (d_p/0.46)^{{1.33}}$
    """
    return ( E_p - np.log10(1.1))/1.33 + np.log10(0.46) 



class Summary(Paper):

    def spectral_selection(self, ecut = (0.25,6), dp_min= 0.15):
        log_ecut = np.log10(ecut)

        df= self.df.query(f'curvature>{dp_min} & {log_ecut[1]} >log_epeak>{log_ecut[0]}').copy()
        df['log_dp'] = np.log10(df.curvature)
        df['xi'] = pulsar_curvature(df.log_epeak)
        show(f"""* Select {ecut[0]} < $E_p$ < {ecut[1]} and $d_p$>{dp_min} """)
        self.df=df

    def show_dp_vs_ep(self, fignum):
        df = self.df
        fig, ax = plt.subplots(figsize=(10,6))
        
        data_kw= dict(data=self.df, x='log_epeak', y='log_dp' )
        size_kw = dict(size='log TS', sizes=(20,200) )
        sns.scatterplot(**data_kw,  **self.hue_kw,  **size_kw ,)
        
        xx = np.log10((0.25,6)) 
        ax.plot(xx, pulsar_curvature(xx), 'k', ls=':');
        xticks, xticklabels = np.log10([0.25,0.5,1,2,4]), '0.25 0.5 1 2 4'.split()
        ax.set(xlabel='$E_p$ (GeV)',xticks=xticks, xticklabels=xticklabels,
                ylabel=r'$\log(d_p)$',)
        update_legend(ax, df, hue='source type',  fontsize=12,
                    bbox_to_anchor=(0.85,0.55))
        show(fig, fignum=fignum, caption="""Scatter plot of $d_p$ vs $E_p$ for UNID_PSR and MSP sources.
        The dotted line is the measured dependence of the mean for pulsars.""")
        
        show(f"""### Plot the difference""")
        
        fig, ax = plt.subplots(figsize=(6,3))
        sns.histplot(df, ax=ax, x=df.log_dp-df.xi, **self.hue_kw, 
                    kde='True', element='step');
        ax.set(xlabel=r'$\log(d_p-\xi)$');
        ax.axvline(0, ls='--', color='0.6')
        update_legend(ax, df, hue='source type');
        show(fig, fignum=fignum+1, 
            caption="""Histogram of the log curvature difference. """)
        
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

