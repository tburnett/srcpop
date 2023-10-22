from paper import *
from pathlib import Path
sns.set_theme('notebook' if 'talk' not in sys.argv else 'talk', font_scale=1.25) 
if 'dark' in sys.argv:
#'talk')
    plt.style.use('dark_background') #s
    plt.rcParams['grid.color']='0.5'
    dark_mode=True
else: dark_mode=False
spectral_model='uw'if 'uw' in sys.argv else 'fgl'


def pulsar_curvature(logE_p):
    r""" Return log of average curvature $\log(d_p)$ for a given $\log(E_p)$, based on 3PC measurement of each
    with respect to $\dot{E}$.<br>
    Inverts $E_p = 1.1\ \mathrm{{GeV}}\ (d_p/0.46)^{{1.33}}$
    """
    return ( logE_p - np.log10(1.1))/1.33 + np.log10(0.46) 



class Summary(Paper):

    def __init__(self, summary_file=None, #'files/summary.csv', 
                 *pars, **kwargs):
        if summary_file is None:
            summary_file = f'files/{spectral_model}_summary.csv'

        assert Path(summary_file).exists(), f'{summary_file} ?'
        show(f'# {kwargs.pop("title", "")}')
        show_date()
        self.df = pd.read_csv(summary_file, index_col=0)
        # make log10 columns
        log_list = dict(Ep='log_epeak',Fp='log_fpeak', ts='log TS')
        for k,v in log_list.items():
            self.df[v]= np.log10(self.df[k])      
        self.df['abs_sin_b']= np.abs(np.sin(np.radians(self.df.glat)))                      
        show(f'* Read file `{summary_file}` with {len(self.df)} sources.')


        # add diffuse entry if not saved
        assert 'diffuse' in self.df.columns
        self.df['log_diff']  = self.df.diffuse # 

        # Set seaborn hue stuff
        self.hue_kw =dict(hue='source type', 
                          hue_order='UNID-PSR MSP young'.split())
        self.size_kw = dict(size='log TS', sizes=(20,200) )
        if dark_mode:
            self.hue_kw.update(palette='yellow magenta cyan'.split(), edgecolor=None)
        else:
            self.hue_kw.update( palette='green red blue'.split())
        
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

        
        def show_d_vs_ep(self, df, xy='log_epeak d'.split(), hue_kw={} ):
            show(f""" ## Curvature vs peak energy for pulsar-like sources""")

            data=df
            x,y =  xy
            g = sns.JointGrid(height=12, ratio=4 )
            ax = g.ax_joint
            size_kw = dict(size='log TS', sizes=(20,200) )
            hkw = self.hue_kw
            hkw.update(hue_kw)
            sns.scatterplot(data, ax=ax, x=x,y=y, **hkw, **size_kw);
            axis_kw= lambda a, label, v: {
                f'{a}label':label,f'{a}ticks':np.log10(v), f'{a}ticklabels':v }
            
            ax.set(**axis_kw('x','$E_p$ (GeV)', [0.1, 0.25,0.5,1,2,4]), 
                yticks=np.arange(0,1.51, 0.5), ylim=(-0.1,2), 
                xlim=np.log10((0.08, 5)) )
            ax.axhline(0, color='grey')
            ax.axhline(4/3, color='orange', ls='--')
            xx = np.linspace(-1,np.log10(5)); xx
            # ax.plot(xx, 10**(pulsar_curvature(xx)), 'orange', ls='--');
            
            hkw = dict(element='step', kde=True, bins=25, **self.hue_kw,
                    legend=False)
            sns.histplot(data, y=y, ax=g.ax_marg_y, **hkw)
            sns.histplot(data, x=x, ax=g.ax_marg_x, **hkw)
            
            update_legend(ax, df, hue='source type',  fontsize=14, 
                        loc='upper left')
            show(g.fig)

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

    def show_fp_vs_diffuse(self, df=None):
        if df is None: df=self.df
            
        def fp_vs_diffuse( ):
            fig, ax = plt.subplots(figsize=(8,6))
            size_kw = dict(size='log TS', sizes=(20,150) )
            sns.scatterplot(df, ax=ax, x='log_diff',y='log_fpeak', **self.hue_kw, **size_kw );
            plt.legend(loc='upper right', fontsize=12,bbox_to_anchor=(1.1,1.1));
            ax.set(**self.diff.fluxticks('x'), **fpeak_kw('y'), ylim=(-2,4));
            return fig
            
        show(f"""### Peak flux vs. diffuse
        Define the diffuse energy flux (DEF) as the Galacic diffuse energy flux at
        the position of the source.""")
        show(fp_vs_diffuse())
        show(f"""Note that the DEF plays two roles:
        1. Separates MSP from young, via correlation with Galactic material.
        2. Defines detection and association thresholds.  """)

    def show_logN_logS(self):
        show("""## Examine logN-logS
        Here we define "S" as the ratio of the peak energy flux to that of the Galactic diffuse at the source position,
             basically a signal/noise ratio.
        
        """)
        hkw=self.hue_kw.copy()
        hkw.pop('edgecolor', '')
        df = self.df
        fig, ax = plt.subplots(figsize=(6,6))
        x=df.log_fpeak-df.log_diff
        
        sns.ecdfplot(df,x=x ,ax=ax, complementary=True,**hkw ,  log_scale=(False, True), stat='count');
        ax.set(ylim=(None, 4e3), xlabel='log(flux ratio /deg^2)', ylabel='N(> S)');
        show(fig)

    def show_curvature_vs_diffuse(self, df=None, ):
        if df is None: df = self.df
        fig, ax = plt.subplots(figsize=(10,6))
        size_kw = dict(size='log TS', sizes=(20,150) )
        sns.scatterplot(df,ax=ax, x='log_diff', y='d', **self.hue_kw, **size_kw );
        plt.legend(loc='upper right', fontsize=12,bbox_to_anchor=(1.05,1.1))

        ax.set(**self.diff.fluxticks('x'));#, **fpeak_kw('y'), ylim=(-2,4));
        show(r"""### Curvature vs diffuse 
        Check that can ignore curvature.""")
        show(fig)

    def flux_ratio_vs_diffuse(self, df, hue_kw, height=10, ratio=2 ,**kwargs):
        if ratio is None:
            fig, ax = plt.subplots(figsize=(height,height))
        else:
            g = sns.JointGrid(height=height, ratio=ratio )
            ax = g.ax_joint
       
        size_kw = self.size_kw 
        x, y =  df.log_diff, df.log_fpeak-df.log_diff,
        sns.scatterplot(df,ax=ax, x=x, y=y, **hue_kw, **size_kw , **kwargs);
                    
        update_legend(ax, df, hue=hue_kw['hue'], fontsize=12,  
                      loc='upper right', bbox_to_anchor=(1.08,1.08))
        if  ratio is not None:
            hkw = dict(element='step', kde=True, bins=40, **hue_kw,
                            legend=False, log_scale=(False, False),)
            sns.histplot(df, y=y, ax=g.ax_marg_y, **hkw)
            sns.histplot(df, x=x, ax=g.ax_marg_x, **hkw)
        xticks=np.arange(-1,2.1,1)
        yticks = np.arange(-1,2.1,1)
        ax.set(ylim=(-2.1, 3), xlabel='Diffuse energy flux at 900 MeV (eV cm-2 s-1 deg-2)', 
        xticks=xticks,  xticklabels=[rf'$10^{{{int(t)}}}$' for t in xticks],
        yticks=yticks, yticklabels=[rf'$10^{{{int(t)}}}$' for t in yticks],
                ylabel = r'Flux ratio ($\rm{deg}^2)$')
        return ax
        
    def show_ratio_vs_diffuse(self, df=None, hue_kw=None):
        if df is None: df=self.df
        if hue_kw is None: hue_kw = self.hue_kw

        g = self.flux_ratio_vs_diffuse(df.query('ts<1.1e4'), hue_kw=self.hue_kw)
        show(r"""### Flux ratio vs diffuse energy flux""")
        show(g.figure)

    def setup_diffuse(self):
        from pylib.diffuse import Diffuse
        self.diff = Diffuse()

#--------------------------------------------------------------------------------
if 'doc' in sys.argv:
    self = Summary(show_confusion=False, #summary_file=None,
             title="""Study diffuse background """)
    from pylib.diffuse import Diffuse
    self.diff = diff = Diffuse()
    show(diff.eflux_plot())
    self.show_fp_vs_diffuse(self.df)
    self.show_ratio_vs_diffuse()
    self.show_logN_logS()
    self.show_curvature_vs_diffuse()
