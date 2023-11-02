"""
"""
from pylib.ml_fitter import *
from pylib.diffuse import Diffuse
from pylib.ipynb_docgen import show, capture_hide, capture_show, show_fig


sns.set_theme('notebook' if 'talk' not in sys.argv else 'talk', font_scale=1.25) 
if 'dark' in sys.argv:
    plt.style.use('dark_background') #s
    plt.rcParams['grid.color']='0.5'
    dark_mode=True
else:
    dark_mode=False
fontsize = plt.rcParams["font.size"] # needed to be persistent??
def fpeak_kw(axis='x'):
    return {axis+'label':r'$F_p \ \ \mathrm{(eV\ s^{-1}\ cm^{-2})}$', 
            axis+'ticks': np.arange(-2,5,2),
            axis+'ticklabels': '$10^{-2}$ 1 100 $10^4$'.split(), 
            }
def epeak_kw(axis='x'):
    return {axis+'label':'$E_p$  (GeV)',
            axis+'ticks': np.arange(-1,1.1,1),
            axis+'ticklabels':'0.1 1 10 '.split(),
            }
            
class UnidAnalysis( Diffuse, MLspec):
    
    def __init__(self, title=None):
        if title is None: title = f'Unid analysis with {dataset}'
        show(f"""# {title}""")
        filename = f'files/{dataset}_pulsar_summary.csv'
        self.df = df =pd.read_csv(filename, index_col=0)
        show(f"""* read summary file `{filename}` """)
        self.dark_mode = dark_mode
        # super().__init__()
        
        df['log_epeak'] = np.log10(df.Ep)
        df['log_fpeak'] = np.log10(df.Fp).clip(-2,5)  
        df['log_signif'] = np.log10(df.significance)
        self.size_kw = dict(size='log_signif', sizes=(20,200))
        self.hue_kw = dict(hue='source_type', 
                          hue_order='unid-pulsar msp psr'.split(),
                          palette=self.palette)
        if dark_mode:
            self.hue_kw.update(palette='yellow magenta cyan'.split(), edgecolor=None)
        else:
            self.hue_kw.update(palette='green red blue'.split())

    def check_hpm(self):
        # for plotting 
        if hasattr(self, 'diffuse_hpm'): return
        super().__init__()

    def diffuse_vs_ep(self, df=None, hue_kw=None, kde=None):
        """ Diffuse vs peak energy for pulsar-like sources.
        """
        # fig, ax = plt.subplots(figsize=(15,8))
        data=df if df is not None else self.df
        if hue_kw is None: hue_kw = self.hue_kw
        x,y = 'log_epeak diffuse'.split()
        g = sns.JointGrid(height=8, ratio=4 )
        ax = g.ax_joint
        size_kw = dict(size='log TS', sizes=(20,200) ) if not hasattr(self,'size_kw') else self.size_kw
        sns.kdeplot(data, ax=ax, x=x, y=y, 
                    hue=hue_kw['hue'], hue_order=hue_kw['hue_order'][1:3], palette = self.palette[1:3])#  **size_kw);
        sns.scatterplot(data, ax=ax, x=x,y=y, hue=hue_kw['hue'], hue_order=hue_kw['hue_order'][0:1], 
                        palette=self.palette[0:1], edgecolor='none',  alpha=0.5,)# **size_kw)
        axis_kw= lambda a, label, v: {f'{a}label':label,f'{a}ticks':np.log10(v), f'{a}ticklabels':v }
        
        ax.set(**axis_kw('x','$E_p$ (GeV)', [0.1, 0.25,0.5,1,2,4]),xlim=np.log10((0.1,6)), 
            **self.fluxticks('y')
            )
        # ax.scatter(orion.log_epeak, orion.diffuse,  marker='o', s=200, color='k', facecolor='none', label='selected')
        hkw = dict(element='step', kde=True, bins=25, **hue_kw, legend=False)
        sns.histplot(data, y=y, ax=g.ax_marg_y, **hkw)
        sns.histplot(data, x=x, ax=g.ax_marg_x, **hkw)
        update_legend(ax, data, hue=hue_kw['hue'],  fontsize=12,   loc='lower left')
        return g.fig   
    
    def _plot_psr(self, ax, hue_order=None):
        df = self.df
        for hue, marker, color in zip(self.hue_kw['hue_order'] if hue_order is None else hue_order, 
                                    '*oD',   self.hue_kw['palette']):
            t =  df[df.loc[:,self.hue_kw['hue']]==hue]
            ax.scatter(t, marker=marker, s=50, color=color, label = f'({len(t)}) {hue}')
        ax.legend(fontsize=14)
        
    def ait(self, hue_order=None):
        """Aitoff plot of the galactic diffuse showing locations of pulsars and the unid-pulsar category.
        """
        self.check_hpm()
        ax = self.ait_plot(figsize=(20,8))#cmap='gist_gray', log=False,);
        self._plot_psr(ax, hue_order)
        return ax.figure

    def zea(self, *args, size=90):
        """ZEA projection of the galactic diffuse with positions of pulsars and the unid-pulsar category"""
        self.check_hpm()
        ax = self.zea_plot(*args, size=size)
        self._plot_psr(ax)
        ax.grid(color='0.5', ls='-')
        # ax.legend(loc='upper left');
        return ax.figure

    def pulsar_pairplot(self, vars):
        """The corner plot with KDE contours for the spectral features and the diffuse value.  
        """
        data=self.df
        kw = self.hue_kw.copy()
        kw.pop('edgecolor')
        return sns.pairplot(data, kind='kde', vars=vars,  corner=True, **kw).figure
        
    def apply_kde(self, df=None, 
            vars='log_epeak log_fpeak d diffuse'.split(),
            pulsars=None):
        """
        Evaluate the KDE for the pure pulsar types, {pulsars}, using the {vars} features.
        Apply to all sources, adding columns with the pulsar type names with "_kde" appended.
        """
        from pylib.kde import Gaussian_kde
        if df is None: df=self.df
        if pulsars is not None: pulsars = self.hue_kw['hue_order'][1:]
        
        for pname, subdf in df.groupby('source_type'):
            if pname not in pulsars: continue
            # Generate a KDE  function with the subset
            gkde = Gaussian_kde(subdf, cols=vars)

            # and apply it to all sources
            df[pname+'_kde'] = gkde(df)   

    def plot_kde(self,  order):
        """Scatter plots of the `msp` vs `psr` KDE probabilities for each source type.
        """
        data = self.df
        g = sns.FacetGrid(data, col='source_type', col_wrap=4, 
                        col_order=order, sharex=True, sharey=True)
        g.map(sns.scatterplot, 'msp_kde', 'psr_kde', s=10);
        counts = data.groupby('source_type').size()[order]
        for ax, n  in zip(g.axes.flat, counts):
            ax.set_title(ax.get_title().split('=')[1]+ f' ({n})')
        return g.figure
   
    def kde_with_scatter(self, df=None, hue_kw=None):
        """Unid-pulsar parameter scatter plots.
        The two plots show the spectral shape parameters, curvature $d$ and $E_p$ on the right, and the flux parameters
        $F_p$ and log diffuse energy flux on the left, as well as KDE contours for the  `psr` and `msp` sources.
        """
        data=df if df is not None else self.df
        if hue_kw is None: hue_kw = self.hue_kw
            
        def kscat( ax, title,  x='log_epeak', y='diffuse', **kwargs ):    

            sns.kdeplot(data, x=x, y=y,  ax=ax,  
                        hue=hue_kw['hue'], hue_order=hue_kw['hue_order'][1:3], palette=self.palette[1:3])
        
            sns.scatterplot(data, ax=ax, x=x,y=y, hue=hue_kw['hue'], hue_order=hue_kw['hue_order'][0:1], 
                            palette=self.palette[0:1], edgecolor='none',  alpha=0.5, legend=False)
            ax.set(title=title, **kwargs)
            ax.get_legend().set_title('KDE contours')

        fig, (ax2,ax1) = plt.subplots(ncols=2, figsize=(15,7), gridspec_kw=dict(wspace=0.3))
        kscat( ax=ax1, title='Spectral shape',  x='log_epeak', y='d', 
            **epeak_kw(), ylim=(-0.2,2.1));
        kscat( ax=ax2, title='Fluxes', y='log_fpeak', x='diffuse',
            **fpeak_kw('y'), xticks=np.arange(-1,2.1,1), xlabel='log diffuse flux')
        return fig


if 'unid-doc' in sys.argv:
    #=================================================================================================================
    self = UnidAnalysis(title='Unid-pulsar analysis')
    show(f'[Unid-pulsar analysis Confluence page]'\
        '(https://confluence.slac.stanford.edu/display/SCIGRPS/Unid+analysis)')
    show_date()

    show(f"""This section examines the properties of the "unid-pulsar" sources, the unassociated sources predicted to be pulsars.
    
    There clearly must be young pulsars (`psr`) and MSPs (`msp`) in this set. The point of this section is to compare the properties,
    estimate how many of each there are, and discuss how the remainder are unique.

    We add the diffuse energy flux background discussed in the Diffuse Background section to the three spectral features, and
    start with a "corner" plot showing the four distributions and correlations.
    """)
    section=3
    vars = 'diffuse log_fpeak log_epeak d'.split()
    pulsars = self.hue_kw['hue_order'][1:]

    show(f"""###  Corner plot for spectral properties and diffuse
    """)
    show_fig(self.pulsar_pairplot, fignum=section+.1, vars=vars)

    show("""The upper and lower corners show separations of the three source types. In the following plots we expand those,
    with a scatter plot for the `unid-pulsar` sources.
    """)
    show_fig(self.kde_with_scatter, fignum=section+0.2)
    

    show(f"""## Apply KDE
        """)
    self.apply_kde(  vars=vars, pulsars=pulsars, )

    show(self.apply_kde.__doc__.format_map(locals()))

    show(f"""### KDE scatter plot for the pulsar types:""")
    show_fig( self.plot_kde,  order=pulsars, fignum=section+0.3)

    show("""Separation is good. Now we examing the KDE values for the rest""")
    other_names = ['unid-pulsar'] + [name  for name in np.unique(self.df.source_type)
                                    if name!='unid-pulsar' and name not in pulsars]
    show_fig( self.plot_kde,  order=other_names, fignum=section+0.4)
    show(f"""The unid-pulsar and bcu-pulsar have many with low kde's for both.
    """)
    show('---')

 