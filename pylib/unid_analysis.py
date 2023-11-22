"""
"""
from pylib.ml_fitter import *
from pylib.diffuse import Diffuse
from pylib.ipynb_docgen import show, capture_hide, capture_show, show_fig

dataset='dr3' if 'dr3' in sys.argv else 'dr4'

sns.set_theme('notebook' if 'talk' not in sys.argv else 'talk', font_scale=1.25) 
if 'dark' in sys.argv:
    plt.style.use('dark_background') #s
    plt.rcParams['grid.color']='0.5'
    dark_mode=True
else:
    dark_mode=False
    plt.rcParams['figure.facecolor']='white'
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

           
class Mystery:
    """ Implement selection of the mysterious pulsar-like sources
    """
    class Triangle:
        def __init__(self, ax, x=(0.06,0.55), y=(0.04, 0.54)):
            a,b = x
            c,d = y
            alpha = (c-d)/(b-a)
            beta = c-alpha*b
            # inside is in the triangle or rectangle at origin
            self.inside = lambda x,y: ((x>a) & (y>c)  & (y < alpha*x+beta ))  | ((x<a) & (y<c))
            if ax is not None: ax.plot([a,b,a,a], [c,c,d, c], '--', color='1.', lw=2)
    
        def __call__(self, df, x, y):
            return self.inside(df[x],df[y])     
    class Square(Triangle):
        def __init__(self, ax, a=0.1, b=0.08):
            z=-0.025
            self.inside =  lambda x,y: (x<a) & (y<b)
            if ax is not None: ax.plot([z, a, a, z, z] , [z, z, b,b,z], ':', color='1.', lw=2)
            
    class Rectangle(Triangle):
        def __init__(self, ax, x=(0.06,0.55), y=(0.04, 0.54)):
            a,b = x
            c,d = y
            self.inside = lambda x,y: (x>a) & (y>c)  & (x<b) & (y<d)
            if ax is not None: ax.plot([a,b,b,a,a], [c,c,d,d, c], '--', color='1.', lw=2)
   
    def __init__(self, df, hue_order, ax=None):
        self.df =df
        self.r = self.Rectangle(ax=ax, x=(0.05,0.4), y=(0.05,0.2))
        self.s = self.Square(ax=ax, a=0.1, b=0.15)
        self.info= pd.Series(data= dict( ( (st, sum(self.selected(st)))
                                        for st in  hue_order)),
                        name='selected count')
        self.mystery_df =df[df.source_type=="unid-pulsar"][self.selected('unid-pulsar')] 
        
    def selected(self, st):
            pars =self.df[self.df['source_type']==st],  'psr_kde', 'msp_kde' 
            return self.s(*pars) | self.r(*pars)

class UnidAnalysis( Diffuse, MLspec):
    
    def __init__(self, title=None):
        if title is None: title = f'Unid analysis with {dataset}'
        show(f"""# {title}""")
        filename = f'files/{dataset}_classification.csv' #pulsar_summary.csv'
        self.df = df =pd.read_csv(filename, index_col=0)
        print(f"""read summary file `{filename}` """)
        self.dark_mode = dark_mode
        
        df['log_epeak'] = np.log10(df.Ep)
        df['log_fpeak'] = np.log10(df.Fp).clip(-2,5)  
        df['log_signif'] = np.log10(df.significance)

        self.size_kw = dict(size='log_signif', sizes=(10,100))
        self.hue_kw = dict(hue='source_type', 
                          hue_order='unid-pulsar msp psr'.split(),
                          palette=self.palette)
        if dark_mode:
            self.hue_kw.update(palette='yellow magenta cyan'.split(), edgecolor=None)
        else:
            self.hue_kw.update(palette='green red blue'.split())

    def __repr__(self):
        return f"""UnidAnalysis applied to 4FGL-{dataset.upper()} \n{super().__repr__()}
            """

    def check_hpm(self):
        # for diffuse plotting--avoid overhead if not needed 
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
    
    def _plot_psr(self, ax, hue_order=None, s=50, df=None ):
        df = self.df if df is None else df
        for hue, marker, color in zip(self.hue_kw['hue_order'] if hue_order is None else hue_order, 
                                    '*oD',   self.hue_kw['palette']):
            t =  df[df.loc[:,self.hue_kw['hue']]==hue]
            ax.scatter(t, marker=marker, s=s, color=color, label = f'({len(t)}) {hue}')
        ax.legend(fontsize=14)
        
    def ait(self, hue_order=None):
        """Aitoff plot of the galactic diffuse showing locations of pulsars and the unid-pulsar category.
        """
        self.check_hpm()
        ax = self.ait_plot(figsize=(20,8))#cmap='gist_gray', log=False,);
        self._plot_psr(ax, hue_order=hue_order, s=10)
        return ax.figure

    def zea(self, *args, size=90, hue_order=None, df=None):
        """ZEA projection of the galactic diffuse with positions of pulsars and the unid-pulsar category"""
        self.check_hpm()
        ax = self.zea_plot(*args, size=size)
        self._plot_psr(ax, df=df, hue_order=hue_order)
        ax.grid(color='0.5', ls='-')
        # ax.legend(loc='upper left');
        return ax.figure

    def pulsar_pairplot(self, vars):
        """The corner plot with KDE contours for the spectral features and the diffuse value.  
        """
        data=self.df
        kw = self.hue_kw.copy()
        kw.pop('edgecolor', '')
        return sns.pairplot(data, kind='kde', vars=vars,  corner=True, **kw).figure


    def pulsar_kde(self,  df=None,
               vars='log_epeak log_fpeak d diffuse'.split()):
        """
        Calculate KDE for the pulsar types {pulsars}, using the {vars} features,
        then apply normalized value to all sources, adding columns with "_kde" appended.
        """
        from pylib.kde import Gaussian_kde
        df = self.df if df is None else df    

        for name, sdf in df.groupby(self.hue_kw['hue']):
            if name in self.psr_names[:2]: #pulsars:
                gde = Gaussian_kde(sdf,  vars)
                u = gde(df)
                df[name+'_kde'] = u/np.max(u)
        return df

    def plot_kde(self,  order,  df=None, height=4):
        """Scatter plots of the `msp` vs `psr` KDE probabilities for the shown source types.
        """
        data = self.df if df is None else df
        g = sns.FacetGrid(data, col='source_type', col_wrap=3, height=height,
                        col_order=order, sharex=True, sharey=True)
        g.map(sns.scatterplot, 'msp_kde', 'psr_kde', s=10, color=self.palette[0]);
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
        
            sns.scatterplot(data[data.source_type=="unid-pulsar"], # this should not be needed 
                            ax=ax, x=x,y=y, hue=hue_kw['hue'], hue_order=hue_kw['hue_order'][0:1], 
                            palette=self.palette[0:1], edgecolor='none',  alpha=0.5, legend=False)
            ax.set(title=title, **kwargs)
            ax.get_legend().set_title('KDE contours')

        fig, (ax2,ax1) = plt.subplots(ncols=2, figsize=(15,7), gridspec_kw=dict(wspace=0.3))
        kscat( ax=ax1, title='Spectral shape',  x='log_epeak', y='d', 
            **epeak_kw(), ylim=(-0.2,2.1));
        kscat( ax=ax2, title='Fluxes', y='log_fpeak', x='diffuse',
            **fpeak_kw('y'), xticks=np.arange(-1,2.1,1), xlabel='log diffuse flux')
        return fig

    def fp_vs_diffuse(self ):
        """Scatter plot of $F_p$ vs. DEF. 
        """
        fig, ax = plt.subplots(figsize=(10,8))
        data = self.df
        sns.scatterplot(data, ax=ax, x='diffuse',y='log_fpeak', 
                        **self.hue_kw, **self.size_kw );
        plt.legend(loc='upper right', fontsize=12,bbox_to_anchor=(1.1,1.1));
        ax.set(**self.fluxticks('x'), **fpeak_kw('y'), ylim=(-2,4));
        return fig
    
    def flux_ratio_vs_diffuse(self):
        """Flux ratio vs diffuse.
        """
        data =df = self.df
        fig, ax = plt.subplots(figsize=(12,6))
        log_flux_ratio = df.log_fpeak-df.diffuse
        sns.scatterplot(data, x='diffuse', y=log_flux_ratio, ax=ax,
                        **self.hue_kw, **self.size_kw)\
            .set(ylim=(-2,3),xlim=(-1,3), xticks=np.arange(-1,3,1),
                ylabel='log flux ratio', xlabel='log diffuse');
        return fig
    
    def significance_vs_flux_ratio(self):
        """Significance vs flux ratio.
        """
        data = df = self.df
        log_flux_ratio = df.log_fpeak-df.diffuse
        fig, ax = plt.subplots(figsize=(12,6))
        skw = self.size_kw.copy(); skw.update(sizes=(10,100))
        sns.scatterplot(data, x='log_signif', y=log_flux_ratio, 
                        ax=ax, **self.hue_kw, **skw).set(ylabel='log flux ratio');
        return fig

        
    def select_mystery(self):
        """Scatter plot of normalized pulsar KDE values, showing selection regions. 
        """
        df = self.df
        fig, ax = plt.subplots(figsize=(8,8))
        # size_kw = dict(size='log TS', sizes=(20,100) )
        # hue_kw = dict(hue='source_type', hue_order='young MSP UNID-PSR'.split(),
        #              palette='yellow magenta cyan'.split(), edgecolor=None)
        x,y = 'psr_kde', 'msp_kde'
        sns.scatterplot(df, ax=ax,  x=x, y=y, s=10,  **self.hue_kw)#, **size_kw);
        update_legend(ax, df, hue='source_type' )
        ax.set(xlabel='Normalized young probability', ylabel='Normalized MSP probability')

        mys = Mystery(self.df, self.hue_kw['hue_order'], ax=ax)
        self.mystery_info = mys.info   
        # tag the selected unid-pulsar
        self.mystery_df = mys.mystery_df# df[df.source_type=="unid-pulsar"][mys.selected('unid-pulsar')]
        
        return fig

    def curvature_plots(self, dfm, dfp):
        """Curvature $d$ distributions. Upper panels: histogram, lower panels: 
        scatter plot with the uncertainty $\\sigma_d$. Left side is the "mystery" sources, right the pulsars.
        """
        def plot_curvature(df,  ax1, ax2):    
            global fgl
            if 'd_unc' not in df:
                fgl = Fermi4FGL(dataset) if 'fgl' not in globals() else fgl
                dunc= pd.Series(2*fgl.field('Unc_LP_beta'), index=fgl.index, name='d_unc').loc[df.index]
            else:
                dunc = df.d_unc
            sns.histplot(df, ax=ax1, x='d', element='step',bins=np.linspace(0,2,21), edgecolor='white');
            
            sns.scatterplot(df, ax=ax2, x='d', y=dunc, s=20)
        
            ax2.set(ylabel ='$\sigma_d$', ylim=(0,1.5), xticks=np.arange(0,2.01,0.5),
                    yticks=np.arange(0,1.2, 0.5))
            ax2.axvline(4/3, color='pink', ls='--', 
                        label=f"""4/3 "limit"\n> 4/3: {100*sum(df.d>4/3)/len(df):.0f}% """)
        
            ax2.legend(loc='upper center');
        
        fig = plt.figure(figsize=(16,8))
        gs = fig.add_gridspec(2,2, height_ratios=(1,3),hspace=0.05)
        ax1 = fig.add_subplot(gs[0,0])
        ax2 = fig.add_subplot(gs[1,0], sharex=ax1)
        ax3 = fig.add_subplot(gs[0,1,])
        ax4 = fig.add_subplot(gs[1,1], sharex=ax3)

        plot_curvature(dfm, ax1, ax2)
        plot_curvature(dfp, ax3, ax4)
        return fig

def unid_doc():

    #=================================================================================================================
    self = UnidAnalysis(title=f'Unid-pulsar analysis ({dataset.upper()})')
    show(f'[Unid-pulsar analysis Confluence page]'\
        '(https://confluence.slac.stanford.edu/display/SCIGRPS/Unid+analysis)')
    show_date()
    show(f"""This section examines the properties of the "unid-pulsar" sources, the unassociated sources predicted to be pulsars.
    
    There clearly must be young pulsars (`psr`) and MSPs (`msp`) in this set. The point of this section is to compare the properties,
    estimate how many of each there are, and discuss how the remainder are unique.

    We add the diffuse energy flux background discussed in the Diffuse Background section to the three spectral features, and
    start with a "corner" plot showing the four distributions and correlations.
    """)
    fignum = FigNum(3, 0.1)
 
    vars = 'diffuse log_fpeak log_epeak d'.split()
    pulsars = list(self.psr_names) #self.hue_kw['hue_order'][1:]

    show(f"""###  Corner plot for spectral properties and diffuse
    """)
    show_fig(self.pulsar_pairplot, fignum=fignum.next, vars=vars)

    show("""The upper and lower corners show separations of the three source types. In the following plots we expand those,
    with a scatter plot for the `unid-pulsar` sources.
    """)
    show_fig(self.kde_with_scatter, fignum=fignum.next)
    

    show(f"""## Apply KDE  """)
    self.pulsar_kde()#  vars=vars, pulsars=pulsars[:2], )

    show(self.pulsar_kde.__doc__.format_map(dict(pulsars=pulsars[:2], vars=vars)))

    show(f"""### KDE scatter plots
         In Figure {fignum.next} we show scatter plots of the two KDE values for actual pulsars in the upper row,
         and the three most numerous pulsar predictions in the lower row.
         """)

    show_fig( self.plot_kde, order=pulsars + [x+'-pulsar' for x in 'unid bcu spp'.split()],
             fignum=fignum)

    show(f"""The upper row, with the three actual pulsar classes, shows that 
         separation for the `psr` and `msp` class is good, while `glc` shows an
         apparent mixture. 
         
         Now we examine the KDE values for the largest pulsar predictions.""")

    show(f"""## Use KDE values to select non-pulsars
         Figure {fignum.next} show the KDE values for the `unid-pulsar` sources,
         as well as the `msp` and 'psr` sources from which the KDE's were derived.
         Also shown are two square regions used to select a subset of the 
         `unid-pulsar` sources that are less likely to be real pulsars.  
         """)
    show_fig(self.select_mystery, fignum=fignum )
    show('Selected in regions shown',)
    show(self.mystery_info)
    show(f"""## Curvature plots 
         """)
    dfp = self.df[self.df.source_type.apply(lambda s: s in 'msp psr'.split())]
    show_fig(self.curvature_plots, self.mystery_df, dfp)
    show('---')
    return self
#-------------------------------------------------------------
self = None

if 'unid-doc' in sys.argv: 
    self = unid_doc()

def diffuse_doc(self, fn):

    show("""# Diffuse background as a feature""")
    show_date()
         
    show("""
    In this section, we examine the Galactic diffuse component of the gamma-ray sky flux, 
    evaluating it at the position of each source, then treating it like a "feature".
    The following plots examine this for the known pulsars, and the "unid-pulsar" category.
    """)
    #------
    show(f"""Figure {fn.next} below shows the  diffuse spectral energy distribution
    $E^2 dN/dE$ at galactic longitude 0, for several values of the latitude.
    """)
    show_fig(self.eflux_plot, fignum=fn)
    show("""It peaks around 1 GeV. We will associate this value with each source. 
    """)
    #-----
    show("""### Diffuse correlation for Unid predictions
         Now we show the diffuse correlation plots:""")
    show_fig( self.plot_diffuse_flux, unid=True, fignum=fn.next)
    show("""Recall that the classification did not use position information. 
         This verifies that the predicted blazar distributions are approximately isotropic--we expect
         an enhancement near the Galactic plane since there is a gap in the blazar catalogs here.
         The pulsar predictions are highly Galactic. """)
    #-------
    show(f"""Figure {fn.next} shows a sky map of the 1 GeV energy flux with the positions of the known pulsars
    and the unid-pulsar category.
    """)
    show_fig(self.ait, fignum=fn)

    show(f"""## Diffuse energy flux distributions
    In Figure {fn.next} we show the distributions of the diffuse energy flux (DEF) 
    for the three categories of interest. Note the separation between the `psr` and
    `msp` categories--the correlation with the DEF corresponds to the distinction
    according to age, in that the `psr` or young pulars tend to be close the the
    Galactic plane, where the flux is largest, while `msp`s, being very old, have
    migrated from where they were formed.
    """)
    #------
    show_fig(self.plot_diffuse_flux, fignum=fn ); 
    show(f"""The lower panel in Figure {fn} shows the empirical cumulative 
    distribution functions (ECDF) for the three categories. The gray area delimits two 
    extremes, complete isotropy, and a uniform distribution in the Galactic ridge. 
    """)
    #------

    show(f"""### Peak flux vs. diffuse
        The DEF, when compared with the peak flux $F_p$, has another role, displaying
        apparent thresholds for detection and association, as seen in Figure {fn.next},
        the correlation.
        """)
    show_fig(self.fp_vs_diffuse,  fignum=fn)
   
    show(f"""The ratio of the peak flux to the DEF, in Figure {fn.next}, shows the 
    thresholds pretty clearly in this ratio.""")
    show_fig(self.flux_ratio_vs_diffuse, fignum=fn)
    
    show(f"""Finally, we note that the ratio, basically a signal to noise, correlates
    well with the source significance.""")
    #-----
    show_fig(self.significance_vs_flux_ratio, fignum=fn.next)
    return self

    
if 'diffuse-doc' in sys.argv: 
    fn = FigNum(n=2, dn=0.1)
    with capture_hide('Setup printout') as setup :
        self = UnidAnalysis(title="")
        self.check_hpm()
    self = diffuse_doc(self,fn)

# set self
if self is None:
    self = UnidAnalysis(title=f'UnidAnalysis ({dataset.upper()})')
    show_date()