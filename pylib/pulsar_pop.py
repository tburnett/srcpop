from pylib.fermi_sources import *
from pathlib import Path

def get_source_data(filename='files/classification.pkl'):
    f = Path(filename)
    if not f.is_file():
        FermiSources().train_predict(save_to=filename)
        assert f.is_file(), f'Failed to create {filename}'
    df =  pd.read_pickle(filename)
    df.index.name='4FGL-DR4'
    return df

def introduction(npsr=312, nunid=762, nsgu=None):
    show(f"""# LAT pulsars: identified and predicted""")
    show_date()
    show(f"""
        This is a study of LAT sources, comparing the {npsr} *identified* as pulsars in 4FGL-DR4 with the {nunid} unassociated sources
        *predicted* to be pulsars using a [ML classification analysis](machine_learning.ipynb). 
        The predictions were based on the spectral and variability properties measured 
        in the `uw1410` all-sky analysis used to provide seeds for 4FGL-DR4 and a `wtlike` study, respectively.
    
        Here we look carefully at both spectral and positional properties of the two sets to look for inconsistencies.
        
        ### Two possible conclusions:
        Assuming that these are dominated by a single source type, the possibilities are:<br>
        
        **A)** The unid's predicted to be pulsars are mostly undetected pulsars <br>
        **B)** There is a new, unknown galactic population of gamma-ray sources that only *look* like pulsars
         
        Perhaps applicable philosophical principles: Occam's razor and the [duck test](https://en.wikipedia.org/wiki/Duck_test). 
        Both would favor option A. But option A implies that we understand how such a 
        large number of pulsars would be easily detected by the LAT, yet not seen in radio.
    
        """)

class Pulsars:
    def __init__(self):
        from utilities.catalogs import Fermi4FGL, UWcat
        sns.set_context('notebook')
        self.df =  get_source_data()
        
        df = self.df
        glon = df.glon.values
        glon[glon>180] -= 360
        df.glon = glon
        df['abs_glon'] = np.abs(glon)
        df['hemisphere'] = df.glon.apply(
            lambda glon: 'Inner' if (abs(glon)<90) | (glon>270) else 'Outer')
        df['pred_type'] = df.prediction.apply(lambda x: 'Pulsar' if x=='psr' else 'Blazar') 
    
        def actual_type(a):
            if a=='psr':  return 'Pulsar'
            if a in 'bcu bll fsrq'.split(): return 'Blazar'
            return 'other'
        df['actual_type'] = df.association.apply( actual_type)
        df['pulsar_type']= df.apply(plike, axis=1) 

        with capture_hide():
            fcat = Fermi4FGL()
            uw = UWcat().set_index('jname')
        cnamed = list(map( lambda n: n[:-1], filter(lambda n: n[-1]=='c', fcat.index))); 
        # reindex to add "C" 
        self.df.index = [ name +'c' if name in cnamed else name  for name in self.df.index]
        self.df.loc[:,'sgu'] = fcat['flags'].apply(lambda f: f[14])
        self.df.loc[:,'fcat_epeak'] = fcat.specfunc.apply(lambda f: f.sedfun.peak)
        self.df.loc[:,'fcat_curvature']= fcat.specfunc.apply(lambda f: f.curvature())   

        uwsf = uw.loc[self.df.uw_name,'specfunc']
        fpeak = uwsf.apply(lambda x: x.fpeak).values
        self.df['log_fpeak'] = fpeak 

def sinbplots(data, row, row_order, bins=np.linspace(0.01,1,41)):
    (sns.FacetGrid(data,  row=row, row_order=row_order,
                   col='hemisphere',  col_order=['Inner', 'Outer'],
                   aspect=2, height=2.5,margin_titles=True,
                   gridspec_kws=dict(hspace=None),
                  subplot_kws=dict( xlim=(0,1)))
     .map_dataframe(sns.histplot, x='abs_sin_b',  bins=bins,element='step')
     .set_titles(row_template='{row_name} ', col_template='{col_name} Galaxy')
     .set_xlabels(r'$|\sin(b)|$')
    )
    show(plt.gcf())

def select_data(self=None, outfile=None):
    df = self.df if self is not None else Pulsars().df

    data = df.query('actual_type=="Pulsar" |' \
        '(pred_type=="Pulsar" & association=="unid" )').copy()
    data.index.name='4FGL-DR4'
    data['pulsar_id'] = data.apply(lambda x:'IDed' if x.association=='psr' else 'predicted',axis=1)
    t = data.groupby('pulsar_id').size(); t.name='count'
    npred = t.predicted
    nsgu = data.set_index('pulsar_id').loc['predicted'].groupby('sgu').size()[True]
    
    introduction(t.IDed, npred, nsgu)
    if outfile is not None:
        try: 
            data.to_csv(outfile)
            show(f'Wrote data to {outfile}')
        except:
            show(f'Failed to generate output {outfile}')
    return data

def old_corner(data):
    show(r""" ## Source property corner plot 
    This "corner plot" shows the correlations of source properties grouped with the 
    spectral parameters `log_eflux`, `curvature`, and `epeak` at the top, and galactic coordinates 
    `abs_sin_b` and `abs_glon`($|\sin(b)|$, $|l|$) and  at the bottom.""")
    
    (sns.PairGrid(data,hue='pulsar_id',corner=True,
            vars='log_eflux curvature log_epeak abs_sin_b abs_glon'.split())
     .map_diag(sns.histplot,  kde=True, element='step')
     .map_lower(sns.kdeplot,)
     .add_legend(loc='upper right',bbox_to_anchor=(0.5, 0.95))
    );
    show(plt.gcf(), fignum=1, caption='')
    show(""" Notes:
    * Spectral variables:<br>
    curvature vs. eflux: there is a strong correlation in the unids not reflected by the pulsars. 
    Large curvatures for them corresponds to lower energies. But some of this couild be a systematic.
    * Positions:<br>
    The actual pulsars are two populations, MSP and young pulsars, the former dominating high latitudes.
    The unids are broader in $b$, but narrower in $l$.
    * Spectral vs. position:<br>
    Both show correlation between $|b|$ and eflux.  For pulsars it is probably detection efficiency, 
    but there is a clear correlation for the unids. 
    """)

def reset_labels(self, vars, labels):
    def relabel(ax):
        labeldict= dict(zip(vars, labels))
        ax.set(xlabel=labeldict.get(ax.get_xlabel(),''),
               ylabel=labeldict.get(ax.get_ylabel(),''),)
        
    for ax in self.axes.flat: 
        if ax is not None: relabel(ax)

def corner(data, fignum=1,
          vars='log_fpeak log_epeak curvature abs_sin_b abs_glon'.split(),
          labels=[r'$\log(F_{peak})$',r'$\log(E_{peak})$', 'curvature',r'$|\sin(b)|$',r'$|l|$'],
          ):
    show(f""" ## Source property corner plot 
        This "corner plot" shows the correlations of source properties grouped with the 
        spectral parameters {labels[0]}, {labels[1]}, and {labels[2]} at the top, and galactic coordinates 
        {labels[3]} and {labels[4]} at the bottom.""")
              
    pg=(sns.PairGrid(data,hue='pulsar_id',corner=True,
            vars=vars,)
     .map_diag(sns.histplot,  kde=True, element='step')
     .map_lower(sns.kdeplot,)
     .add_legend(loc='upper right',bbox_to_anchor=(0.5, 0.95))
     .apply(reset_labels, vars, labels)
    )
    
    show(plt.gcf(), fignum=fignum, caption='')
    show(r""" Notes:
    * Spectral variables:<br>
    The curvature distribution shows there are many with large values for the predicted
    not reflected by the IDed ones, and clear correlations. We will expand on this below.
    * Positions:<br>
    The actual pulsars are in two populations, MSP and young pulsars, the former dominating high latitudes.
    The unids are broader in $b$, but narrower in $l$. We will distinguish these in a detailed plot below.
    * Spectral vs. position:<br>
    There are clear correlations of both populations in the $F_{peak}$ vs. $|\sin(b)]$ which we look at below.
    """)
                  
def plike(rec):
    class1 = rec.class1
    if not pd.isna(class1) and class1.lower() in ('msp','psr'): 
        return dict(msp='MSP', psr='RPP')[rec.class1.lower()]
    if rec.association=='unid' and rec.prediction=='psr': return 'predicted'

def show_positions(data, fignum=None):
    show(f"""### Source positions
    We saw in Fig. 1 that the position distributions for the pulsar-like unids
    differed from the true pulsars. Here we expand that correlation plot, 
    distinguishing the young, rotationaly-powered pulsars ("RPP") from MSPs ("MSP").
    """)
    x,y,hue = data.glon, data.abs_sin_b, data.pulsar_type
    
    g=sns.JointGrid(  height=10,ratio=4)
    sns.scatterplot(x=x, y=y, hue=hue,size=data.log_fpeak, sizes=(10,200), ax=g.ax_joint)
    sns.histplot(x=x, hue=hue, ax=g.ax_marg_x, element='step', kde=True)
    sns.histplot(y=y, hue=hue, ax=g.ax_marg_y, element='step', kde=True)
    for ax in (g.ax_joint, g.ax_marg_x):
        ax.set(xlim=(180,-180), xticks=np.arange(180,-181, -45))
    for ax in (g.ax_joint, g.ax_marg_y):
        ax.set(ylim=(0,1))
    g.set_axis_labels(xlabel='$l$', ylabel=r'$|\sin(b)|$')
    for ax in (g.ax_marg_x, g.ax_marg_y):
        ax.get_legend().set_visible(False)
   
    show(plt.gcf(), fignum=fignum, caption='')
    show(r"""
        Here we see that the pulsar-like unids are found at all latituedes and longitudes,
         but there is a concentration for $|l|<45^\circ$ and $|b|<20^\circ$.
         """)

def show_flux_vs_b(data):
    show(rf"""### Flux $vs.$ latitude
    In Fig. 2, the unid's seem concentrated within longitudes $|l|<45^o$, and out
    to $|\sin b|$ =0.4. Here we explore that subset, and follow up on the flux 
    dependence:
    """)
    ax=sns.scatterplot(data.query('-45<glon<45 & 0.01<abs_sin_b<0.4 & log_eflux<-10'), 
                    x='abs_sin_b', y='log_fpeak', #'log_eflux',
                    hue='pulsar_type', #data.apply(plike, axis=1) ,
                   )
    show(ax.figure, fignum=3, caption='')

def show_flux_vs_curvature(data):
    show(f"""### Flux vs. curvature
    Using the same location selection as above, here we explore the spectral 
    correlation mentioned above. 
    """)
    ax=sns.scatterplot(data.query('-45<glon<45 & 0.01<abs_sin_b<0.4 & log_eflux<-10'), 
                    x='curvature', y='log_eflux',
                    hue=data.apply(plike, axis=1) ,
                   )
    show(ax.figure, fignum=4, caption='')
    show(f"""Clearly there is a dramatic dependence, weaker sources having 
    narrower SED profiles. Is this really a source property, or is it a measurment 
    systematic?
    """)

def describe(data):
    t = data.groupby('pulsar_type').size(); t.name='Counts'
    show(t)
    return data    

def subset(data, quiet=False):
    df = data.query('-45<glon<45 & 0.017<abs_sin_b<0.259') #-0.523<log_epeak<0.523 &

    if not quiet:
        show(rf"""
        Position selection: $-45^\circ < l < 45^\circ$ and  $1^\circ < |b| < 15^\circ$
        
        This results in {len(df)} sources, of which the types are: 
        """)  
        describe(df) 
    return df


def show_peak_properties(data, select=describe, 
                        heading="""## Spectral properties """):
    show(heading)
    axkw = dict( xlabel='Peak Energy (GeV)', xticks=[-1,0,1], xticklabels='0.1 1 10'.split(),
            ylabel='Peak Flux (eV cm-2 s-1)', yticks=np.arange(-1,4.1,1), 
            yticklabels='0.1 1 10 100 $10^3$ $10^4$'.split(), ylim=(None, 4.5), )

    (sns.JointGrid(data if select is None else select(data),
                   x='log_epeak', y='log_fpeak', 
                   hue='pulsar_type', hue_order='RPP unid MSP'.split(),  
                   height=8, ratio=3)
     .plot_joint(sns.scatterplot, s=100, alpha=0.8)
     .plot_marginals(sns.histplot, element='step', kde=True)
     .apply(lambda s: s.ax_joint.set( **axkw ))
     
    )
    show(plt.gcf(), )

def curvature_vs_fpeak(data, fignum=None):
    show(f"""### Curvature *vs.* $F_{{peak}}$
    """)
    y,x,hue = data.curvature, data.log_fpeak, data.pulsar_type
    g=sns.JointGrid(  height=8,ratio=5)
    sns.scatterplot(x=x, y=y, hue=hue,size=data.log_fpeak, sizes=(10,200), ax=g.ax_joint)
    sns.histplot(x=x, hue=hue, ax=g.ax_marg_x, element='step', kde=True)
    sns.histplot(y=y, hue=hue, ax=g.ax_marg_y, element='step', kde=True)
    
    for ax in (g.ax_joint, g.ax_marg_x):
        ax.set(xticks=np.arange(0,5.1,2), xticklabels=' 1 100 $10^4$'.split(),
              xlim=(-1.5,5))
    g.set_axis_labels(ylabel='Curvature', xlabel=r'$F_{peak}$ (eV cm-2 s-1)')
    for ax in (g.ax_marg_x, g.ax_marg_y):
        ax.get_legend().set_visible(False)
    show(g.ax_joint.figure, fignum=fignum, caption='')

def sed_plots(data):
    show(f"""## SED plots 
    These are SED plots for the pulsar-like unids in the selected region.
    """)
    
    cols = 'pulsar_type ts nbb var glon glat log_fpeak log_epeak curvature sgu uw_name'.split() 
    # df = data.query('association=="unid" & prediction=="psr" & -45<glon<45 & abs_sin_b<0.2').sort_values('log_fpeak')
    df = subset(data).query('pulsar_type=="unid"')
    
    
    
    issgu = df[df.sgu==True]; len(issgu)
    
    show(f""" SGUs: {len(issgu)}""")
    sedplotgrid(issgu[cols], nrows=19, ncols=10,  figsize=(10, 19/2))
    
    notsgu= df[df.sgu==False]; 
    show(f""" Not SGUs: {len(notsgu)}""")
    sedplotgrid(notsgu[cols], nrows=13, ncols=10, )

def spectral_correlations(data, fignum=None):

    
    show(r"""### Curvature vs peak parameters
        We are characterizing spectra with properties of the peak in the SED: its flux, 
         $F_{peak}$, and
        energy, $E_{peak}$. The latter is bounded to be greater than 100 MeV. 
         """)
    fig, (ax1,ax2) = plt.subplots(ncols=2, figsize=(15,7), sharey=True, 
                                 gridspec_kw=dict(wspace=0.05))
    sns.scatterplot(data, y='curvature',x='log_epeak', hue='pulsar_type', alpha=0.5,
                    size='log_fpeak', sizes=(10,200),ax=ax1);
    sns.kdeplot(data, y='curvature',x='log_epeak', hue='pulsar_type', ax=ax1);
    ax1.set(xticks=[-1,0,1], xticklabels='0.1 1 10'.split(), xlabel=r'$E_{peak}$ (GeV)',
           xlim=(-1.5,1), yticks=np.arange(0,1.1,0.25))
    
    sns.scatterplot(data, y='curvature',x='log_fpeak', hue='pulsar_type',alpha=0.5,
                    size='log_fpeak', sizes=(10,200),ax=ax2);
    sns.kdeplot(data, y='curvature',x='log_fpeak', hue='pulsar_type',ax=ax2)
    ax2.set(xlabel=r'$F_{peak}$ (eV cm-2 s-1)', xticks=[0,2,4,6], 
            xticklabels='1 100 $10^4$ $10^6$'.split(), xlim=(-1.9,5.5));
    ax2.get_legend().set_visible(False)
    show(fig, fignum=fignum, caption='')

def sinb_vs_fpeak(data, fignum=None):
    show(r"""### $|\sin(b)|$ vs. $F_{peak}$ """)
    fig, ax = plt.subplots(figsize=(8,6))
    sns.scatterplot(data, y='abs_sin_b',x='log_fpeak', hue='pulsar_type', alpha=0.5,
                        size='log_fpeak', sizes=(10,200),ax=ax);
    sns.kdeplot(data, y='abs_sin_b',x='log_fpeak', hue='pulsar_type', ax=ax);
    ax.set(xlim=(-1.5,3.9), ylim=(-0.05,0.5), 
           xlabel='$F_{peak}$ (eV cm-2 s-1)', ylabel=f'$|\sin(b)|$', 
           xticks=[0,2,], xticklabels='1 100'.split(), );
    show(fig, fignum=fignum, caption='')

def position_cut(data, quiet=False, l_lim=(-45,45), b_lim =(2,15)):
    # df = data.query('-45<glon<45 & 0.034<abs_sin_b<0.259') #-0.523<log_epeak<0.523 &
    l, abs_b = data.glon, np.abs(data.glat)
    df = data[ (l>l_lim[0]) & (l<l_lim[1]) & (abs_b>b_lim[0]) & (abs_b < b_lim[1])]
    if not quiet:
        show(rf"""
        Position cut:  $l$ in {l_lim} and $|b|$ in {b_lim}. 
        
        This results in {len(df)} sources, of which the types are: 
        """)  
        describe(df) 
    return df
def spectral_cut(data, quiet=False, cmin=0.5, fpeak_max = 90):
    df = data[(data.curvature>cmin) & (data.log_fpeak<np.log10(fpeak_max))]
    if not quiet:
        show(f"""Spectral cut: $F_{{peak}}$ < {fpeak_max} and curvature>{cmin} 
        """)
    describe(df)
    return df 
def special(data, fignum=None):
    show(f"""## Selection of a special region
    Here we select a subset of the pulsar-predictions to enhance this signal.      
    """)
    special = spectral_cut(position_cut(data))
    cols = 'ts glat glon log_fpeak log_epeak curvature sgu uw_name'.split()
    df = special.query('pulsar_type=="predicted"')[cols].sort_values('ts', 
                                                        ascending=False)
    show("""### SED plots """)
    sedplotgrid(df)
    return df

def main():
    import warnings
    warnings.filterwarnings("ignore")
    sns.set_context('talk')
    data = select_data()
    corner(data, fignum=1)
    show(f"""
    ## Three pulsar categories 
    There are two distinct pulsar types: young, rotation-powered (RPP), and those spun up by a
    companion star to millisecond periods, (MSP). These were combined for the classification process, but since
    there are small spectral and large position difference, we distinguish them below. The unidentified sources
    predicted to be pulsars represent a third "type".  The numbers:           
    """)
    describe(data)
    show_positions(data, fignum=2)
    spectral_correlations(data, fignum=3)
    sinb_vs_fpeak(data, fignum=4)          
    df = special(data, fignum=5)
    df.index.name = '4FGL-DR4'
    filename = 'files/psr_candidates.csv'
    pd.set_option('display.precision',3)  
    df.to_csv(filename)
    show(f"""Wrote the selected pulsar candidates, sorted by TS, to `{filename}` with {len(df)} entries.""")

#-------------------------------------------------------------------------------    
if len(sys.argv)>1:
    if sys.argv[1]=='main':
        main()


