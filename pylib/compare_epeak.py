from fermi_sources import *

from utilities.catalogs import *

show(f"""# Compare peak energies

     The 4FGL spectral fitting procedure tends to favor selection of a 
     power-law over a log-parabola for weak sources. 
     This represents a loss of potential information in the curvature.
     A loss of information about the peak may be made up by the pivot energy.
""")

with capture_hide("""Load the 4FGL-DR4 catalog to look the spectral function""") as cat:
    fgl = Fermi4FGL()
    # uw  = UWcat('uw1410')
fs_data = FermiSources('files/fermi_sources_v2.csv')

a = set(fs_data.df.index); b=set(fgl.index)
good_id = list(a.intersection(b))
df = fs_data.df.loc[good_id]
df['log_fgl_epeak'] = fgl.loc[df.index].specfunc.apply( lambda f:  SedFun(f).peak)

show(f""" The following plot compares the power-law and/or spectral shapes used
     in the uw1410 and 4FGL-DR4 analyses. The peak energy is the energy within the
     bounds 0.1 to 1000 GeV at which the SED, or $E^2 dN/dE$, has its maximum.
     
     """)

fig, ax =plt.subplots(figsize=(8,8))
sns.scatterplot(df, x='log_epeak', y='log_fgl_epeak',ax=ax);
ax.set(xlabel='uw1410', ylabel='4FGL-DR4', 
       title=r'$\log( E_{peak}/1GeV)$ comparison',
       yticks = [-1,0,1,2,3])
M = np.histogram2d(df.log_epeak, df.log_fgl_epeak, 
                   bins=[-1.0, -0.99, 2.9, 3.0])[0].astype(int)
agree = np.sum(M.diagonal())
opposite = M[2,0]+M[0,2]
uw_lim = M[2,1] + M[0,1]
fgl_lim  = M[1,2] + M[1,0]
show(fig, caption='')
show(f"""The sources on the boundaries represent where one or both
SED's are power-laws, soft ($\Gamma>2$) or hard ($\Gamma<2$).

The following numbers are counts of sources in the center, corners or edges:
| Agree (central and diagonal corners) | 4FGL only | UW only | Opposite corners
|-----:|----:|---:|---:
| {agree} | {fgl_lim} ({100*fgl_lim/np.sum(M):.0f}%)| {uw_lim} | {opposite}

So there are eight times as many sources for which the 4FGL-DR4 spectrum
is a power-law in disagreement wtih uw1410 with respect to the opposite.
""")

show(f"""## Projected Epeak distributions

Here we look at the uw1410 Epeak distribution within the limits,
and the fraction of that for which the 4FGL-DR4 measurement was also not at limit.
The difference represents the 
""")
fig, ax = plt.subplots(figsize=(6,4))
bins = np.linspace(-0.999,2.999,51)
sns.histplot(df,x='log_epeak', bins=bins, label='all', ax=ax);
sns.histplot(df.query('-1<log_fgl_epeak<3'),x='log_epeak', bins=bins, label='4FGL not at limit', ax=ax);
ax.legend()
show(fig)