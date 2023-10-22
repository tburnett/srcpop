"""
"""
import numpy as np
import matplotlib.pyplot as plt
import healpy
import seaborn as sns
from .skymaps import healpy, HPmap, _process_args
from .ipynb_docgen import show
from .fermi_sources import update_legend

class Diffuse:
    
    def __init__(self, filename='gll_iem_v07_hpx.fits', field=11, energy=893):
        import os
        self.energy = energy
        fits_file = os.path.expandvars(f'$FERMI/diffuse/{filename}')
        self.diffuse_hpm = HPmap.from_FITS(fits_file, field=field, name='',)
        self.unit = r'$\rm{eV\ cm^{-2}\ s^{-1}\ deg^{-2}}$' #self.diffuse_hpm.unit
        show(f"""* Load diffuse file,  `{fits_file}`<br>  unit={self.unit}<br>  select energy= {energy} MeV""")
        
        # convert units of the HEALPix array in the HPmap guy
        dmap = (np.log10(self.diffuse_hpm.map) 
                + 2*np.log10(energy) +6   # multiply by E^2, convert from MeV to eV
                - 2*np.log10(180/np.pi) # convert from sr to deg^2
                ) 
        self.diffuse_hpm.map = dmap
        self.dark_mode = plt.rcParams['figure.facecolor']=='black'
        
        # mask for the ridge selection
        glon,glat = healpy.pix2ang(self.diffuse_hpm.nside, range(len(dmap)), lonlat=True)
        glon[glon>180]-=360
        self.ridge_mask= (np.abs(glat)<2) & (np.abs(glon)<45)
        
    def eflux_plot(self, glon=0):
        from astropy.io import fits
        import os

        filename='gll_iem_v07_hpx.fits'
        fits_file = os.path.expandvars(f'$FERMI/diffuse/{filename}')

        with fits.open(fits_file) as hdus:
            data = hdus[1].data
            energies = hdus[2].data.field(0)

        fig, ax = plt.subplots(figsize=(6,6))
        ax.set(xlabel='Energy (GeV)',
            ylabel=r'Energy flux (eV s-1 cm-2 deg-2)')#\ ($\rm{MeV \s^{-1}\ cm^{-2}\ sr^{-1}}$)') 
        for b in (-90,-30, -2 ,0,2,30,90):
            pix = healpy.ang2pix(512, glon,b, lonlat=True)
            ax.loglog(energies/1e3, energies**2*data[pix]*1e6/3282.80635, label=f'{b}');
        ax.legend(title='b', fontsize=12);
        ax.set_title(f'Diffuse energy flux at $l$={glon}');
        return fig
    
    def get_values_at(self, *args):
        """Return log10 values in units of eV s-1 cm-2 deg-2
        """
        return self.diffuse_hpm(_process_args(*args)[0])
        
    def plot_limits(self, ax):
        sns.ecdfplot(ax=ax, x=self.diffuse_hpm.map, label='isotropic', ls='--', color='grey')
        sns.ecdfplot(ax=ax, x=self.diffuse_hpm.map[self.ridge_mask], label='ridge', ls=':', color='grey')
        ax.legend()

    def ait_plot(self, figsize=(30,12), **kwargs):
        fig=plt.figure(figsize=figsize)
        kw = dict(log=False, fig=fig, grid_color='grey', pixelsize=1, colorbar=False, 
                        cmap='gist_gray' if self.dark_mode else 'Greys')
        kw.update(kwargs)
        return self.diffuse_hpm.ait_plot(**kw)#, alpha=0.1)

    def zea_plot(self, *args, fig=None, size=10, **kwargs):
        if fig is None:
            fig = plt.figure(figsize=(10,10))
        kw = dict(colorbar=False, log=False,  cmap='gist_gray' if self.dark_mode else 'Greys')
        kw.update(kwargs)
        center, _ = _process_args(*args)
        return self.diffuse_hpm.zea_plot( center,  size=size, fig=fig, axpos=111, **kw, )
        
    def fluxticks(self, x, ):
        ticks =  np.arange(0,3.1,1).astype(int)
        return {x+'ticks':ticks,
                x+'ticklabels' : [f'$10^{{{x}}}$' for x in ticks],
                x+'label': f'Diffuse energy flux at {self.energy:.0f} MeV ({self.unit})' }
        
    def show_diffuse_flux(self,  df, hue_kw, figsize=(8,8), fignum=None, title=None):

        show(f"""## Diffuse flux value at sources """ if title is None else title)
        
        fig, (ax1,ax2) = plt.subplots(nrows=2, figsize=figsize, sharex=True,
                                    gridspec_kw=dict(hspace=0.1))
        hkw = hue_kw.copy()
        hkw.pop('edgecolor', '')
        ax=sns.histplot(df, ax=ax1, x='diffuse', **hkw, bins=25, kde=True, element='step')
        update_legend(ax, df, hue_kw['hue'])
        ax.set(**self.fluxticks('x') )
        ax=sns.ecdfplot(df, ax=ax2, x='diffuse', legend=False,  **hkw)
        self.plot_limits(ax)
        ax.set(**self.fluxticks('x') );
        
        show(fig, fignum=fignum)  

    def show_diffuse_vs_ep(self, df, hue_kw):
        show(f""" ## Diffuse vs peak energy for pulsar-like sources""")
        # fig, ax = plt.subplots(figsize=(15,8))
        data=df
        x,y = 'log_epeak diffuse'.split()
        g = sns.JointGrid(height=12, ratio=4 )
        ax = g.ax_joint
        size_kw = dict(size='log TS', sizes=(20,200) )
        sns.scatterplot(data, ax=ax, x=x, y=y, **hue_kw, **size_kw);
        axis_kw= lambda a, label, v: {f'{a}label':label,f'{a}ticks':np.log10(v), f'{a}ticklabels':v }
        
        ax.set(**axis_kw('x','$E_p$ (GeV)', [0.1, 0.25,0.5,1,2,4]),xlim=np.log10((0.1,6)), 
            **self.fluxticks('y')
            )
        # ax.scatter(orion.log_epeak, orion.diffuse,  marker='o', s=200, color='k', facecolor='none', label='selected')
        hkw = dict(element='step', kde=True, bins=25, **hue_kw, legend=False)
        sns.histplot(data, y=y, ax=g.ax_marg_y, **hkw)
        sns.histplot(data, x=x, ax=g.ax_marg_x, **hkw)
        update_legend(ax, df, hue='source type',  fontsize=12,   loc='lower left')
        show(g.fig)     
