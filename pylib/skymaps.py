"""

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.io import fits
import healpy

#| export
def _process_args(*args):
    """ 
    Helper for map displays position specifictio
    Expect  args to contain:
    * a SkyCoord object (perhaps a list of positions)
    * lists of l, b in degrees
    * a DataFrame with glon and glat columns
    * The name of a source   

    Returns a SkyCoord object and the remaining args
    Subsequently will convert to radians for AIT or pixels for ZEA. 
    """
        
    if len(args)==0: raise ValueError('No args')
    arg, rest = args[0], args[1:]
    if isinstance(arg, pd.DataFrame):
        df = arg
        sc  = SkyCoord(df.glon, df.glat, unit='deg', frame='galactic')
    elif isinstance(arg, SkyCoord):
        sc = arg.galactic
    elif type(arg)==str:
        sc = SkyCoord.from_name(arg).galactic
    elif len(args)>1:
        l,b = args[:2]
        rest = args[2:]
        sc = SkyCoord(l,b, unit='deg', frame='galactic')
    else:
        raise ValueError('Expect DataFrame with glat,glon or SkyCoord or name or l,b')
    return sc, rest

def _to_radians(*args):
 
    sc, rest = _process_args(*args)
    # convert to radians 
    l, b = sc.l.deg, sc.b.deg
    x  = -np.radians(np.atleast_1d(l))
    x[x<-np.pi] += 2*np.pi # equivalent to mod(l+pi,2pi)-pi I think
    y = np.radians(np.atleast_1d(b))
    return [x,y] + list(rest) 

def _to_pixels(*args, wcs):
    sc, rest = _process_args(*args)
    return wcs.world_to_pixel(sc) + tuple(rest) 


class AitoffFigure():

    """ Implement plot and text conversion from (l,b) in degrees, or a SkyCoord.

    """
    def __init__(self, fig=None, figsize=(10,5), **kwargs):
        self.fig = fig or plt.figure(figsize=figsize)
        if len(self.fig.axes)==0:
            ax=self.fig.add_subplot(111, projection='aitoff')
            ax.set(xticklabels=[], yticklabels=[], visible=True)
            ax.grid(color='grey')
        self.ax = self.fig.axes[0]
        assert self.ax.__class__.__name__.startswith('Aitoff'), 'expect figure to have aitoff Axes instance'
        self.ax.set(**kwargs)

    def plot(self, *args, **kwargs):
        self.ax.plot(*_to_radians(*args), **kwargs)

    def text(self, *args, **kwargs):
        self.ax.text(*_to_radians(*args), **kwargs)

    def scatter(self, *args, **kwargs):
        return self.ax.scatter(*_to_radians(*args), **kwargs)
    def __getattr__(self, name):
        # pass everything else to self.ax
        return self.ax.__getattribute__( name)


def ait_plot(mappable,
        pars=[],
        label='',
        title='',
        fig=None, ax=None, fignum=1, figsize=(20,8),
        pixelsize:'pixel size in deg'=1.0,
        projection='aitoff',
        cmap='inferno',
        vmin=None, vmax=None,  vlim=None, pctlim=None,
        log=False,
        colorbar:bool=True,
        cblabel='',
        unit='',
        grid_color='grey',
        cb_kw={},
        axes_pos=111,
        axes_kw={},
        tick_labels=False,
        alpha:'apply to pcolormesh'=None,
        ):

    """
    """
    from matplotlib import colors
    #
    # code inspired by https://stackoverflow.com/questions/46063033/matplotlib-extent-with-mollweide-projection
    # make a mesh grid
    nx, ny = int(360/pixelsize), int(180/pixelsize)
    lon = np.linspace(180,-180, nx) # note reversed
    lat = np.linspace(-90., 90, ny)
    Lon,Lat = np.meshgrid(lon,lat)

    #  an arrary of values corresponding to the grid
    # dirs = SkyDir.from_galactic(Lon, Lat)
    dirs = SkyCoord(Lon, Lat, unit='deg')
    arr = mappable(dirs, *np.atleast_1d(pars))

    if ax is not None:
        fig = ax.figure
        assert ax.__class__.__name__.startswith('AitoffAxes'), 'Require that be a AitoffAxes object'
    else:
        fig = plt.figure(figsize=figsize, num=fignum) if fig is None else fig
        # this needs to be more flexible
        ax = fig.add_subplot(axes_pos, projection=projection, **axes_kw)

    # reverse longitude sign here for display
    if pctlim is not None:
        vlim=np.percentile(mappable.map, np.array(pctlim)) #[40, 99.98])).round(),

    if vlim is not None:
        vmin, vmax = vlim

    if log:
        norm = colors.LogNorm(vmin=vmin,vmax=vmax)
        vmin=vmax=None
    else:
        norm = None
    im = ax.pcolormesh(-np.radians(Lon), np.radians(Lat), arr,  shading='nearest',
        norm=norm, cmap=cmap,  vmin=vmin, vmax=vmax, alpha=alpha)

    if tick_labels:
        ff = lambda d: d if d>=0 else d+360
        ax.set_xticklabels([f'${ff(d):d}^\degree$' for d in np.linspace(150,-150, 11).astype(int)])
    else:
        ax.set(xticklabels=[], yticklabels=[], visible=True)

    if colorbar:
        ticklabels = cb_kw.pop('ticklabels', None)
        cb_kw.update(label=unit,)
        cb = plt.colorbar(im, ax=ax, **cb_kw)
        if ticklabels is not None:
            cb.ax.set_yticklabels(ticklabels)
    if grid_color: 
        ax.grid(color=grid_color)
        ax.set_axisbelow(False) # klugy thing to show grid on top of image
    if label:
        ax.text( 0., 0.97, label, transform=ax.transAxes)
    if title:
        plt.suptitle(title, fontsize=12)
    return AitoffFigure(fig)




class SquareWCS(WCS):
    """
    Create and use a WCS object

    - center : a SkyCoord that will be the center
    - size   : width and height of the display (deg)
    - pixsize [0.1] : pixel size
    - frame [None] : The frame is taken from the center SkyCoord, unless specified here --  only accept "galactic" or "fk5"
    - proj ["ZEA"] : projection to use
    
    To get the WCS properties from the generated Axes object (actually WCSAxesSubplot): ax.wcs.wcs.crval for (l,b)
    """

    def __init__(self, center, size, pixsize=0.1, frame=None, proj='ZEA', unit=''):
        """

        """
        if type(center)==str:
            center = SkyCoord.from_name(center)
        assert isinstance(center, SkyCoord), 'Expect SkyCoord'

        frame = frame or center.frame.name
        if frame=='galactic':
            lon, lat = center.galactic.l.deg, center.galactic.b.deg
            lon_name,lat_name = 'GLON','GLAT'
            self.axis_labels='$l$', '$b$'
        elif frame=='fk5':
            lon,lat = center.fk5.ra.deg, center.fk5.dec.deg
            lon_name, lat_name = 'RA--', 'DEC-'
            self.axis_labels = 'RA', 'Dec'
        else:
            raise Exception(f'Expect frame to be "galactic" or "fk5", not {frame}')

        nx=ny=naxis = int(size/pixsize) | 1 # make odd so central pixel has source in middle
        self.center = center
        self.frame=frame
        self.unit=unit
        self.galactic = frame=='galactic'
        super().__init__(
                         dict(
            NAXIS1=nx, CTYPE1=f'{lon_name}-{proj}', CUNIT1='deg', CRPIX1=nx//2+1, CRVAL1=lon, CDELT1=-pixsize,
            NAXIS2=ny, CTYPE2=f'{lat_name}-{proj}', CUNIT2='deg', CRPIX2=ny//2+1, CRVAL2=lat, CDELT2=pixsize, )
              )

    def _make_grid(self):
        # get coordinates of every pixel`
        nx, ny = self.array_shape
        pixlists = list(range(1,nx+1)),list(range(1,ny+1))
        cgrid = self.pixel_to_world(*np.meshgrid(*pixlists) )
        if not self.galactic:
            cgrid = cgrid.galactic
        lon, lat = (cgrid.l.deg, cgrid.b.deg)
        return lon, lat

    def plot_map(self, hmap, fig=None, axpos=111, figsize=(8,8),
             log=False, cmap='jet', colorbar=True, 
             unit='', vmin=None, vmax=None, cb_kw={},
             title=None, **kwargs):
        """
        Plot a map
        
        - hmap -- a HEALPix map
        - fig  [None] -- a Figure
        - axpos [111] 


        Return a WCSAxes object
        """

        import healpy as hp
        from matplotlib import colors

        wcs = self
        grid = self._make_grid();
        nside = hp.get_nside(hmap)

        ipix = hp.ang2pix(nside, *grid, lonlat=True)

        fig = fig or plt.figure(figsize=figsize)
        if type(axpos)==tuple:
            ax = fig.add_subplot(*axpos, projection=self)
        else:
            ax = fig.add_subplot(axpos, projection=self)
        ax = fig.axes[-1] ####  KLUGE
        if log:
            norm = colors.LogNorm(vmin=vmin,vmax=vmax)
            vmin=vmax=None
        else:
            norm = None
        ipix = hp.ang2pix(nside, *grid, lonlat=True)
        im = ax.imshow(hmap[ipix], cmap=cmap, origin='lower', 
                       norm=norm,  vmin=vmin, vmax=vmax);

        nx, ny = wcs.array_shape
        ax.set(xlabel=self.axis_labels[0], xlim=(-0.5, nx-0.5),
               ylabel=self.axis_labels[1], ylim=(-0.5, ny-0.5),
               title= title)
        ax.grid()
        if colorbar:
            ticklabels = cb_kw.pop('ticklabels', None)
            cb_kw.update(label=self.unit,)
            cb = plt.colorbar(im, ax=ax,  **cb_kw)
            if ticklabels is not None:
                cb.ax.set_yticklabels(ticklabels)
        # set the default transform for world
        ax.transAxes = ax.get_transform(self.frame)
            # annotator(ax, self.frame)
        return ax
    
class ZEAaxis:
    """
    Wraps WCSaxis

    """
    def __init__(self, ax):
        self.ax= ax

    def _to_pixel(self, *args):
        return _to_pixels(*args, wcs=self.ax.wcs)
        # return self.ax.wcs.world_to_pixel(sc) + tuple(rest) 
    
    def plot(self, *args, **kwargs):
        return self.ax.plot(*self._to_pixel(*args), **kwargs)
    
    def scatter(self, *args, **kwargs):
        return self.ax.scatter(*self._to_pixel(*args), **kwargs)
    def text(self, *args, **kwargs):
        return self.ax.text(*self._to_pixel(*args),  **kwargs)

    def __getattr__(self, name):
        # pass everything else to self.ax
        return self.ax.__getattribute__( name)
    

class HPmap(object):
    """
    Manage HEALPix array
    """
    def __init__(self,
            hpmap:'HEALPix array',
            name='',
            cblabel=None,
            unit='',
            sigma:'smooth parameter'=None,
            nest=False):
        """create from a HEALPix array
        """
        self.name = name
        self.cblabel = cblabel if cblabel is not None else unit
        self.unit = unit
        self.nside = healpy.get_nside(hpmap)
        # reorder as defaut RING if nest is set
        self.map = hpmap if not nest else healpy.reorder(hpmap, n2r=True)

        if sigma is not None:
            self.smooth(sigma)

    def __str__(self):
        return f'<{self.__class__.__name__}>, name "{self.name}" nside {self.nside} unit "{self.unit}"'
    def __repr__(self): return str(self)

    def __call__(self, sc:'SkyCoord') -> 'value[s]':
        """
        Return value of corresponding pixel(s)
        """
        sp = sc.spherical # avoid dependence on specific frame
        skyindex = healpy.ang2pix(self.nside, sp.lon.deg, sp.lat.deg, lonlat=True)
        return self.map[skyindex]

    def get_all_neighbors(self, pix):
        return healpy.get_all_neighbours(self.nside, pix)

    def smooth(self, sigma):
            self.map = healpy.smoothing(self.map, np.radians(sigma))


    def ait_plot(self, **kwargs):
        """
        Invoke the function ait_plot to draw a representation

        """
        kw= dict(label=self.name, cblabel=self.unit,)
        kw.update(**kwargs)
        return ait_plot(self, **kw)

        
    def zea_plot(self, *args, size=10, pixsize=0.1, 
                fig=None, figsize=(8,8), **kwargs):
        """
        Create a SquareWCS object that defines its WCS as a ZEA square centered on `skycoord`, which can be a source name.
        Return the ZEAaxes object
        """
        sc, rest = _process_args(*args)
        
        # if not isinstance(skycoord, SkyCoord):
        #     skycoord = SkyCoord.from_name(skycoord).galactic
        swcs = SquareWCS(sc, size, pixsize)
        return ZEAaxis(
            swcs.plot_map(self.map, unit=self.unit, fig=fig, figsize=figsize, **kwargs)
        )


    def convolve(self, beam_window=None, sigma=0):
        """Convolve the map with a "beam", or PSF

        - beam_window: Legendre coefficients or None
        - sigma  Gaussian sigma in degrees

        return a HEALPix array
        """
        import healpy

        return healpy.alm2map(
            healpy.smoothalm( healpy.map2alm(self.map),
                              sigma=np.radians(sigma),
                              beam_window=beam_window
                            ),
            nside=self.nside
            )

    def to_FITS(self,  filename=''):
        """return a HDUlist object with one skymap column

        - filename [''] write to the file if set
        """

        column = fits.Column(name=self.name, format='E', array=self.map, unit=self.unit)

        nside = self.nside
        cards = [fits.Card(*pars) for pars in [
            ('FIRSTPIX', 0,             'First pixel (0 based)'),
            ('LASTPIX',  12*nside**2, 'Last pixel (0 based)'),
            ('MAPTYPE',  'Fullsky' , ''  ),
            ('PIXTYPE',  'HEALPIX',      'Pixel algorithm',),
            ('ORDERING', 'RING',         'Ordering scheme'),
            ('NSIDE' ,   nside,    'Resolution Parameter'),
            ('ORDER',    int(np.log2(nside)),   'redundant'),
            ('INDXSCHM', 'IMPLICIT' ,''),
            ('OBJECT' ,  'FULLSKY', ''),
            ('COORDSYS', 'GALACTIC', ''),
        ]]
        table = fits.BinTableHDU.from_columns([column],header=fits.Header(cards))
        table.name = 'SKYMAP'

        hdus = fits.HDUList([
                    fits.PrimaryHDU(header=None),
                    table,
                ])
        if filename:
            hdus.writeto(filename, overwrite=True)
            print(f'Wrote FITS file {filename}')
        return hdus
    
    @classmethod
    def from_FITS(cls, filename, field=0, *pars, **kwargs):
        with fits.open(filename) as hdus:
            header, data = hdus[1].header, hdus[1].data
        kw = dict(unit=header.get('TUNIT1', ''), name=header['TTYPE1'])
        kw.update(**kwargs)
        return cls(data.field(field), *pars, **kw)