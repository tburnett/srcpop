from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

class KDE:
    """ Select columns from dataframe, create kernel"""
    def __init__(self, data, x='x', y='y', 
                reflectx=True):
        self.reflectx=reflectx
        self.dfxy = data.loc[:,(x,y)]
        x, y = self.dfxy.to_numpy().T
        self.kernel = stats.gaussian_kde((x,y))

    def __call__(self, p):
        """ Evaluate kernel at points"""
        xy = np.atleast_1d(p)
        ret =self.kernel(xy)
        if self.reflectx: #reflect about x=0
            reflect =np.array((-1,1))
            ret += self.kernel((xy.T*reflect).T)
        return ret
        
    def _grid(self, bins=100):
        jbins =complex(0,bins) # flag to use max        
        a,c  = self.dfxy.min().values
        b,d = self.dfxy.max().values
        extent = (a,b,c,d)
        X, Y = np.mgrid[a:b:jbins, c:d:jbins]
        grid_positions = np.vstack([X.flatten(), Y.flatten()])
        return np.reshape(self(grid_positions), X.shape), extent
        
    def plot_map(self, cmap='gist_earth_r', ax=None):
        Z, extent = self._grid()
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        ax.imshow(np.rot90(Z),  cmap=cmap,  extent=extent)

    def plot_points(self,ax=None):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        col = self.dfxy.columns
        ax.set(xlabel=col[0],ylabel=col[1])
        x,y = self.dfxy.iloc[:,:2].to_numpy().T
        ax.plot(x,y, 'k.', markersize=2)
       
    def plot(self, cmap='gist_earth_r', ax=None):
        fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
        self.plot_map(cmap=cmap, ax=ax)
        self.plot_points(ax=ax)

    @classmethod
    def test(cls, n=2000):
        def meas(n):
            m1 = np.random.normal(size=n)
            m2 = np.random.normal(scale=1/2, size=n)
            return pd.DataFrame.from_dict(dict(x=m1-m2, y=m1+m2, ))
        self = cls(meas(n), reflectx=False)
        self.plot() #cmap='viridis')