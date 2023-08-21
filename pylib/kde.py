from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


class Gaussian_kde(stats.gaussian_kde):
    """Adapt the stats version to use a DataFrame as input
    """
    def __init__(self, data: pd.DataFrame, 
                 cols=None, 
                 **kwargs):
        df = data if cols is None else data.loc[:,cols] 
        self.columns = df.columns.values
        self.limits = df.describe().loc[('min','max'),:]
        super().__init__(df.to_numpy().T, **kwargs)

    def __repr__(self):
        return f'Gaussian_kde with columns {list(self.columns)}, {self.n} data points'

    def __call__(self, df:pd.DataFrame):
        """Override the __call__ to apply to a DataFrame, which must have columns with same name as
        used to generate this object.
        """
        assert np.all(np.isin( self.columns, df.columns)), \
            f'The DataFrame does not have the columns {list(self.columns)}'
        return  self.evaluate(df.loc[:,self.columns].to_numpy().T) 
        
    @classmethod
    def example(cls, n=2000):
        """ The 2-D example from the reference"""
        def meas(n=2000):
            m1 = np.random.normal(size=n)
            m2 = np.random.normal(scale=1/2, size=n)
            return pd.DataFrame.from_dict(dict(x=m1-m2, y=m1+m2, ))
        return cls(meas(n), 'x y'.split())

def apply_kde(data, hue, vars, hue_value):
    assert np.all(np.isin(vars, data.columns)), f'{vars} not all in data'    
    # create KDE from the data for which data[hue]==hue_value
    kde = Gaussian_kde(data[data[hue]==hue_value], vars)    
    # and apply it to all, normalizing to maximum
    cdf = kde(data)
    return cdf/np.max(cdf) 

class KDE:
    """ 
    Implements a 2-D KDE, like example    
    [Reference](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.gaussian_kde.html)
    Select columns from dataframe, create kernel
    """
    def __init__(self, data, x='x', y='y', 
                reflectx=False):
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
        _, ax = plt.subplots() if ax is None else (ax.figure, ax)
        self.plot_map(cmap=cmap, ax=ax)
        self.plot_points(ax=ax)

    @classmethod
    def test(cls, n=2000):
        """Should produce exactly the plot in the documentation
          for `scipy.stats.gaussian_kde`"""
        def meas(n):
            m1 = np.random.normal(size=n)
            m2 = np.random.normal(scale=1/2, size=n)
            return pd.DataFrame.from_dict(dict(x=m1-m2, y=m1+m2, ))
        cls( meas(n) ).plot(cmap='gist_earth_r') 