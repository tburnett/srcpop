from scipy import stats
import numpy as np
import pandas as pd


class Gaussian_kde(stats.gaussian_kde):
    """
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
    
    def pdf(self, df):
        """ For convenience"""
        return self.evaluate(df.loc[:,self.columns].T) 
    
    @property
    def extent(self):
        """For imshow"""
        return self.limits.values.T.ravel() 
    
        
    @classmethod
    def example(cls, n=2000):
        """ The 2-D example from the reference"""
        def meas(n=2000):
            m1 = np.random.normal(size=n)
            m2 = np.random.normal(scale=1/2, size=n)
            return pd.DataFrame.from_dict(dict(x=m1-m2, y=m1+m2, ))
        return cls(meas(n), 'x y'.split())

Gaussian_kde.__doc__ = f"""\
Adapt stats.gaussian_kde to use a DataFrame as input

The example 
```
self = Gaussian_kde.example()
import matplotlib.pyplot as plt
X,Y = np.mgrid[-4:4:100j, -4:4:100j]
positions = pd.DataFrame.from_dict(dict(x=X.ravel(),y=Y.ravel()))
Z = np.reshape(self(positions).T, X.shape)
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(np.rot90(Z), cmap=plt.cm.gist_earth_r, extent=self.extent, )
plt.show()
```

"""

class ComponentProb(Gaussian_kde): 
    """
    Implement KDE with bounds
    """

    def __init__(self, data:pd.DataFrame, 
                 features:[], 
                 bounds={},  
                ):
        super().__init__(data, features)
        self.bounds=bounds
        self.features=features
    
    def __call__(self, data:pd.DataFrame):

        t = data.loc[:,self.columns].to_numpy().T
        ret = self.evaluate(t)
        # add reflections
        for key, (lo, hi) in self.bounds.items():
            idx = self.features.index(key)
            if lo is not None:   # print(self.features[idx], '>', lo )
                t2 = t.copy()
                t2[idx, :] = t[idx,:] - 2*lo
                ret += self.evaluate(t2)

            if hi is not None:   # print(self.features[idx], '<', hi)
                t2 = t.copy()
                t2[idx, :] = 2*hi-t[idx,:]
                ret += self.evaluate(t2)
        return ret
