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
{stats.gaussian_kde.__dict__}
"""