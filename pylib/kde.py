from scipy import stats
import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd
# import seaborn as sns
# from fermi_sources import update_legend
# from ipynb_docgen import show


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

# def kde_analysis(df, hue_kw, size_kw):
#     """Scatter plot of the normalized msp and psr KDE estimates for unid-pulsar, msp, and psr categories.
#     The triangular and square regions select sources most likely to be neither.

#     """

#     def kde_check(cols = 'log_epeak log_fpeak d diffuse'.split()):
#         tdf = df.copy()
#         for t in hue_kw['hue_order'][1:]:
#             gde = Gaussian_kde( df[df[hue_kw['hue']]==t],  cols)
#             arg ='pdf_'+t
#             u = gde(df)
#             tdf[arg] = u/np.max(u)
        
#         fig, ax = plt.subplots(figsize=(8,8))
#         # size_kw = dict(size='log TS', sizes=(20,100) )
#         # hue_kw = dict(**hue_kw, #hue='source type', hue_order='young MSP UNID-PSR'.split(),
#         #              palette='yellow magenta cyan'.split(), edgecolor=None)
#         y,x = tdf.iloc[:, -2:].values.T #tdf.pdf_young, tdf.pdf_MSP
#         sns.scatterplot(tdf, ax=ax,  x=x, y=y,  **hue_kw, s=10,

#          )#, **size_kw);
#         # update_legend(ax, df, hue=hue_kw )
#         ax.set(xlabel='Normalized young probability', ylabel='Normalized MSP probability')
#         show(f"""## KDE analysis 
#             vars: {cols}""")
#         return ax, tdf

#     ax, df = kde_check('log_epeak log_fpeak d diffuse'.split())
    
#     class Triangle:
#         def __init__(self, ax, x=(0.06,0.55), y=(0.04, 0.54)):
#             a,b = x
#             c,d = y
#             alpha = (c-d)/(b-a)
#             beta = c-alpha*b
#             # inside is in the triangle or rectangle at origin
#             self.inside = lambda x,y: ((x>a) & (y>c)  & (y < alpha*x+beta ))  | ((x<a) & (y<c))
#             ax.plot([a,b,a,a], [c,c,d, c], '--', color='1.', lw=2)
    
#         def __call__(self, df, x, y):
#             return self.inside(df[x],df[y]) 

#     class Square(Triangle):
#         def __init__(self, ax, a=0.1, b=0.08):
#             z=-0.025
#             self.inside =  lambda x,y: (x<a) & (y<b)
#             ax.plot([z, a, a, z, z] , [z, z, b,b,z], ':', color='1.', lw=4)
    
#     t = Triangle(ax)
#     s = Square(ax)
#     return ax.figure
#     return df,t,s

    # pars =df[df[hue_kw['hue']]==hue_kw['hue_order']] #'UNID-PSR'],  'pdf_young', 'pdf_MSP' 
    # q = t(*pars) | s(*pars)
    # show(f'Selected {sum(q)} of the {hue_kw["hue_order"][0]} sources inside the square and triangular regions')

# def apply_kde(data, hue, vars, hue_value):
#     assert np.all(np.isin(vars, data.columns)), f'{vars} not all in data'    
#     # create KDE from the data for which data[hue]==hue_value
#     kde = Gaussian_kde(data[data[hue]==hue_value], vars)    
#     # and apply it to all, normalizing to maximum
#     cdf = kde(data)
#     return cdf/np.max(cdf) 



#============= not used? ================
# class KDE:
#     """ 
#     Implements a 2-D KDE, like example    
#     [Reference](https://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.gaussian_kde.html)
#     Select columns from dataframe, create kernel
#     """
#     def __init__(self, data, x='x', y='y', 
#                 reflectx=False):
#         self.reflectx=reflectx
#         self.dfxy = data.loc[:,(x,y)]
#         x, y = self.dfxy.to_numpy().T
#         self.kernel = stats.gaussian_kde((x,y))

#     def __call__(self, p):
#         """ Evaluate kernel at points"""
#         xy = np.atleast_1d(p)
#         ret =self.kernel(xy)
#         if self.reflectx: #reflect about x=0
#             reflect =np.array((-1,1))
#             ret += self.kernel((xy.T*reflect).T)
#         return ret
        
#     def _grid(self, bins=100):
#         jbins =complex(0,bins) # flag to use max        
#         a,c  = self.dfxy.min().values
#         b,d = self.dfxy.max().values
#         extent = (a,b,c,d)
#         X, Y = np.mgrid[a:b:jbins, c:d:jbins]
#         grid_positions = np.vstack([X.flatten(), Y.flatten()])
#         return np.reshape(self(grid_positions), X.shape), extent
        
#     def plot_map(self, cmap='gist_earth_r', ax=None):
#         Z, extent = self._grid()
#         fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
#         ax.imshow(np.rot90(Z),  cmap=cmap,  extent=extent)

#     def plot_points(self,ax=None):
#         fig, ax = plt.subplots() if ax is None else (ax.figure, ax)
#         col = self.dfxy.columns
#         ax.set(xlabel=col[0],ylabel=col[1])
#         x,y = self.dfxy.iloc[:,:2].to_numpy().T
#         ax.plot(x,y, 'k.', markersize=2)
       
#     def plot(self, cmap='gist_earth_r', ax=None):
#         _, ax = plt.subplots() if ax is None else (ax.figure, ax)
#         self.plot_map(cmap=cmap, ax=ax)
#         self.plot_points(ax=ax)

#     @classmethod
#     def test(cls, n=2000):
#         """Should produce exactly the plot in the documentation
#           for `scipy.stats.gaussian_kde`"""
#         def meas(n):
#             m1 = np.random.normal(size=n)
#             m2 = np.random.normal(scale=1/2, size=n)
#             return pd.DataFrame.from_dict(dict(x=m1-m2, y=m1+m2, ))
#         cls( meas(n) ).plot(cmap='gist_earth_r') 