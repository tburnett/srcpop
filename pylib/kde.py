from scipy import stats
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


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

class FeatureSpace(dict):
    """ class FeatureSpace:
        Manage binning properties and evaluation in the KDE features space
    """

    default_limits =  dict( sqrt_d=(0.,np.sqrt(2)),
                    log_epeak=np.log10([0.15, 4]),
                    diffuse=(-1, 2), )
    
    def __init__(self, limits=None,  N=25):
        # set up axes of the grid and define it
        if limits is None: limits = self.default_limits
        self['names'] = names = list(limits.keys())#names
        self['limits'] = limits 
        self['bins'] = bins =dict( (var, np.linspace(*self['limits'][var],num=N+1 ).round(3)) for var in names)
        cvals = lambda b: (b[1:]+b[:-1])/2
        self['centers'] = dict((nm, cvals(v).round(3)) for nm,v in bins.items()) 
        delta = lambda name: self['limits'][name][1]-self['limits'][name][0]
        self['size'] = dict((name, delta(name)) for name in names)
        self['N']=N
        self['volume']=  np.prod( list(self['size'].values())) 

        self.__dict__.update(self)

    def evaluate_kdes(self, kdes):
        """ Use a set of KDE functions to populate a grid
        """
        self.grid = pd.DataFrame(dict( (name, mg.flatten()) for name,mg in
                       zip(self.names, np.meshgrid(*self.centers.values()))))

        # Evaluate KDE on the grid, measure integrals, renormalize"
        self.grid_kde = pd.DataFrame()
        for comp in  kdes.keys():
            self.grid[comp+'_kde'] = kdes[comp](self.grid)
            self.grid_kde = kdes[comp](self.grid)

        # calculate normalization, save factors
        self.norms = self.grid_kde.sum() * self.volume/self.N**3
        self.grid_kde /= self.norms
        # self.grid.iloc[:, -4:] /= self.norms
        assert np.any(~pd.isna(self.grid_kde)), 'Found a Nan on the grid!'


    def __repr__(self):
        return f'class {self.__class__.__name__}:\n'+'\n'.join( [f'{key}: {value}' for key, value in self.items()])

    def generate_kde(self, df, bw_method=None):
        """ Return a KDE using the DataFrame df
        """
        return Gaussian_kde(df, self.names, bw_method=bw_method)

    # ------------These assume evaluate_kdes has been called to add columns -------------
    
    def projection(self, varname):
        """ Return DataFrame with index varname's value and columns for class components + unid,
        basically an integral over the other variables
        """        
        t = self.grid.copy().groupby(varname).sum()*(self.volume/self.size[varname])/self.N**2
        return t.iloc[:, -4:]
        
    def projection_dict(self, var_name, cls_name):
        td = self.projection(var_name)
        return dict(x=td.index, y=td[cls_name+'_kde'])
        
    def  normalize(self, var, norm:dict):
        """ return df with normalized counts
        """
        td = self.projection(var)
        norm = pd.Series(norm)
        coldata = td.loc[:, [n+'_kde' for n in norm.index]].to_numpy() * norm.values *self.size[var]/self.N
        df = pd.DataFrame(coldata ,  index=td.index, columns=norm.index)
        df['sum'] = df.sum(axis=1)
        return df
    
    # ---------------------- Following make plots  --------------------

    def projection_check(self, df_dict, palette):
        """Density plots for each of the features comparing the training data with projections of the KDE fits.

        """
        fig, axx = plt.subplots(ncols=3,nrows=3, figsize=(12,7), sharex='col',
                            gridspec_kw=dict(hspace=0.1, wspace=0.1))
        
        for (i, (cls_name, df)), color in zip(enumerate(df_dict.items()), palette):
            
            for j, var_name in enumerate( self.names ):
                ax = axx[i,j]
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['left'].set_visible(False)
                sns.histplot(ax=ax,  x=df[var_name], bins=self.bins[var_name], 
                element='step', stat='density', color='0.2', edgecolor='w')
        
                sns.lineplot( ax=ax, **self.projection_dict(var_name, cls_name), 
                            color=color, ls='-', marker='', lw=2)
                ax.set( xlim=self.limits[var_name], ylabel='', yticks=[] )
            axx[i,0].set_ylabel(cls_name+f'\n{len(df)}', rotation='horizontal', ha='center')
        return fig
    
    def component_comparison(self, unid, norm, palette):
        """Histograms of Unid data for the KDE feature variables, compared with an estimate
        of the class contents. Each component was normalized to the total shown in the legend.
        """
        fig, axx =plt.subplots(ncols=3, figsize=(12,4), sharey=True, 
                            gridspec_kw=dict(wspace=0.1))
        
        for var,ax in zip(self.names, axx):
            
            df = self.normalize( var, norm)   
            x= df.index
            for y,color in zip(df.columns, palette+['white']):
                sns.lineplot(df,ax=ax, x=x, y=y,  color=color, 
                            label=f'{norm[y] if y!="sum" else round(np.sum(df.iloc[:,-1]))} {y}', 
                            lw=2 if y=='sum' else 1,     legend=None)
            sns.histplot(unid, bins=self.bins[var], ax=ax, x=var, element='step', color='0.2', edgecolor='w', 
                        label=f'{len(unid)} unid') 
            ax.set(ylabel='counts / bin')
                        
        ax.legend(fontsize=12, bbox_to_anchor=(0.9,1.15), loc='upper left');
        ax.set(yticks=np.arange(0,151,50))
        return fig
    