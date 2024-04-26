# from pylib.ipynb_docgen import show
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import optimize # type: ignore

var_labels = dict(sqrt_d = '$\sqrt{d}$',
              log_epeak = '$\log_{10}(E_p)$',
              diffuse = '$D$')

def data_model(self, norms, unid, fig=None, palette=None):

    if fig is None:
        axd = plt.figure(figsize=(20,6), layout="constrained").subplot_mosaic([self.names], sharey=True)
        # fig, axx = plt.subplots(ncols=3, figsize=(20,6), sharey=True,
        #                        gridspec_kw=dict(wspace=0.1))
    else:
        # axx = fig.subplots(ncols=3, sharey=True, gridspec_kw=dict(wspace=0.1))
        axd = fig.subplot_mosaic([self.names], sharey=True)
    pi = self.projection_info(unid)
    xtick_dict=dict(sqrt_d=np.arange(0,1.5,0.5), 
                    log_epeak=np.arange(-1, 0.6, 0.5), 
                    diffuse=np.arange(-1,2.1,1))
    # for var_name, ax in zip(self.names, axx.flat): ]
    for var_name in self.names:
        ax = axd[var_name]    
        df =pi[var_name]
        x = df.x
        ax.errorbar(x=x, y=df.unid, xerr=self.size[var_name]/2/self.N, 
                    yerr=np.sqrt(df.unid), fmt='.', label='unid')
        ax.set(xlabel=var_labels[var_name], xticks=xtick_dict[var_name],
               ylim=(0,None))
        var_norm = self.size[var_name]/self.N
        total = np.zeros(self.N)    
        
        for cls_name, color in zip(self.class_names, palette):
            y = norms[cls_name]*var_norm* df[cls_name]
            total+=y
            ax.plot(x, y,   color=color, label=cls_name)
            
        ax.plot(x,total, color='w' if self.dark_mode else 'k', label='sum', lw=4)
        if var_name=='log_epeak':
            ax.legend(loc='upper right', fontsize='small',frameon=False,
                      ncols=2,  bbox_to_anchor=(1, 1.3))
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)

    axd['sqrt_d'].set( ylabel='Counts / bin',yticks=np.arange(0,151,50));
    return fig

def get_deltas(self, norms, unid):
    pi = self.projection_info(unid)
    deltas = dict()
    for var_name in self.names: 
    
        df = pi[var_name]
        x = df.x
        unid = df.unid
    
        var_norm = self.size[var_name]/self.N
        total = np.zeros(self.N)    
        
        for cls_name in self.class_names:
            y = norms[cls_name]*var_norm* df[cls_name]
            total+=y
        deltas[var_name] = total-unid    
    return deltas

def show_diffs(self, norms, unid, fig=None, palette=None):
    """
    """
    if fig is None:
        fig, ax = plt.subplots(figsize=(8,6))
    else:
        ax = fig.subplot_mosaic('..AAA.')['A']
    deltas = get_deltas(self, norms, unid)
    for var_name, m, color in zip(self.names, 'oDv', palette):
        ax.plot(-deltas[var_name], m, ls='--' , ms=8, 
                label=var_labels[var_name], color=color)
    ax.legend(loc='upper left', bbox_to_anchor=(1., 1),fontsize='small')
    ax.axhline(0, color='0.5')
    ax.set(xlabel='bin number', ylabel='Unid data - model');
    return fig

def slice_loglike(self, x, norms, cls_name, *, var_name, idx,):  
    """Evaluate log likelihood

    norms: dict with input norms
    x : dependent variable, number of counts for cls_name
    var_name 
    """
    n = norms.copy()
    n[cls_name] = x
    pi = self.projection_info(unid)
    df = pi[var_name]
    N = df.unid[idx].sum() 
    
    var_norm = self.size[var_name]/self.N
    
    model = np.zeros(self.N)        
    for cls_name in self.class_names:
        y = n[cls_name]*var_norm* df[cls_name]
        model+=y
    mu = model[idx].sum()#.round(3) 
    return N * np.log(mu) - mu

def fit_plots(self, norms, unid, palette):
    fig = plt.figure(layout='constrained', figsize=(12, 8))
    subfig1, subfig2= fig.subfigures(nrows=2,  hspace=0.1, height_ratios=(3,2))
    data_model(self,  norms, unid, subfig1, palette)
    show_diffs(self,  norms, unid, subfig2, palette)
    fig.text(0.2, 0.4, 'Model contents\n'+'\n'.join(str(pd.Series(norms)).split('\n')[:-1]),
             ha='right',va='top')
    return fig;


class Fitter:
    
    def __init__(self, fs, unid,  fit_slice): 
        """fs: FeatureSpace object
        """
        self.fs = fs
        self.unid = unid
        self.fit_slice=fit_slice
        self.proj_info = fs.projection_info(unid)

    def fit_function(self,norms):
        """Return a dict with the three slice likelihood functions
        """
        
        def slice_loglike( x, cls_name, *, var_name, idx,):  
            """Evaluate log likelihood
        
            norms: dict with input norms
            x : dependent variable, number of counts for cls_name
            var_name 
            """
            n = norms.copy()
            n[cls_name] = x
            fs = self.fs
            df = self.proj_info[var_name]

            N = df.unid[idx].sum() 
            
            var_norm = fs.size[var_name]/fs.N
            
            model = np.zeros(fs.N)        
            for cls_name in fs.class_names:
                y = n[cls_name]*var_norm* df[cls_name]
                model+=y
            mu = model[idx].sum()#.round(3) 
            return N * np.log(mu) - mu
 
        return dict(
            blazar= lambda x: slice_loglike(x, 'blazar', **self.fit_slice['blazar']),
            psr   = lambda x: slice_loglike(x, 'psr',    **self.fit_slice['psr']),
            msp  =  lambda x: slice_loglike(x, 'msp',    **self.fit_slice['msp']),
            )

    def new_fit(self, norms):
        import copy
        fit_info = copy.deepcopy(self.fit_slice)

        f = self.fit_function(norms)
        for cls_name in self.fit_slice.keys(): #, ax in axd.items():
            
            opt = optimize.minimize(
                lambda x: -np.array([f[cls_name](x)]), 
                x0=norms[cls_name], 
                )            
            fit_info[cls_name]['fit'] = round(opt.x[0])
            fit_info[cls_name]['sigma'] = round(np.sqrt(opt.hess_inv).diagonal()[0])
            
        fits = pd.DataFrame(fit_info).T
        def get_range(r):
            t = self.fs.bins[r.var_name][r.idx]
            return (t[0], t[-1])
        fits['range']= fits.apply(get_range, axis=1) 
        fits.index.name='class name'
        return fits       
        
       
    def fit_3d(self, norms):
        """Return 3-d optimization object
        """

        fitter = self
        class F3:
            def __init__(self, norms):
                """
                * fitter: the Fitter object
                implements fit_function
                * norms: a dict-like object with class names and initial values for fit
                """
                self.classes = norms.keys()
                self.norms = dict(norms) # convert Series if necessary
                
            def __call__(self, x):
                """x : 3-d array, order as in norms.keys()
                return negative of log likelihood
                """
                # make a new dict
                xnorm = dict( (n,v) for n,v in zip(self.classes, x ))
                # get dict of functions set for x
                f = fitter.fit_function(xnorm)
                return  -np.sum([f[n]( xnorm[n] ) for n in self.classes])  
                    
            def maximize(self,):
                """Maximize the log likelihood
                set `opt` in outer class with miminize result
                return DF with fit values, diagonal uncertainties
                """
                x0 = list(self.norms.values())
                fitter.opt = opt =   optimize.minimize(self, x0)
                fitval = pd.Series(dict( (k,v) for k,v in zip(self.classes, opt.x.round(1))),name='fit')
                cov = opt.hess_inv
                fitunc = pd.Series(dict( (k,v) for k,v in
                    zip(self.classes,np.sqrt( cov.diagonal()).round(1))),name='unc')
                return pd.DataFrame([fitval, fitunc])
            
        return F3( norms)
    
