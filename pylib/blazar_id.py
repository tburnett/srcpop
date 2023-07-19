from pylib.fermi_sources import *
from pathlib import Path

def get_source_data(filename='files/classification.pkl'):
    f = Path(filename)
    if not f.is_file():
        FermiSources().train_predict(save_to=filename)
        assert f.is_file(), f'Failed to create {filename}'
    df =  pd.read_pickle(filename)
    show(f"""# Blazar studies: 
        Read in "{filename}" with {len(df)} source entries.
        """)
    return df
    
class BlazarEff:
    def __init__(self, b=0.31, c=5, integral=False):
        self.__dict__.update(b=b, c=c, int=integral)

    def __call__(self,x):
        b,c = self.b, self.c
        return  b*(1-c* np.exp(-c*x))  - b*(1-c) if not self.int else \
            x+self.b*(x-1+np.exp(-self.c*x)) 
    
class Blazars:
    def __init__(self):
        self.df =  get_source_data()
        show_date()
            
    def observation(self,  hue='association'):
        show(f"""## Distributions of associated blazars""")
        fig, ax =plt.subplots(figsize=(8,4))
        ax.set(xlabel=r'$|\sin(b)|$', xlim=(0,1))
        sns.histplot(self.df.set_index('association').loc['bcu bll fsrq'.split()], 
                     hue=hue, multiple='stack', ax=ax, 
                      x='abs_sin_b', bins=np.linspace(0.,1,26), element='step')
        ax.legend() # leg = ax.get_legend(); leg(loc='upper left')
        be = BlazarEff(c=3)
        x = np.linspace(0,1,)
        norm =  be(1)/180
        ax.plot(x, be(x)/norm);
        show(fig, caption=f"""
             Stacked histogram of absolute blazar galactic latitudes, according to the classification. 
             """)
        


    
# self = Blazars()
