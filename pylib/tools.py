from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def set_theme(argv):

    sns.set_theme('notebook' if 'talk' not in argv else 'talk', font_scale=1.25) 
    if 'dark' in argv:
        plt.style.use('dark_background') #s
        plt.rcParams['grid.color']='0.5'
        dark_mode=True
    else:
        dark_mode=False
    return dark_mode



def ternary_plot(df, columns=None, ax=None):
    import ternary
    if columns is None: 
        columns=df.columns
    assert len(columns==3)
    fig, ax = plt.subplots(figsize=(8,8))

    tax = ternary.TernaryAxesSubplot(ax=ax,)
    
    tax.right_corner_label(columns[0], fontsize=16)
    tax.top_corner_label(columns[1], fontsize=16)
    tax.left_corner_label(columns[2], fontsize=16)
    tax.scatter(df.iloc[:,0:3].to_numpy(), marker='o',
                s=10,c=None, vmin=None, vmax=None)#'cyan');
    ax.grid(False); ax.axis('off')
    tax.clear_matplotlib_ticks()
    tax.set_background_color('0.3')
    tax.boundary()
    return fig


@dataclass
class FigNum:
    n : float = 0
    dn : float= 1
    @property
    def current(self): return self.n if self.dn==1 else f'{self.n:.1f}'
    @property
    def next(self):
        self.n += self.dn
        return self.current
    def __repr__(self):
        return self.current
    
def show_date():
    from pylib.ipynb_docgen import show
    import datetime
    date=str(datetime.datetime.now())[:16]
    show(f"""<h5 style="text-align:right; margin-right:15px"> {date}</h5>""")

def update_legend(ax, data, hue, **kwargs):
    """ seaborn companion to insert counts in legend,
    perhaps change location or fontsize
    """
    gs = data.groupby(hue).size()
    leg = ax.get_legend()
    fontsize = kwargs.pop('fontsize', None)
    leg.set(**kwargs)

    for tobj in leg.get_texts():
        text = tobj.get_text()
        if fontsize is not None: tobj.set(fontsize=fontsize)
        if text in gs.index:
            tobj.set_text(f'({gs[text]}) {text}', )


def curly(x,y, scale, ax=None, color='k'):
    import matplotlib.transforms as mtrans
    from matplotlib.text import TextPath
    from matplotlib.patches import PathPatch
    
    if not ax: ax=plt.gca()
    tp = TextPath((0, 0), "}", size=1)
    trans = mtrans.Affine2D().scale(1, scale) + \
        mtrans.Affine2D().translate(x,y) + ax.transData
    pp = PathPatch(tp, lw=0, fc=color, transform=trans)
    ax.add_artist(pp)

def curly_demo():
    X = [0,1,2,3,4]
    Y = [1,1,2,2,3]
    S = [1,2,3,4,1]
    fig, ax = plt.subplots()

    for x,y,s in zip(X,Y,S):
        curly(x,y,s, ax=ax)

    ax.axis([0,5,0,7])
    plt.show()
