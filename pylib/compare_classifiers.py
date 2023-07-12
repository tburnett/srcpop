import fermi_sources
from fermi_sources import *
fs_data = FermiSources('files/fermi_sources_v2.csv')
unid =  FermiSources('files/fermi_sources_v2.csv')
specs1 = MLspec(features=('log_epeak','pindex','curvature','log_e0'), target='association')
specs2 = MLspec(features=('log_epeak','pindex','curvature','log_e0'), target='association', target_names=('unid'))

from pathlib import Path
import os, sys, glob
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.inspection import DecisionBoundaryDisplay, permutation_importance



class Compare_Classifiers:
    names: list = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        #"Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]
    
    classifiers: list = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        #GaussianProcessClassifier(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100, max_features=2),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]
    
    def __init__(self, tr_data, test_data, path, names=names, classifiers=classifiers):
        
        self.classifiers = classifiers
        
        self.names = names
        
        self.X,self.y = tr_data
        
        self.X_te = test_data[0]
        
        self.datasets = [
            (self.X.iloc[:,:2],self.y),
            (self.X.iloc[:,::2],self.y),
            (self.X.iloc[:,::3],self.y),
            (self.X.iloc[:,1:3],self.y),
            (self.X.iloc[:,1::2],self.y),
            (self.X.iloc[:,2:4],self.y)
        ]
        
        
        self.testval = [
            (self.X_te.iloc[:,:2],self.y),
            (self.X_te.iloc[:,::2],self.y),
            (self.X_te.iloc[:,::3],self.y),
            (self.X_te.iloc[:,1:3],self.y),
            (self.X_te.iloc[:,1::2],self.y),
            (self.X_te.iloc[:,2:4],self.y)
        ]

        
        self.names = names
        self.classifiers = classifiers
        
        self.p = Path(path)
        self.p.mkdir(exist_ok=True)
        
        self.scores = np.empty(len(self.classifiers))
        
        
    def plt_grid(self, figsize=(11,16), trainPoints=False, testPoints=True, score=True):
        
        self.make_plots(3, figsize, "grid", trainPoints, testPoints, score)
        
        
        
    def plt_single(self, figsize=(22,32), trainPoints=False, testPoints=True, score=True):
        
        self.make_plots(3, figsize, "single", trainPoints, testPoints, score)
        
        
    def make_plots(self, col, figsize, plt_type, trainPoints=False, testPoints=True, doscore=True):
        
        figure = plt.figure(figsize=figsize)
        
        k = 0
        
        for name, clf in zip(self.names, self.classifiers):
            
            i = 1
            j = 0
            
            total = 0
            
            if plt_type == "single":
                
                dirpath = os.path.join(self.p, name)
                
                try:
                    os.mkdir(dirpath)
                except FileExistsError:
                    print('Directory {} already exists'.format(dirpath))
                else:
                    print('Directory {} created'.format(dirpath))
                    
                newp = Path(dirpath)
                
            
            for ds_cnt, ds in enumerate(self.datasets):
                
                theX, they = ds
                model = clf
                model.fit(self.X,self.y)

                Xnew = self.testval[j][0]
                ynew = model.predict(self.X_te)

                x_min, x_max = theX.iloc[:, 0].min() - 0.5, theX.iloc[:, 0].max() + 0.5
                y_min, y_max = theX.iloc[:, 1].min() - 0.5, theX.iloc[:, 1].max() + 0.5
                
                they = self.str2num(they)
                
                
                cm = plt.cm.rainbow

                #                            Blue       Green      Red
                cm_bright = ListedColormap(["#5F00DB", "#5EDA94", "#DB0000"]) #in order: bll, fsrq, psr

                ax = plt.subplot(len(self.datasets), col, i)

                clf = make_pipeline(StandardScaler(), clf)
                
                
                if doscore:
                    
                    clf2 = clf
                    
                    X_train, X_test, y_train, y_test = train_test_split(
                        theX, they, test_size=0.4, random_state=42
                    )
                    
                    clf2.fit(X_train, y_train.to_numpy(dtype='int64'))
                    
                    score = clf2.score(X_test, y_test.to_numpy(dtype='int64'))
                    total += score
                    
                    
                clf.fit(theX, they.to_numpy(dtype='int64'))
                
                DecisionBoundaryDisplay.from_estimator(
                    clf, theX, cmap=cm, alpha=0.8, ax=ax, eps=0.5
                )

                #Plot the training points
                if trainPoints:
                    ax.scatter(
                       theX.iloc[:, 0], theX.iloc[:, 1], c=they, cmap=cm_bright, edgecolors="k", alpha=0.6,
                    )
                
                # Plot the testing points
                if testPoints:
                    ycopy = self.str2num(ynew)
                    ax.scatter(
                        Xnew.iloc[:, 0],
                        Xnew.iloc[:, 1],
                        c=ycopy.astype(int),
                        marker = '*',
                        cmap=cm_bright,
                        edgecolors="k",
                        alpha=0.8,
                    )

                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_xticks(())
                ax.set_yticks(())
                
                
                #if plt_type == 'grid' and ds_cnt == 1:
                    #ax.set_title(name)
                #elif plt_type != 'grid' and ds_cnt == 0:
                 #   ax.set_title(name)
                
                if doscore:
                    ax.text(
                        x_max - 0.3,
                        y_min + 0.3,
                        ("%.2f" % score).lstrip("0"),
                        size=15,
                        horizontalalignment="right",
                    )



                if plt_type == "single": 
                    ax.set_title(name)
                    filename = newp/f'{name}'
                    plt.tight_layout()
                    plt.savefig(f'{filename}_plt_{i}.png', bbox_inches='tight')

                    figure.clear()
                
                i += 1
                j += 1
            
            if plt_type != "single":
                total = total/6
                if doscore:  
                    dispsco = ("%.2f" % total).lstrip("0")
                    plt.subplot(632).set_title(f"{name}\n{dispsco}%")
                else:
                    plt.subplot(632).set_title(f'{name}')
                    
                self.scores[k] = total    
                
                filename = self.p/f'{name}'
                plt.tight_layout()
                plt.savefig(f'{filename}_plt.png', bbox_inches='tight')
                
                figure.clear()
            k += 1

                  
    def str2num(self,data):
        
        datacopy = data.copy()
        
        for t in range(data.size):
            if data[t] == 'bll':
                datacopy[t] = 0
            elif data[t] == 'fsrq':
                datacopy[t] = 1
            elif data[t] == 'psr':
                datacopy[t] = 2
                
        return datacopy
    
    def make_barplot(self):
    
        names = np.array(self.names)
        
        ind = np.flip(self.scores.argsort())
        
        df = pd.DataFrame({'score':self.scores[ind],'classifier':names[ind]})
        
        bc = sns.barplot(df, x = 'score', y = 'classifier')
        
        bc.set_xscale('log')
        ticks = [0.7, 0.75, 0.78, 0.8, 0.815]
        bc.set_xticks(ticks)
        bc.set_xticklabels(ticks)
        _=bc.set()
        
        
    
    def Compare_Variables(self):
        specs = MLspec(
                       features=('log_epeak','pindex','curvature','log_e0','var','log_nbb'), 
                       target=('association'), 
                       target_names=('bll','fsrq','psr')
                      )

        X, y = fs_data.getXy(specs)
        
        ycopy = self.str2num(y)
        
        ft = np.array(('log_epeak','pindex','curvature','log_e0','var','log_nbb'))

        X_train, X_test, y_train, y_test = train_test_split(X, ycopy, test_size=0.25, random_state=12)

        rf = RandomForestRegressor(n_estimators=100,max_features=2)
        rf.fit(X_train, y_train)

        perm_importance = permutation_importance(rf, X_test, y_test)

        ind = np.flip(perm_importance.importances_mean.argsort())

        df = pd.DataFrame({'Permutation Importance':perm_importance.importances_mean[ind],'features':ft[ind]})

        bc = sns.barplot(df, x = 'Permutation Importance', y = 'features')

        bc.set_xscale('linear')

        ticks = [0.02, 0.15, 0.4, 0.6]
        
        bc.set_xticks(ticks)
        
        bc.set_xticklabels(ticks)

        _=bc.set()