import fermi_sources2
from fermi_sources2 import *

fs_data = FermiSources('files/fermi_sources_v2.csv')
unid =  FermiSources('files/fermi_sources_v2.csv')
specs1 = MLspec(features=('log_var', 'curvature', 'log_fpeak', 'log_epeak'), target='association')
specs2 = MLspec(features=('log_var', 'curvature', 'log_fpeak', 'log_epeak'), target='association', target_names=('unid'))

#fs_data = FermiSources('files/unid_table.csv')
#unid =  FermiSources('files/unid_table.csv')
#specs1 = MLspec(features=('log_nbb','pindex','curvature','log_e0'), target='association')
#specs2 = MLspec(features=('log_nbb','pindex','curvature','log_e0'), target='association', target_names=('unid'))

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
from sklearn.metrics import PrecisionRecallDisplay
from sklearn.metrics import precision_score


class Compare_Classifiers:
    names: list = [
        #"Nearest Neighbors",
        #"Linear SVM",
        "RBF SVM",
        #"Gaussian Process",
        #"Decision Tree",
        "Random Forest",
        "Neural Net",
        #"AdaBoost",
        #"Naive Bayes",
        #"QDA",
    ]
    
    classifiers: list = [
        #KNeighborsClassifier(3),
        #SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        #GaussianProcessClassifier(),
        #DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=100, max_features=2),
        MLPClassifier(alpha=1, max_iter=1000),
        #AdaBoostClassifier(),
        #GaussianNB(),
        #QuadraticDiscriminantAnalysis(),
    ]
    
    def __init__(self, tr_data, test_data, path, names=names, classifiers=classifiers):
        
        self.classifiers = classifiers
        
        self.proba = pd.DataFrame()
        
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
        
        #The unidentified sources
        self.testval = [
            (self.X_te.iloc[:,:2],self.y),
            (self.X_te.iloc[:,::2],self.y),
            (self.X_te.iloc[:,::3],self.y),
            (self.X_te.iloc[:,1:3],self.y),
            (self.X_te.iloc[:,1::2],self.y),
            (self.X_te.iloc[:,2:4],self.y)
        ]

        
        self.p = Path(path)
        self.p.mkdir(exist_ok=True)
        
        self.scores = np.empty(len(self.classifiers))
        
        
    def analyze_data(self):

        k = 0
        
        for name, clf in zip(self.names, self.classifiers):
            
            i = 1
            j = 0
            
            probaDF = pd.DataFrame()
            
            for ds_cnt, ds in enumerate(self.datasets):
                
                theX, they = ds
                model = clf
                model.fit(self.X,self.y)

                Xnew = self.testval[j][0]
                ynew = model.predict(self.X_te)
                
                #theproba = model.predict_proba(self.X_te)
                theproba = np.around(model.predict_proba(self.X_te), decimals = 3)
                #thisproba = pd.DataFrame(theproba, columns = ['bll','fsrq','psr'])
                #
                #probaDF+thisproba
                
                if j==0:
                    probaDF = pd.DataFrame(theproba, columns = ["bll", "fsrq", "psr"])
                    probaDF["Actual Classification"] = ynew

                
                i += 1
                j += 1
            
            k += 1
            
            #probaDF.divide(6)
            probaDF.to_csv(f"{name}Classification.csv", index=False)
            
            
            
    def analyze_PR(self):

        k = 0
        
        for name, clf in zip(self.names, self.classifiers):
            
            i = 1
            j = 0
            
            
            for ds_cnt, ds in enumerate(self.datasets):
                
                theX, they = ds
                
                X_train, X_test, y_train, y_test = train_test_split(theX, they, test_size=0.33, random_state=42)
                #print(f"X_train {X_train}, X_test{X_test}, y_train {y_train}, y_test{y_test}")
                
                model = clf
                
                model.fit(X_train, y_train)
                
                y_pred = model.predict_proba(X_test)[:, 1]
                
                #print(precision_score(y_test, y_pred, average=None))
                
                PrecisionRecallDisplay.from_estimator(clf, X_test, y_test, name=name, plot_chance_level=True)
                
                #return PrecisionRecallDisplay.from_predictions(y_test, y_pred)
                
                
            
            #k += 1