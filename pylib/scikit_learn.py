"""
Interface to scikit-learn, adapting a DataFrame like interface
"""

import numpy as np
import pandas as pd


class SKlearn():
    """
    df: DataFrame
    skprop: dict with keys:
            features: list of columns in the dataframe
            targets: dict, keys names of groups, values list of column names
            model_name:
            truth_field : column with "truth"
            target_field: well create column with target
    """

    def __init__(self, 
                 df:pd.DataFrame, 
                 skprop:dict):

        self._set_model(skprop)
        self._set_df(df)

    def __repr__(self):
        return f"""\
SKlearn specifications: 
* features: {', '.join(self.features)}
* classes: {', '.join(self.targets.keys())}
* model: {self.model}"""
    
    def _set_model(self, skprop):
        self.__dict__.update(skprop)

        from sklearn.naive_bayes import GaussianNB 
        from sklearn.svm import  SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.ensemble import RandomForestClassifier

        # instantiate the model by looking up the name
        cdict = dict(GNB = (GaussianNB, {}),
                    SVC = (SVC, dict(gamma=2, C=1)), 
                    tree= (DecisionTreeClassifier, {}),
                    RFC = (RandomForestClassifier, dict(n_estimators=100, max_features=2)),
                    NN  = (MLPClassifier, dict(alpha=1, max_iter=1000)),
                    )
        F,kw = cdict[self.model_name]

        self.model = F(**kw)


    def _set_df(self, df):
        """
        Set up a DataFrame for fitting:
        Add a column 'target' to the DataFrame `df` depending on column 'association'
        If if is in the targets list, set the target name
        """

        assert np.all(np.isin(self.features, df.columns) ),f'One or more feature {self.feature} missing'
        assert self.truth_field in df.columns, 'No "association" column'
        cdict = dict()
        for key,vars  in self.targets.items():
            for var in vars:
                cdict[var] = key
        df[self.target_field] = df.association.apply(lambda x: cdict.get(x, np.nan) )
        self.target_counts = df.groupby(self.target_field, sort=False).size()
        assert np.all(self.target_counts>0), f'empty target category {self.target_counts}'
        self.df = df

    @property
    def target_names(self):
        return self.target_counts.keys()
    

    def getXy(self,df=None) : 
        """Return an X,y pair for ML training
        """
        if df is not None: self.set_df(df)
        df = self.df
        tsel = ~pd.isna(df[self.target_field])
        assert sum(tsel)>0, 'No data selected for training.'
        return df.loc[tsel, self.features], df.loc[tsel, self.target_field]
    
    def fit(self):
        """
        """
        X,y = self.getXy() 
        return self.model.fit(X,y)

    def predict(self, query=None):
        """Return a "prediction" vector using the classifier, required to be a trained model

        - query -- optional query string
        return a Series 
        """
        # the feature names used for the classification -- expect all to be in the dataframe
        assert hasattr(self, 'classifier'), 'Model was not fit'
        fnames = getattr(self.classifier, 'feature_names_in_', [])
        assert np.all(np.isin(fnames, self.df.columns)), f'classifier not set properly'
        dfq = self.df if query is None else self.df.query(query) 
        assert len(dfq)>0, 'No data selected'
        ypred = self.classifier.predict(dfq.loc[:,fnames])
        return pd.Series(ypred, index=dfq.index, name='prediction')

    def train_predict(self):
        
        model = self.model 
        try:
            X,y = self.getXy() 
            model.probability=True # needed to get probabilities
            self.classifier =  model.fit(X,y)

        except ValueError as err:
            print(f"""Bad data? {err}""")
            print(self.df.loc[:,self.features].describe())
            return

        self.df['prediction'] = self.predict()

    def predict_prob(self, query=None):
        """Return DF with fit probabilities
        """
        mdl = self.classifier
        assert mdl.probability, 'Fit must be with probability  True' 
        dfq = self.df.query(query) if query is not None else self.df
        X = dfq.loc[:, self.features]
        return pd.DataFrame(mdl.predict_proba(X), index=dfq.index,
                            columns=['p_'+ n for n in self.target_names])
    
    
    
    
    
    def getPrecRec(self, X, y, ax):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.333)

        classifier = self.classifier.fit(X_train, y_train)

        #Get precision and recall values
        tem = PrecisionRecallDisplay.from_estimator(
            classifier, X_test, y_test, name=name, ax=ax, plot_chance_level=True
        )
        tem.figure_.clear()
    
        return tem


    #Takes a tuple of classifiers and a tuple of their names
    def pltAvgPrecRec(self, classifiers, names):

        #set up
        f = plt.axes()

        thePrec = np.empty(0)
        theRecall = np.empty(0)
        theSet = np.empty(0)


        #Loop through classifiers
        for name, clf in zip(names, classifiers):
            X,y = self.getXy()
            y = (y=='psr')

            pr = self.getPrecRec(X, y, f)

            count = 0
            prec = pr.precision
            recall = pr.recall


            while((count:=count+1) <= 20):

                pr = self.getPrecRec(X, y, f)

                if prec.size < pr.precision.size:
                    prec += pr.precision[:prec.size]
                    recall += pr.recall[:recall.size]
                else:
                    p = np.ones(prec.size)
                    r = np.ones(recall.size)

                    p = prec/count
                    r = recall/count

                    p[:pr.precision.size] = pr.precision
                    r[:pr.recall.size] = pr.recall
                    prec += p
                    recall += r



            theSet = np.concatenate((theSet, np.full((prec.size), name)))
            thePrec = np.concatenate((thePrec, prec/count))
            theRecall = np.concatenate((theRecall, recall/count))

        d = {"prec": thePrec, "recall": theRecall, "group": theSet}

        prdf = pd.DataFrame(data=d)

        sns.lineplot(data=prdf, x="recall", y="prec", hue='group')
