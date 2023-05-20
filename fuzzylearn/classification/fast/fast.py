from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from statistics import mode
import itertools
import asyncio
import time
import math

class FLfastClassifier:
    """FuzzyLearning class"""

    def __init__(self,*args,**kwargs):
        self.number_of_intervals = kwargs['number_of_intervals']
        self.metric=kwargs['metric']
        self.threshold=kwargs['threshold']
        self.trained = {}
        self.X_paired_weights=self.output_weights=None
        self.X_train_F = None
        self.smaller_better = True
        self.rhss=None
        self.lhss=None
    @property
    def number_of_intervals(self):
        return self._number_of_intervals

    @number_of_intervals.setter
    def number_of_intervals(self, value):
        self._number_of_intervals = value
    
    @property
    def metric(self):
        return self._metric

    @metric.setter
    def metric(self, value):
        self._metric = value
    
    @property
    def threshold(self):
        return self._threshold

    @threshold.setter
    def threshold(self, value):
        self._threshold = value

    def background(f):
        """A wrapper for asyncio task make for loops faster"""
        def wrapped(*args, **kwargs):
            return asyncio.get_event_loop().run_in_executor(None, f, *args, **kwargs)
    
    def _fuzzify_info(self,*args, **kwargs):
        """ An internal function to get a  fuzzifying info as a dictinary """

        split_dict = {}
        data = kwargs['data']
        if isinstance(data,pd.DataFrame):
           data,columns=self._pd_to_np(data=data)
        elif isinstance(data,np.ndarray):
            data = data
        else:
            raise ValueError(f"{type(data)} is not supported!")

        number_of_intervals = kwargs['number_of_intervals']
        index = 0
        n = len(data.T)
        for col in data.T:
            min_col = col.min()
            max_col = col.max()
            if isinstance(number_of_intervals,int):
                len_col = int((max_col-min_col)/number_of_intervals)
            if isinstance(number_of_intervals, list):
                if all([isinstance(item, int) for item in number_of_intervals]):
                    len_col = int((max_col-min_col)/number_of_intervals[index])
            if number_of_intervals=='sturges':
                len_col = int(math.log(n,2))+1
            if number_of_intervals=='rice':
                len_col = int(2*n**1/3)
            if number_of_intervals=='freedman':
                q3, q1 = np.percentile(data.T[:,index], [75 ,25])
                iqr = q3 - q1
                h = 2*iqr/(n**1/3)
                len_col = int((max_col-min_col)/h)
            split_dict[index] = [min_col,max_col,len_col]
            index+=1
        return split_dict

    def _np_to_pd(self,*args, **kwargs):
        """ An internal function to transform pandas dataframe to 2D numpy array """

        columns = kwargs["columns"]
        arr = kwargs["arr"]
        df = pd.DataFrame(arr, columns = columns)
        return df


    def _pd_to_np(self,*args, **kwargs):
        """ An internal function to transform 2D numpy array to pandas dataframe  """

        df = kwargs['data']
        columns = df.columns
        return df.to_numpy(),columns

    def _process_train_data(self,*args, **kwargs):
        """ An internal function to return X_train and y_train"""

        X_train=kwargs['X_train']
        y_train=kwargs['y_train']
        X_valid=kwargs['X_valid']
        y_valid=kwargs['y_valid']

        if isinstance(X_train,np.ndarray) and isinstance(y_train,np.ndarray):
            if isinstance(X_valid,np.ndarray) and isinstance(y_valid,np.ndarray):
                return X_train,y_train,X_valid,y_valid
            elif not isinstance(X_valid,np.ndarray) or not isinstance(y_valid,np.ndarray):
                raise ValueError("X_valid or y_valid are not in proper types !")
        if X_train is not None and isinstance(X_train,pd.DataFrame):
            X_train=X_train.reset_index(drop=True)
        if y_train is not None and isinstance(y_train,pd.DataFrame):
            y_train=y_train.reset_index(drop=True)
        if X_valid is not None and isinstance(X_valid,pd.DataFrame):
            X_valid=X_valid.reset_index(drop=True)
        if y_valid is not None and isinstance(y_valid,pd.DataFrame):
            y_valid=y_valid.reset_index(drop=True)

        return X_train,y_train,X_valid,y_valid

    def _fuzzifying(self,*args, **kwargs):
        """ An internal function to get a pandas dataframe and return fuzzifying of its variables """

        X=kwargs['X']
        if isinstance(X,pd.DataFrame):
           X,columns=self._pd_to_np(data=X)
        elif isinstance(X,np.ndarray):
            X = X
        else:
            raise ValueError(f"{type(X)} is not supported!")

        number_of_intervals=kwargs['number_of_intervals']
        split_dict = self._fuzzify_info(data = X, number_of_intervals=number_of_intervals)
        index = 0
        for _ in X.T:
            if split_dict[index][2]!=0 and not isinstance(split_dict[index][2],list):
                X[:,index]=((X[:,index]-split_dict[index][0])/split_dict[index][2]).round(decimals=0).astype(int)
            if isinstance(split_dict[index][2],list):
                X[:,index]=((X[:,index]-split_dict[index][0])/split_dict[index][2][index]).round(decimals=0).astype(int)
            index+=1
        return X

    def _lhs_rhs_creator(self,*args, **kwargs):
        trained = {}
        X = kwargs['X']
        y = kwargs['y']
        metric = kwargs['metric']
        threshold = kwargs['threshold']
    
        X_paired_weights = pairwise_distances(X, X,metric=metric)
        if self.smaller_better:
            X_paired_weights[X_paired_weights>self.threshold] = np.inf
        else:
            print('TODO')
        
        i_index = 0
        lhss = np.empty(shape=X.shape)
        rhss= np.empty(shape=y.shape)
        for row in X_paired_weights:
            lhs = []
            rhs =[]
            j_index=0
            for member in row:
                if member !=np.inf:
                    lhs.append(self.X_train_F[j_index,:])
                    rhs.append(y[j_index])
                j_index+=1
            temp = np.mean(lhs, 0).tolist()
            lhss[i_index] = temp
            rhs = list(itertools.chain(*rhs))
            

            rhss[i_index] = mode(rhs)
            i_index+=1
    
        return lhss,rhss


    def trained_model_for_X_y(self,*args, **kwargs):
        """ training function."""
        X=kwargs['X']
        y=kwargs['y']
        metric=kwargs['metric']
        threshold=kwargs['threshold']

        if isinstance(X,pd.DataFrame):
            X, X_cols = self._pd_to_np(data=X)
        if isinstance(y,pd.DataFrame):
            y, y_cols = self._pd_to_np(data=y)
        
        lhss,rhss = self._lhs_rhs_creator(X=X,y=y,metric=metric,threshold=threshold)

        return lhss,rhss

    def fit(self,*args, **kwargs):
        """ Fit function."""

        # arguments for the fit
        X_valid = kwargs['X_valid']
        y_valid = kwargs['y_valid']
        X_train = kwargs['X']
        y_train = kwargs['y']
        X_train,y_train,X_valid,y_valid = self._process_train_data(X_train=X_train,y_train=y_train,X_valid=X_valid,y_valid=y_valid)
        # fuzzifying X_train_new
        X_train_F = self._fuzzifying(X=X_train,number_of_intervals=self.number_of_intervals)
        self.X_train_F=X_train_F
        # training and return rhs and lhs
        self.lhss,self.rhss = self.trained_model_for_X_y(X=X_train_F,y=y_train,metric=self.metric,threshold=self.threshold)

        return self
    
    #@background
    def predict(self,*args, **kwargs):
        """Predict function"""
        X_test = kwargs['X']
        X_test_F = self._fuzzifying(X=X_test,number_of_intervals=self.number_of_intervals)
        predictions =[]
        y_paired_weights = pairwise_distances(X_test_F, self.lhss,metric=self.metric)

        index = 0
        for y_p in y_paired_weights:
            y_index= np.argmin(y_p,axis=0)
            predictions.append(self.rhss[y_index])
            index+=1


        return predictions

