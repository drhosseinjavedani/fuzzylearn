from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import paired_distances
from statistics import mode
import itertools
import asyncio
import time


class FLClassifier:
    """FuzzyLearning class"""

    def __init__(self,*args,**kwargs):
        self.number_of_intervals = kwargs['number_of_intervals']
        self.metric=kwargs['metric']
        self.threshold=kwargs['threshold']
        self.trained = {}
    
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
        for col in data.T:
            min_col = col.min()
            max_col = col.max()
            len_col = int((max_col-min_col)/number_of_intervals)
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
            if split_dict[index][2]!=0:
                X[:,index]=((X[:,index]-split_dict[index][0])/split_dict[index][2]).round(decimals=0).astype(int)
            index+=1
        return X

    def _lhs_rhs_creator(self,*args, **kwargs):
        trained = {}
        X = kwargs['X']
        y = kwargs['y']
        metric = kwargs['metric']
        threshold = kwargs['threshold']
        first_index = 0
        for first_row in X:
            rhs = []
            lhs = []
            lhs.append(first_row)
            rhs.append(y[first_index,:])
            
            second_index = 0
            for _ in X:
                if first_index!=second_index:
                    sampled_row = X[first_index,:]
                    candidate_row = X[second_index,:]
                    distance_metric = paired_distances([sampled_row], [candidate_row],metric=metric)
                    if distance_metric < threshold:
                        lhs.append(candidate_row)
                        rhs.append(y[second_index,:])
                second_index +=1
            rhs = list(itertools.chain(*rhs))
            if len(rhs)>0:
                rhs=mode(rhs)
            lhs = [np.mean(lhs, 0).tolist()]
            trained[first_index]=[first_row,lhs,rhs]
            first_index+=1
        return trained


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
        
        trained = self._lhs_rhs_creator(X=X,y=y,metric=metric,threshold=threshold)

        self.trained=trained

        return self.trained

    #@background
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
        # training and return rhs and lhs
        self.trained = self.trained_model_for_X_y(X=X_train_F,y=y_train,metric=self.metric,threshold=self.threshold)

        return self
    
    #@background
    def predict(self,*args, **kwargs):
        """Predict function"""
        X_test = kwargs['X']
        X_test_F = self._fuzzifying(X=X_test,number_of_intervals=self.number_of_intervals)
        
        predictions =[]
        index=0
        for row in X_test_F:
            max_membership_for_sample = []
            sampled_row = row
            for index_in_train in self.trained:
                lhs= self.trained[index_in_train][1]
                max_ds = 0
                for lh in lhs:
                    paired_d = paired_distances([sampled_row], [lh],metric=self.metric)
                    max_ds = max_ds + paired_d[0]/len(lhs)
                max_membership_for_sample.append(max_ds)
            min_index=np.argmin(max_membership_for_sample)
            y_forecast = self.trained[min_index][2]
            predictions.append(y_forecast)

        return predictions


