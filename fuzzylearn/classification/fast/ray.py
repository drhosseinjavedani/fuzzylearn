from sklearn.model_selection import train_test_split
from fuzzylearn.interfaces.interfaces import IFLRayClassifier
from sklearn.metrics.pairwise import pairwise_distances
from statistics import mode
from sklearn.metrics import *
from fuzzylearn.util.read_data import read_yaml_file
from sklearn.metrics import roc_auc_score
import ray
from fuzzylearn.util.helpers import fuzzifying,process_train_data,trained_model_for_X_y
from fuzzylearn.classification.fast.fast import FLClassifier
ray.init()

class FLRayClassifier:
    """FuzzyLearning class"""

    def __init__(self,*args,**kwargs):
        self.number_of_intervals = kwargs['number_of_intervals']
        self.metric=kwargs['metric']
        self.threshold=kwargs['threshold']
        self.fuzzy_type = kwargs['fuzzy_type']
        try:        
            self.n_trials = kwargs['n_trials']
        except:
            self.n_trials = None
        try:        
            self.fuzzy_cut = kwargs['fuzzy_cut']
        except:
            self.fuzzy_cut = None
        self.iflrayclaccifier = IFLRayClassifier.remote(
            number_of_intervals =self.number_of_intervals,
            metric=self.metric,
            threshold=self.threshold,
            fuzzy_type = self.fuzzy_type,
            fuzzy_cut = self.fuzzy_cut,
            n_trials=self.n_trials)


    def __str__(self):
        return f"number_of_intervals :{self.number_of_intervals} \n metric : {self.metric} \n threshold: {self.threshold} \n "

    @property
    def iflrayclaccifier(self):
        return self._iflrayclaccifier

    @iflrayclaccifier.setter
    def iflrayclaccifier(self, value):
        self._iflrayclaccifier= value
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

    @property   
    def features(self):
        return self._features

    @features.setter
    def features(self, value):
        self._features = value

    @property   
    def n_trials(self):
        return self._n_trials

    @n_trials.setter
    def n_trials(self, value):
        self._n_trials= value
   

    @property   
    def fuzzy_type(self):
        return self._fuzzy_type

    @fuzzy_type.setter
    def fuzzy_type(self, value):
        self._fuzzy_type= value
   

    @property   
    def fuzzy_cut(self):
        return self._fuzzy_cut

    @fuzzy_cut.setter
    def fuzzy_cut(self, value):
        self._fuzzy_cut= value
   

    def fit(self,*args, **kwargs):
        """ Fit function."""
        
        return ray.get(self.iflrayclaccifier.fit.remote(*args,**kwargs))
    
    def predict(self,*args, **kwargs):
        """Predict function"""
        return ray.get(self.iflrayclaccifier.predict.remote(*args,**kwargs))
