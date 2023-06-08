from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from statistics import mode
import itertools
import math
from sklearn.metrics import *
from fuzzylearn.util.read_data import read_yaml_file
import optuna
from sklearn.metrics import roc_auc_score
import yaml

class FLOptimumClassifier:
    """FuzzyLearning class"""

    def __init__(self,*args,**kwargs):
        self.number_of_intervals = kwargs['number_of_intervals']
        self.metric=kwargs['metric']
        self.threshold=kwargs['threshold']
        self.optimizer = kwargs['optimizer']
        self.trained = {}
        self.X_paired_weights=self.output_weights=None
        self.X_train_F = None
        self.X_train = None
        self.y_train = None
        self.X_valid = None
        self.y_valid = None
        self.smaller_better = True
        self.rhss=None
        self.lhss=None
        self.features = None
        self.y_for_color_code = None
        self.metric_for_optimum = None
        self.threshold_for_optimum = None
        self.number_of_intervals_for_optimum = None
        self.metric_for_optimization = None
        try:        
            self.n_trials = kwargs['n_trials']
        except:
            self.n_trials = None
    @property
    def number_of_intervals_for_optimum(self):
        return self._number_of_intervals_for_optimum

    @number_of_intervals_for_optimum.setter
    def number_of_intervals_for_optimum(self, value):
        self._number_of_intervals_for_optimum = value
    
    @property
    def threshold_for_optimum(self):
        return self._threshold_for_optimum

    @threshold_for_optimum.setter
    def threshold_for_optimum(self, value):
        self._threshold_for_optimum = value
    
    @property
    def metric_for_optimum(self):
        return self._metric_for_optimum

    @metric_for_optimum.setter
    def metric_for_optimum(self, value):
        self._metric_for_optimum = value
    
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
    def y_for_color_code(self):
        return self._y_for_color_code

    @y_for_color_code.setter
    def y_for_color_code(self, value):
        self._y_for_color_code = value

    @property   
    def X_train(self):
        return self._X_train

    @X_train.setter
    def X_train(self, value):
        self._X_train = value

    @property   
    def y_train(self):
        return self._y_train

    @y_train.setter
    def y_train(self, value):
        self._y_train = value

    @property   
    def X_valid(self):
        return self._X_valid

    @X_valid.setter
    def X_valid(self, value):
        self._X_valid = value
    @property   
    def y_valid(self):
        return self._y_valid

    @y_valid.setter
    def y_valid(self, value):
        self._y_valid = value

    @property   
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value

    @property   
    def metric_for_optimization(self):
        return self._metric_for_optimization

    @metric_for_optimization.setter
    def metric_for_optimization(self, value):
        self._metric_for_optimization= value

    @property   
    def n_trials(self):
        return self._n_trials

    @n_trials.setter
    def n_trials(self, value):
        self._n_trials= value

    def _optimizer_func(self,*args, **kwargs):
        optimizer=self.optimizer

        # Set the random seed
        seed = 42
        np.random.seed(seed)

        # Get the number of rows in each array
        if self.X_valid is None or self.y_valid is None:

            num_rows = self.X_train.shape[0]

            if isinstance(self.X_train, np.ndarray):
                # # Generate random indices for selecting rows
                # random_indices = np.random.choice(num_rows, size=int(0.5*num_rows), replace=False)
                # # Select rows based on the random indices from self.X_train
                # self.X_valid = self.X_train[random_indices]
                # self.y_valid = self.y_train[random_indices]
                self.X_valid,X_valid_test,self.y_valid,y_valid_test = train_test_split(self.X_train,self.y_train,test_size=.5) 
                num_unique = len(np.unique(self.y_valid))<=2
                if len(np.unique(self.y_valid))<=2:
                    problem_type = 'binery'
                else:
                    problem_type = 'multi-class'

            elif isinstance(self.X_train,pd.DataFrame):
                # self.X_valid = self.X_train.sample(n=int(0.5*num_rows), random_state=seed)
                # self.y_valid = self.y_train.sample(n=int(0.5*num_rows), random_state=seed)
                # self.X_valid_test = self.X_train.sample(n=int(0.5*num_rows), random_state=seed)
                # self.y_valid = self.y_train.sample(n=int(0.5*num_rows), random_state=seed)
                print(self.X_train.shape)
                print(self.y_train.shape)
                self.X_valid,X_valid_test,self.y_valid,y_valid_test = train_test_split(self.X_train,self.y_train,test_size=.5) 


                num_unique = len(self.y_valid.nunique())<=2
                if len(self.y_valid.nunique())<=2:
                    problem_type = 'binery'
                else:
                    problem_type = 'multi-class'

            else:
                raise ValueError(f"this type of data {type(self.X_train)} is not supported for X_valid !!!")
        else: 

            if isinstance(self.X_valid, np.ndarray):
                # Select rows based on the random indices from self.X_train
                num_unique = len(np.unique(self.y_valid))<=2
                if num_unique:
                    problem_type = 'binery'
                else:
                    problem_type = 'multi-class'

            elif isinstance(self.X_train,pd.DataFrame):
                num_unique = self.y_valid.nunique()<=2
                if num_unique:
                    problem_type = 'binery'
                else:
                    problem_type = 'multi-class'

            else:
                raise ValueError(f"this type of data {type(self.X_train)} is not supported for X_valid !!!")
            
    
        # Path to your YAML file
        file_path = 'fuzzylearn/optimization_conf.yaml'

        # Read the YAML file
        with open(file_path, 'r') as file:
            try:
                yaml_data = yaml.safe_load(file)
            except yaml.YAMLError as e:
                print("Error loading YAML file:", e)  
        metrics_for_classification=yaml_data['metrics_for_classification']
        metrics_with_larger_is_better=yaml_data['metrics_with_larger_is_better']
        metrics_for_classification_multi=yaml_data['metrics_for_classification_multi']
        if optimizer=='auto_optuna':                
            if problem_type =='binery':
                metrics_for_classification = "roc_auc_score(y_true,y_pred)"
            if problem_type =='multi-class':
                metrics_for_classification = "roc_auc_score(y_true, y_score)"

            direction="maximize"
            n_trials=None
        if optimizer=='optuna':                
            if problem_type =='binery':
                for mtr in metrics_with_larger_is_better:
                    if self.metric_for_optimization  in mtr:
                        direction="maximize"
                    else:
                        direction="Minimize"
                for mtr in metrics_for_classification:
                    if self.metric_for_optimization in mtr:
                        metrics_for_classification = mtr
                    else:
                        raise ValueError(f'{self.metric_for_optimization} is not supported !')
            if problem_type =='multi-class':
                for mtr in metrics_with_larger_is_better:
                    if self.metric_for_optimization  in mtr:
                        direction="maximize"
                    else:
                        direction="Minimize"
                for mtr in metrics_for_classification_multi:
                    if self.metric_for_optimization in mtr:
                        metrics_for_classification = mtr
                    else:
                        raise ValueError(f'{self.metric_for_optimization} is not supported !')
            n_trials=self.n_trials

        def objective(trial):

            # Define selection parameter
            self.metric_for_optimum = trial.suggest_categorical("metric_for_optimum", self.metric_for_optimum)
            self.number_of_intervals_for_optimum = trial.suggest_int("number_of_intervals_for_optimum", int(self.number_of_intervals_for_optimum[0]),int(self.number_of_intervals_for_optimum[len(number_of_intervals_for_optimum)-1]))
            self.threshold_for_optimum = trial.suggest_float("threshold_for_optimum", float(self.threshold_for_optimum[0]),float(self.threshold_for_optimum[len(self.threshold_for_optimum)-1]))

            params = {
                "metric": metric_for_optimum,
                "number_of_intervals": number_of_intervals_for_optimum,
                "threshold": threshold_for_optimum,
                "optimizer": "auto_optuna",
            }

            model = FLOptimumClassifier(**params)
            model.fit(X=self.X_valid,y=self.y_valid,X_valid=None,y_valid=None)
            y_pred = model.predict(X=X_valid_test)
            y_true = y_valid_test
            # Calculate accuracy score as the objective
            score = eval(metrics_for_classification)
            if trial.number > 500 or score > 0.90:
                # Stop the study if the results are already good enough
                study.stop()

            return score

        study = optuna.create_study(direction=direction)
        study.optimize(objective, n_trials=n_trials)

    def _fuzzify_info(self,*args, **kwargs):
        """ An internal function to get a  fuzzifying info as a dictinary """

        split_dict = {}
        data = kwargs['data']
        if isinstance(data,pd.DataFrame):
           data,columns=self._pd_to_np(data=data)
           data = data.T
        elif isinstance(data,np.ndarray):
            data = data.T
        else:
            raise ValueError(f"{type(data)} is not supported!")

        number_of_intervals = kwargs['number_of_intervals']
        index = 0
        n = len(data)
        for col in data:
            min_col = col.min()
            max_col = col.max()
            if len(np.unique(col)) <=5:
                    ordered_list = sorted(col.tolist())
                    subtracted_list = [ordered_list[i] - ordered_list[i-1] for i in range(1, len(ordered_list))]
                    if abs(min(subtracted_list)) > 0:
                        len_col = abs(min(subtracted_list)/2.0)
                    elif abs(max(subtracted_list) - min(subtracted_list)) > 0:
                        len_col = abs((max(subtracted_list)-min(subtracted_list))/2)
                    else:
                        raise ValueError(f'All value zeros for {col} is not accepted!') 

            else:
                if isinstance(number_of_intervals,int):
                    len_col = abs((max_col-min_col)/number_of_intervals)       
                if isinstance(number_of_intervals, list):
                    if all([isinstance(item, int) for item in number_of_intervals]):
                        len_col = abs((max_col-min_col)/number_of_intervals[index])
                if number_of_intervals=='sturges':
                    len_col = abs(math.log(n,2))+1
                if number_of_intervals=='rice':
                    len_col = abs(2*n**1/3)
                if number_of_intervals=='freedman':
                    q3, q1 = np.percentile(data.T[:,index], [75 ,25])
                    iqr = q3 - q1
                    h = 2*iqr/(n**1/3)
                    len_col = abs((max_col-min_col)/h)
            try:
                split_dict[index] = [min_col,max_col,len_col]
            except:
                print('number_of_intervals',number_of_intervals)
                print('max_col',max_col)
                print('min_col',min_col)
                print('abs((max_col-min_col)/number_of_intervals',abs((max_col-min_col)/number_of_intervals)) 
                print(len(np.unique(col)))
                ordered_list = sorted(col.tolist())
                subtracted_list = [ordered_list[i] - ordered_list[i-1] for i in range(1, len(ordered_list))]
                print('abs(min(subtracted_list))',abs(min(subtracted_list)))
                print('abs(max(subtracted_list) - min(subtracted_list))',abs(max(subtracted_list) - min(subtracted_list)))
                print('min_col', min_col)
                print('max_col',max_col)
                print('index', index)
                print('col',col) 
                print('len_col', len_col)
               
            index+=1
        return split_dict

    def _np_to_pd(self,*args, **kwargs):
        """ An internal function to transform pandas dataframe to 2D numpy array """

        columns = kwargs["columns"]
        arr = kwargs["arr"]
        self.features = columns.to_list()
        df = pd.DataFrame(arr, columns = columns)
        return df


    def _pd_to_np(self,*args, **kwargs):
        """ An internal function to transform 2D numpy array to pandas dataframe  """

        df = kwargs['data']
        columns = df.columns
        self.features = columns.to_list()
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
            elif X_valid is None or y_valid is None:
                return X_train,y_train,X_valid,y_valid
            else:
                raise ValueError("X_valid or y_valid are not in proper types !")
        if X_train is not None and isinstance(X_train,pd.DataFrame):
            X_train=X_train.reset_index(drop=True)
        if y_train is not None and isinstance(y_train,pd.DataFrame):
            y_train=y_train.reset_index(drop=True)
            self.y_for_color_code=y_train
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
            X_paired_weights[X_paired_weights<self.threshold] = np.inf
        
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
            # next lines flatten a list of list to list if any otherwise don't
            # for element in rhs: 
            #     if isinstance(element,list):
            #         rhs = list(itertools.chain(*rhs))
            #     break;
            try :    
                rhs = list(itertools.chain(*rhs))
                rhss[i_index] = mode(rhs)
            except:
                # TODO check this line for np arrays
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
        self.X_valid = kwargs['X_valid']
        self.y_valid = kwargs['y_valid']
        self.X_train = kwargs['X']
        self.y_train = kwargs['y']
        
        
        if self.optimizer=="auto_optuna":
            best_params = self._optimizer_func(optimizer="auto_optuna")
            self.number_of_intervals=best_params['number_of_intervals']
            self.metric=best_params['metric']
            self.threshold=best_params['threshold']

        X_train,y_train,X_valid,y_valid = self._process_train_data(X_train=self.X_train,y_train=self.y_train,X_valid=self.X_valid,y_valid=self.y_valid)
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

    
    def feature_improtance(self,*args,**kwargs):
        """Feature improtance"""
        from collections import OrderedDict
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6,8))

        lhss=self.lhss
        # Get the number of columns in the array
        num_columns = lhss.shape[1]

        # Generate the x-axis values for the bars
        x_values = np.arange(num_columns)
        x_values = self.features
        # Example list of numbers to determine colors
        colors = list(set(self.y_for_color_code.iloc[:,0].to_list()))
        colors = [float(x) for x in colors]

        # Create a color map
        cmap = plt.cm.get_cmap('viridis')  # Choose a colormap, such as 'cool'
        
        # Create lines connecting the bars
        for i in range(lhss.shape[0] - 1):
            plt.plot(lhss[i, :],x_values, linestyle='-', color=cmap(colors[int(self.y_for_color_code.iloc[i,:].to_list()[0])]), label = str(int(self.y_for_color_code.iloc[i,:].to_list()[0])))
        
        plt.legend(self.y_for_color_code)

        # Set labels and a title for the plot
        plt.xlabel('Levels')
        plt.ylabel('Features/Variables')
        plt.title('Fuzzified Features Map')
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = OrderedDict(zip(labels, handles))

        # Display a legend for the rows
        plt.legend(by_label.values(), by_label.keys())

        # Show the plot
        plt.show()