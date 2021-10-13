import random
import sys  # base class

import numpy as np  # baseclass

from BaseClassifier import BaseClassifier
# from setup_model import set_up_constraint, mlp_model
from _optimize_stochgall import optimize
from setup_model import mlp_model


class ALL(BaseClassifier):
    """
    Multi Class Adversarial Label Learning Classifier

    This class implements Multi ALL training on a set of data

    Parameters
    ----------
    max_iter : int, default=300
        For optimize

    """

    def __init__(self, max_iter=300, max_epoch=20, rho=0.1, loss='multilabel', batch_size=32):
    
        self.max_iter = max_iter
        self.max_epoch = max_epoch
        self.model = None
        self.rho = rho
        self.loss = loss
        self.batch_size = batch_size

    def predict_proba(self, X):
        if self.model is None:
            sys.exit("no model")

        to_return = self.model.predict(X)
        
        return to_return.flatten()    

    def fit(self, X, weak_signals_probas, weak_signals_error_bounds):
        """
        Fits MultiAll model

        Parameters
        ----------
        X: ndarray of shape (n_examples, n_features)
            Training matrix, where n_examples is the number of examples and 
            n_features is the number of features for each example

        weak_signals_probas: ndarray of shape (n_weak_signals, n_examples, n_classes)
            A set of soft or hard weak estimates for data examples.
            This may later be changed to accept just the weak signals, and these 
            probabilities will be calculated within the ALL class. 

        weak_signals_error_bounds: dictionary
            error constraints (a_matrix and bounds) of the weak signals. Contains both 
            left (a_matrix) and right (bounds) hand matrix of the inequality 

        Returns
        -------
        self
            Fitted estimator

        """

        # original variables
        # constraint_keys = ["error"]
        num_weak_signals = weak_signals_probas.shape[0]

        active_signals = weak_signals_probas >= 0
        # active_signals = weak_signals_probas[:num_weak_signals, :] >= 0

        # todo: fix this to allow precision
        weak_signals_precision = np.zeros(weak_signals_probas.shape)
        
        constraint_set = {}
        constraint_set['error'] = weak_signals_error_bounds
        # constraint_set['constraints'] = constraint_keys
        constraint_set['weak_signals'] = weak_signals_probas[:num_weak_signals, :, :] * active_signals[:num_weak_signals, :, :]
        constraint_set['num_weak_signals'] = num_weak_signals
        constraint_set['loss'] = self.loss

        # Code for fitting algo
        results = dict()

        m, n, k = constraint_set['weak_signals'].shape

        m = 2 if k == 1 else k

        # initialize final values
        learnable_probabilities = np.ones((n, k)) * 1/m
        y = np.ones((n, k)) * 0.1
        assert y.shape[0] == X.shape[0]

        # initialize hyperparams todo: can this be in init?
        rho = 0.1
        loss = 'multilabel'
        batch_size = 32

        # This is to prevent the learning algo from wasting effort fitting a model to arbitrary y values.
        y, constraint_set = optimize(y, learnable_probabilities, rho, constraint_set, optim='max')

        self.model = mlp_model(X.shape[1], k)

        grad_sum = 0
        epoch = 0
        while epoch < self.max_epoch:
            indices = list(range(n))
            random.shuffle(indices)
            batches = np.array_split(indices, int(n / batch_size))

            rate = 1.0
            old_y = y.copy()

            if epoch % 1 == 0:
                for batch in batches:
                    self.model.train_on_batch(X[batch], y[batch])
                learnable_probabilities = self.model.predict(X)
            
            if epoch % 2 == 0:
                y, constraint_set = optimize(y, learnable_probabilities, rho, constraint_set, iters=10, enable_print=False)
            
            epoch += 1

        return self
