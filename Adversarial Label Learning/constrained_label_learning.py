import numpy as np
import codecs
import json
import tensorflow as tf
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dropout, Dense


# # # # # # # # # # # # # # # #
# supporting code             #
# # # # # # # # # # # # # # # #

def calculate_bounds(true_labels, predicted_labels, mask=None):
    """ Calculate error rate on data points the weak signals label """

    if len(true_labels.shape) == 1:
        predicted_labels = predicted_labels.ravel()
    assert predicted_labels.shape == true_labels.shape

    if mask is None:
        mask = np.ones(predicted_labels.shape)
    if len(true_labels.shape) == 1:
        mask = mask.ravel()

    error_rate = true_labels*(1-predicted_labels) + \
        predicted_labels*(1-true_labels)
    with np.errstate(divide='ignore', invalid='ignore'):
        error_rate = np.sum(error_rate*mask, axis=0) / np.sum(mask, axis=0)
        error_rate = np.nan_to_num(error_rate)

    # check results are scalars
    if np.isscalar(error_rate):
        error_rate = np.asarray([error_rate])
    return error_rate


def get_error_bounds(true_labels, weak_signals):
    """ Get error bounds of the weaks signals
        returns a list of size num_weak x num_classes
    """
    error_bounds = []
    mask = weak_signals >= 0

    for i, weak_probs in enumerate(weak_signals):
        active_mask = mask[i]
        error_rate = calculate_bounds(true_labels, weak_probs, active_mask)
        error_bounds.append(error_rate)
    return error_bounds

def majority_vote_signal(weak_signals):
    """ Calculate majority vote labels for the weak_signals"""

    baseline_weak_labels = np.rint(weak_signals)
    mv_weak_labels = np.ones(baseline_weak_labels.shape)
    mv_weak_labels[baseline_weak_labels == -1] = 0
    mv_weak_labels[baseline_weak_labels == 0] = -1
    mv_weak_labels = np.sign(np.sum(mv_weak_labels, axis=0))
    break_ties = np.random.randint(2, size=int(np.sum(mv_weak_labels == 0)))
    mv_weak_labels[mv_weak_labels == 0] = break_ties
    mv_weak_labels[mv_weak_labels == -1] = 0
    return mv_weak_labels

def build_constraints(a_matrix, bounds):
    """ params:
        a_matrix left hand matrix of the inequality size: num_weak x num_data x num_class type: ndarray
        bounds right hand vectors of the inequality size: num_weak x num_data type: ndarray
        return:
        dictionary containing constraint vectors
    """

    m, n, k = a_matrix.shape

    # # debugging code
    # print("\n\nbounds shape =", bounds.shape)
    # print("(m, k) =", (m, k), "\n\n")

    assert (m, k) == bounds.shape, \
        "The constraint matrix shapes don't match"

    constraints = dict()
    constraints['A'] = a_matrix
    constraints['b'] = bounds
    return constraints

def set_up_constraint(weak_signals, error_bounds):
    """ Set up error constraints for A and b matrices """

    constraint_set = dict()
    m, n, k = weak_signals.shape

    precision_amatrix = np.zeros((m, n, k))
    error_amatrix = np.zeros((m, n, k))
    constants = []

    # # debugging code
    # print("\n\nshape of weak signals", weak_signals.shape ,"\n\n")
    # print("\n\nshape of error amatrix", error_amatrix.shape ,"\n\n")

    for i, weak_signal in enumerate(weak_signals):
        active_signal = weak_signal >= 0
        precision_amatrix[i] = -1 * weak_signal * active_signal / \
            (np.sum(active_signal*weak_signal, axis=0) + 1e-8)
        error_amatrix[i] = (1 - 2 * weak_signal) * active_signal

        # error denom to check abstain signals
        error_denom = np.sum(active_signal, axis=0)
        error_amatrix[i] /= error_denom

        # constants for error constraints
        constant = (weak_signal*active_signal) / error_denom
        constants.append(constant)

    # set up error upper bounds constraints
    constants = np.sum(constants, axis=1)

    # # debugging code
    # print("\n\nConstants shape =", constants.shape)
    # print("error_bounds shape =", error_bounds.shape, "\n\n")


    assert len(constants.shape) == len(error_bounds.shape)
    bounds = error_bounds - constants
    error_set = build_constraints(error_amatrix, bounds)
    constraint_set['error'] = error_set

    return constraint_set


# # # # # # # # # # # # # # # #
# Code for training labels    #
# # # # # # # # # # # # # # # #

def bound_loss(y, a_matrix, bounds):
    """
    Computes the gradient of lagrangian inequality penalty parameters
    :param y: size (num_data, num_class) of estimated labels for the data
    :type y: ndarray
    :param a_matrix: size (num_weak, num_data, num_class) of a constraint matrix
    :type a_matrix: ndarray
    :param bounds: size (num_weak, num_class) of the bounds for the constraint
    :type bounds: ndarray
    :return: loss of the constraint (num_weak, num_class)
    :rtype: ndarray
    """
    constraint = np.zeros(bounds.shape)
    n, k = y.shape

    for i, current_a in enumerate(a_matrix):
        constraint[i] = np.sum(current_a * y, axis=0)
    return constraint - bounds


def y_gradient(y, constraint_set):
    """
    Computes y gradient
    """
    constraint_keys = constraint_set['constraints']
    gradient = 0

    for key in constraint_keys:
        current_constraint = constraint_set[key]
        a_matrix = current_constraint['A']
        bound_loss = current_constraint['bound_loss']

        for i, current_a in enumerate(a_matrix):
            constraint = a_matrix[i]
            gradient += 2*constraint * bound_loss[i]
    return gradient


def run_constraints(y, rho, constraint_set, iters=300, enable_print=False):
    # Run constraints from CLL

    constraint_keys = constraint_set['constraints']
    n, k = y.shape
    rho = n
    grad_sum = 0
    lamdas_sum = 0

    for iter in range(iters):
        print_constraints = [iter]
        print_builder = "Iteration %d, "
        constraint_viol = []
        viol_text = ''

        for key in constraint_keys:
            current_constraint = constraint_set[key]
            a_matrix = current_constraint['A']
            bounds = current_constraint['b']

            # get bound loss for constraint
            loss = bound_loss(y, a_matrix, bounds)
            # update constraint values
            constraint_set[key]['bound_loss'] = loss

            violation = np.linalg.norm(loss.clip(min=0))
            print_builder += key + "_viol: %.4e "
            print_constraints.append(violation)

            viol_text += key + "_viol: %.4e "
            constraint_viol.append(violation)

        y_grad = y_gradient(y, constraint_set)
        grad_sum += y_grad**2
        y = y - y_grad / np.sqrt(grad_sum + 1e-8)
        y = np.clip(y, a_min=0, a_max=1)

        constraint_set['violation'] = [viol_text, constraint_viol]
        if enable_print:
            print(print_builder % tuple(print_constraints))
    return y


def train_algorithm(constraint_set):
    """
    Trains CLL algorithm
    :param constraint_set: dictionary containing error constraints of the weak signals
    :return: average of learned labels over several trials
    :rtype: ndarray
    """
    constraint_set['constraints'] = ['error']
    weak_signals = constraint_set['weak_signals']

    # # debugging code
    # print("\n\n num weak_signals =", len(weak_signals.shape))
    # print("weak_signals =", weak_signals.shape, "\n\n")


    assert len(weak_signals.shape) == 3, "Reshape weak signals to num_weak x num_data x num_class"
    m, n, k = weak_signals.shape
    # initialize y
    y = np.random.rand(n, k)
    # initialize hyperparameters
    rho = 0.1

    t = 3  # number of random trials
    ys = []
    for i in range(t):
        ys.append(run_constraints(y, rho, constraint_set))
    return np.mean(ys, axis=0)



# # # # # # # # # # # # # # # # # # #
# Driver for COMBINED CODE          #
# # # # # # # # # # # # # # # # # # #

def constrained_label_learning(train_data, weak_signals, weak_errors):
    """ 
    Creates probabilistic labels to train data on

    :param train_data: 
    :type  train_data:

    :param weak_signals: 
    :type  weak_signals:

    :param weak_errors: 
    :type  weak_errors:

    :return: 
    :rtype: 
    """

    

    n_examples = train_data.shape[1]
    # labels = 0.5 * np.ones(n_examples)

    # Might get rid of this later, just so I don't accidentally change up the weak signals too much
    new_weak_signals = weak_signals

    new_weak_signals = np.flip(new_weak_signals.T, axis=None)
    

    # Make sure weak signals and weak_errors have the correct dimensions
    if len(new_weak_signals.shape) == 2:
        new_weak_signals = np.expand_dims(new_weak_signals.T, axis=-1)
    
    m, n, k = new_weak_signals.shape
    weak_errors = np.ones((m, k)) * 0.01

    # # debugging code
    # print("\n\nShape weak_errors:", weak_errors.shape)
    # print("Shape weak_signals:", new_weak_signals.shape, "\n\n")
    
    # if len(weak_errors.shape) == 2:
    #     weak_errors = np.expand_dims(weak_errors.T, axis=-1)

    # weak_errors = np.flip(weak_errors.T, axis=None)


    constraints = set_up_constraint(new_weak_signals, weak_errors)
    constraints['weak_signals'] = new_weak_signals
    y = train_algorithm(constraints)



    return y



# # # # # # # # # # # # # # # # # # # # # # #
# Main driver code from STANDALONE CODE     #
# # # # # # # # # # # # # # # # # # # # # # #

def run_CLL_experiment(dataset, true_bound=False):
    """ Run CLL experiments """

    # batch_size = 32
    train_data, train_labels = dataset['train']
    # test_data, test_labels = dataset['test']
    weak_signals = dataset['weak_signals']
    m, n, k = weak_signals.shape

    weak_errors = np.ones((m, k)) * 0.01

    # if true_bound:
    #     weak_errors = get_error_bounds(train_labels, weak_signals)
    #     weak_errors = np.asarray(weak_errors)

    # Set up the constraints
    constraints = set_up_constraint(weak_signals, weak_errors)
    constraints['weak_signals'] = weak_signals
    # mv_labels = majority_vote_signal(weak_signals)

    y = train_algorithm(constraints)
    # accuracy = accuracy_score(train_labels, y)
    # model = mlp_model(train_data.shape[1], k)
    # model.fit(train_data, y, batch_size=batch_size, epochs=20, verbose=1)
    # test_predictions = model.predict(test_data)
    # test_accuracy = accuracy_score(test_labels, test_predictions)

    # print("CLL Label accuracy is: ", accuracy)
    # print("CLL Test accuracy is: \n", test_accuracy)
    # print("Majority vote accuracy is: ", accuracy_score(train_labels, mv_labels))