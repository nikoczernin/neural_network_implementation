# Functions and variables for parameter ranges
import itertools
import math

import numpy as np





def get_MLP_parameter_ranges(input_data, output_data,
                             complexity_threshold=50
                             ):
    activation_functions = ["logistic", "tanh", "relu"]
    solvers = ["lbfgs", "sgd", "adam"]
    learning_rates = ["constant", "invscaling", "adaptive"]

    """
    alpha
    Alpha is a parameter for regularization term, aka penalty term, that combats overfitting 
    by constraining the size of the weights.Increasing alpha may fix high variance 
    (a sign of overfitting) by encouraging smaller weights, resulting in a decision boundary 
    plot that appears with lesser curvatures.Similarly, decreasing alpha may fix high bias (a sign 
    of underfitting) by encouraging larger weights, potentially resulting in a more complicated decision boundary.
    """
    learning_rate_init_range = (0.001, 0.4)
    alpha_range = tuple(np.logspace(-4, 1, 2))
    architecture_ranges = get_hidden_layer_architecture_ranges(input_data, output_data,
                                                               complexity_threshold=complexity_threshold)
    return {
        "activation": activation_functions,
        "solver": solvers,
        "learning_rate": learning_rates,
        "learning_rate_init": learning_rate_init_range,
        "alpha": alpha_range,
        "hidden_layer_sizes": architecture_ranges
    }


def get_hidden_layer_architecture_ranges(input_data, output_data, complexity_threshold=50):
    """
    Get a list of possible parameter for a given dataset
    :param input_data: Pandas dataframe with input data
    :param output_data: Panda series with output data
    :param complexity_threshold: The amount of features or classes to consider a dataset complex
    Complexity in this case warrants using more layers
    :return:
    """

    n_features = input_data.shape[1]
    n_classes = output_data.nunique()

    n_hidden_layers = [1, 2, 3]

    # for complex datasets (whatever complex may mean, we do more than [complexity_threshold] features
    # or classes), add 1 or 2 layers
    if n_features > complexity_threshold or n_classes > complexity_threshold:
        n_hidden_layers += [4, 5]

    n_neurons_min = min(n_features, n_classes)
    n_neurons_max = max(n_features * 2, n_features * 2 / 3 + n_classes, n_classes)
    print("Suggested numbers of hidden layers:", n_hidden_layers)
    print("Number of input neurons:", n_features)
    print("Number of output neurons:", n_classes)
    print("Possible numbers of neurons per hidden layer ranging from ", n_neurons_min,
          " to ", n_neurons_max)

    return {
        "n_hidden_layers": n_hidden_layers,
        "n_neurons_min": n_neurons_min,
        "n_neurons_max": n_neurons_max
    }


def get_hidden_layer_architecture_suggestions(input_data, output_data,
                                              n_neurons_steps=5, complexity_threshold=50):
    """
    Get a list of possible hidden layer architectures for a given dataset
    :param input_data: Pandas dataframe with input data
    :param output_data: Panda series with output data
    :param n_neurons_steps: Number of steps to take between min and max number of neurons
    :param complexity_threshold: The amount of features or classes to consider a dataset complex
    Complexity in this case warrants using more layers
    :return: List of possible architectures as tuples of neuron counts
    """

    n_rows = input_data.shape[0]
    n_features = input_data.shape[1]
    n_classes = output_data.nunique()

    architecture_ranges = get_hidden_layer_architecture_ranges(input_data, output_data,
                                                               complexity_threshold=complexity_threshold)
    n_neurons_max = architecture_ranges["n_neurons_max"]
    n_neurons_min = architecture_ranges["n_neurons_min"]
    n_hidden_layers = architecture_ranges["n_hidden_layers"]

    n_neuron_steps = math.ceil((n_neurons_max - n_neurons_min) / n_neurons_steps)
    n_hidden_neurons = list(range(
        n_neurons_min,  # min n of neurons
        n_neurons_max,  # max number of neurons
        n_neuron_steps  # steps
    ))

    print("Possible numbers of neurons per hidden layer:", n_hidden_neurons)

    # now take permutations the size of n_hidden_layers of these layer sizes
    # that will blow up the complexity
    architectures = []
    for num_layers in n_hidden_layers:
        for i in range(1, num_layers + 1):
            architectures += list(itertools.permutations(n_hidden_neurons, num_layers))
    # preliminary pruning
    # ???

    print("Number of architectures:", len(architectures))
    return architectures
