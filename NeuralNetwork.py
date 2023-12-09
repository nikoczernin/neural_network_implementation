from Help import *
import activationFunctions


class NeuralNetworkClassifier:
    # weights_init: "gaussian", "uniform"
    def __init__(self,
                 hidden_layer_sizes=(),
                 activation_functions=(),
                 output_activation_function=None,
                 weights_init="gaussian",
                 ):

        # the weights have the following structures:
        # weights is a list of numpy arrays
        # each numpy array is a matrix of weights
        # the first dimension is the number of neurons in the left layer
        # the second dimension is the number of neurons in the right layer
        # the first layer is the input layer
        # since the input layer is not determined until training, the weights are not initialized until training
        self.weights = []
        self.weights_init = weights_init

        # check if hidden_layer_sizes and activation_functions are tuples
        # make them into tuples otherwise
        if not isinstance(hidden_layer_sizes, tuple) and not isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = (hidden_layer_sizes,)
        if not isinstance(activation_functions, tuple) and not isinstance(activation_functions, list):
            activation_functions = (activation_functions,)
        # if they are not the same length, raise an error
        # if the same activation function shall be used for all layers, just pass a tuple with one element
        if len(hidden_layer_sizes) != len(activation_functions) and len(activation_functions) > 1:
            raise ValueError("hidden_layer_sizes and activation_functions must have the same length")

        # the input and output layer are not defined until fitting/training
        self.input_layer = None
        self.output_layer = None

        # add hidden layers
        self.layers = []
        for i, size in hidden_layer_sizes:
            # if each hidden layer gets its own activation function, use the one specified
            # otherwise use the only one specified
            activation_fun = activation_functions[min(i, len(activation_functions) - 1)]
            self.layers.append(NeuralNetworkLayer(size=size, activation_function=activation_fun))

        # the output layer gets its own activation function
        # if it's not specified, use the last activation function specified for the hidden layers
        if output_activation_function is None:
            self.output_activation_function = activation_functions[-1]
        else:
            self.output_activation_function = output_activation_function

    def fit(self, X, y):
        # set input and output layer
        # the length of the input layer is the number of columns in X
        self.input_layer = NeuralNetworkLayer(size=X.shape[1])
        # the length of the output layer is the number of unique values in y
        self.output_layer = NeuralNetworkLayer(size=len(np.unique(y)),
                                               activation_function=self.output_activation_function)
        # add the input and output layer to the list of layers
        self.layers.insert(0, self.input_layer)
        self.layers.append(self.output_layer)
        # initialize weights
        self.initialize_weights()
        # TODO: go on with training

    def initialize_weights(self):
        for i in range(len(self.layers) - 1):
            # the first dimension is the number of neurons in the left layer
            # the second dimension is the number of neurons in the right layer
            left = self.layers[i]
            right = self.layers[i + 1]
            # initialize weights randomly
            if self.weights_init == "gaussian":
                # use a gaussian distribution with mean 0 and standard deviation 1
                self.weights.append(np.random.normal(loc=0, scale=1, size=(left.values.shape[0], right.values.shape[0])))
            elif self.weights_init == "uniform":
                # use a uniform distribution between -1 and 1
                self.weights.append(np.random.uniform(low=-1, high=1, size=(left.values.shape[0], right.values.shape[0])))


# Single Neural Network Layer
# this class saves the current values in the layer and the activation function
# the weights are saved in the NeuralNetworkClassifier
class NeuralNetworkLayer:
    def __init_subclass__(self):
        self.values = None
        self.activation_function = None

    # initialize layers with size and activation function
    def __init__(self, *, size: int, activation_function: activationFunctions.BinaryStep(), values: np.array = None):

        self.activation_function = activation_function
        if values is not None:
            self.set_values(values)
        else:
            self.values = np.zeros(size)

    def set_values(self, values: np.array):
        if not check_type(values, int) and not check_type(values, float):
            raise ValueError("Input must be numeric")
        self.values = values

# so hats mit MLP für SK Learn ausgeschaut
# bad example
# this is the model with the highest recall
# obviously it go there by just saying everyone has muscular dystrophy lol
# x_train = x_biomed_train_preprocessed
# y_train = y_biomed_train
# # make a model
# mlp = MLPClassifier(hidden_layer_sizes = (4),
#                     max_iter=300, # epochs
#                     activation = "logistic",
#                     solver="sgd",
#                     learning_rate = "invscaling",
#                     random_state=1234 # kind of seed
#                    )
#
# print("f1:", cross_val_score(mlp, x_train, y_train, cv=10, scoring="f1_weighted").mean())
# print("recall", cross_val_score(mlp, x_train, y_train, cv=10, scoring="recall").mean())
#
# mlp.fit(x_train, y_train)
#
# pd.crosstab(y_biomed_test, mlp.predict(x_biomed_test_preprocessed), colnames=["predicted"], rownames=["actual"])
