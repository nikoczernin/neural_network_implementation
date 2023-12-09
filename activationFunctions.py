# add code for all activation functions here
# main inspiration: https://www.v7labs.com/blog/neural-networks-activation-functions
# TODO: dont forget to cite it
import numpy as np
from Help import *


# Activation Function: General
# the underscore in front of the class name indicates that this class is private
# it should not be used outside this file
class _ActivationFunction:
    def transform(self, values: np.array) -> np.array:
        """
        Transformiere die Inputwerte
        :param values:
        :param weights:
        :return:
        """
        activated_values = values
        return activated_values

    def activate(self, values: np.array, weights: np.array = None) -> np.array:
        """
        :param values: 1D-array with input values
        :param weights: 1D-array with weights for each input value
        :return: single activated value
        """
        # type-checking
        if not isinstance(values, np.ndarray):
            raise ValueError("Input must be a NumPy array")

        # check if all elements are numeric
        if not check_type(values, int) and not check_type(values, float):
            raise ValueError("Input must be numeric")

        # if weights are None, create weights of 1
        if weights is None:
            weights = np.ones(values.shape)

        # check if values and weights have equal shape
        if values.shape != weights.shape:
            raise ValueError("Input and weights must have equal shape")

        # make transformation
        weighted_values = values * weights
        weighted_sum = np.sum(weighted_values)
        transformed_value = self.transform(weighted_sum)

        return transformed_value


# Binary Step Activation Function
# returns 1 if input >= 0, 0 otherwise
class BinaryStep(_ActivationFunction):
    def transform(self, values: np.array) -> np.array:
        activated_values = np.where(values >= 0, 1, 0)
        return activated_values


bin = BinaryStep()
arr = np.array([1, 2, -3, -4, 5])
print("Binary Step")
print(bin.activate(arr))


# Linear Activation Function
# returns input
class Linear(_ActivationFunction):
    pass


lin = Linear()
print("Linear")
print(lin.activate(arr))


# Sigmoid/Logistic Activation Function
# returns 1 / (1 + e^-input)
# output values are between 0 and 1
class Sigmoid(_ActivationFunction):
    def transform(self, values: np.array) -> np.array:
        activated_values = 1 / (1 + np.exp(-values))
        return activated_values


sig = Sigmoid()
print("Sigmoid")
print(sig.activate(arr))


# Hyperbolic Tangent Activation Function
# returns (e^input - e^-input) / (e^input + e^-input)
# output values are between -1 and 1
class TanH(_ActivationFunction):
    def transform(self, values: np.array) -> np.array:
        activated_values = np.tanh(values)
        return activated_values


tanh = TanH()
print("Tanh")
print(tanh.activate(arr))


# ReLU / Rectified Linear Unit Activation Function
# returns input if input >= 0, 0 otherwise
# output values are between 0 and infinity
class ReLU(_ActivationFunction):
    def transform(self, values: np.array) -> np.array:
        activated_values = np.where(values >= 0, values, 0)
        return activated_values


relu = ReLU()
print("ReLU")
print(relu.activate(arr))


# Leaky ReLU Activation Function
# returns input if input >= 0, 0.01 * input otherwise
# output values are between -infinity and infinity
class LeakyReLU(_ActivationFunction):
    def transform(self, values: np.array) -> np.array:
        activated_values = np.where(values >= 0, values, 0.01 * values)
        return activated_values


lrelu = LeakyReLU()
print("Leaky ReLU")
print(lrelu.activate(arr))

