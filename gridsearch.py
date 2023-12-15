import warnings
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd

import parameter_range
from HELP import get_multidimensional_combinations
from test_model import test_model


def get_parameters(input_data, output_data):
    # get the parameter ranges according to the input and output data
    ranges = parameter_range.get_MLP_parameter_ranges(input_data, output_data)
    # convert ranges to numpy arrays of actual values for grid search
    ranges["learning_rate_init"] = np.logspace(ranges["learning_rate_init"][0],
                                               ranges["learning_rate_init"][1],
                                               4)
    ranges["alpha"] = np.logspace(ranges["alpha"][0], ranges["alpha"][1], 5)
    ranges["hidden_layer_sizes"] = parameter_range.get_architecture_suggestions(input_data, output_data)
    # names of parameters
    parameter_names = list(ranges.keys())
    # possible values of parameters (list of lists)
    parameter_values = list(ranges.values())

    # all parameter combinations
    parameter_combinations = get_multidimensional_combinations(parameter_values)
    return parameter_names, parameter_values, parameter_combinations





def grid_search(model, input_data, output_data, data_title=None, model_name="",
                break_after=None, print_progress=True):
    # keep the following scores:
    # fit_time
    # test_balanced_accuracy
    # test_f1_weighted
    # test_precision_weighted
    # test_recall_weighted
    result_columns = ["fit_time", "test_balanced_accuracy", "test_f1_weighted",
                      "test_precision_weighted", "test_recall_weighted"]

    param_names, _, param_combinations = get_parameters(input_data, output_data)

    # pd.DataFrame with the results
    print(param_names + result_columns)
    all_results = pd.DataFrame(columns=param_names + result_columns)

    print()
    n = len(param_combinations)
    for i, param_combo_list in enumerate(param_combinations):
        if print_progress: print(f"{i}/{n}", param_combo_list, end="\r")

        # param_combo_list is a list of args
        params_dict = {param_name: param_combo_list[i] for i, param_name in enumerate(param_names)}

        # create a model using these parameters
        warnings.filterwarnings("ignore") # supress warnings
        model_test_results = test_model(model, params_dict, input_data, output_data)
        model_test_results = list(model_test_results[result_columns])
        # all info on this iteration
        info_full = param_combo_list + model_test_results
        # add it to the all_results df
        all_results.loc[len(all_results.index)] = info_full

        # emergency break
        if break_after is not None:
            if i > break_after:
                break

    # save the results
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    model_name = "MLP"
    all_results.to_csv(f"data/results/grid_search_results_{data_title}_{model_name}_{current_time}.csv")