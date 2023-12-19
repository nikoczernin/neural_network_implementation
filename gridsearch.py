import pickle
import warnings
from datetime import datetime
from pprint import pprint

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier

import parameter_range
from HELP import get_multidimensional_combinations
from test_model import test_model






def grid_search(model, input_data, output_data, data_title=None, model_name="",
                break_after=None, print_progress=True, skip_til=None,
                n_neurons_steps=5):
    # keep the following scores:
    # fit_time
    # test_balanced_accuracy
    # test_f1_weighted
    # test_precision_weighted
    # test_recall_weighted
    result_columns = ["fit_time", "test_balanced_accuracy", "test_f1_weighted",
                      "test_precision_weighted", "test_recall_weighted"]

    param_names, parameter_values, param_combinations = parameter_range.get_MLP_parameters(input_data, output_data,
                                                                                          n_neurons_steps=n_neurons_steps)
    # pd.DataFrame with the results

    print(param_names + result_columns)
    all_results = pd.DataFrame(columns=param_names + result_columns)

    n = len(param_combinations)
    for i, param_combo_list in enumerate(param_combinations):
        if skip_til is not None:
            if i < skip_til:
                continue

        if print_progress: print(f"\r{i}/{n}", param_combo_list, end="")

        # param_combo_list is a list of args
        params_dict = {param_name: param_combo_list[i] for i, param_name in enumerate(param_names)}

        # create a model using these parameters
        warnings.filterwarnings("ignore") # supress warnings
        model_test_results = test_model(model, params_dict, input_data, output_data)
        # if no results, continue
        if model_test_results is None:
            continue
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

def main():
    with open("data/wine/wine_x.pickle", "rb") as f:
        input_data = pickle.load(f)
    with open("data/wine/wine_y.pickle", "rb") as f:
        output_data = pickle.load(f)

    data_title = "wine"
    model = MLPClassifier
    print("starting grid search")
    grid_search(model=model,
                input_data=input_data,
                output_data=output_data,
                data_title=data_title,
                model_name="MLP",
                n_neurons_steps=3,
                #break_after=10
                )

if __name__ == '__main__':
    main()