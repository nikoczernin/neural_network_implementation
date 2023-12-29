import pickle
import warnings
from datetime import datetime
from pprint import pprint
import time

import numpy as np
import random
from collections import OrderedDict
import pandas as pd
from sklearn.neural_network import MLPClassifier

import parameter_range
from HELP import get_multidimensional_combinations
from test_model import test_model



# Simulated Annealing
def simulated_annealing(model, input_data, output_data, data_title=None, model_name="",
                        break_after=None, print_progress=True, skip_til=None,n_neurons_steps=5,
                        temp_schedule='geom', temperature_0=10.0, cooling_alpha=0.95, iterations=100, objective='balanced_accuracy'):
    t0=time.time()
    result_columns = ["fit_time", "test_balanced_accuracy", "test_f1_weighted",
                      "test_precision_weighted", "test_recall_weighted"]
    SA_params = ['temperature']
    param_names, param_values, param_combinations = parameter_range.get_MLP_parameters(input_data, output_data, n_neurons_steps)
    print(len(param_combinations))
    #create result dataframes
    all_results = pd.DataFrame(columns=SA_params + param_names + result_columns)
    energy_results = pd.DataFrame(columns=['iteration','temperature', f'new {objective}', f'current {objective}',f'best {objective}'])
    #define search space
    search_space = OrderedDict(zip(param_names, param_values))
    warnings.filterwarnings("ignore") # supress warnings
    # initialize random set of hyperparameters and corresponding 'energy'
    current_params = {param: random.choice(values) for param, values in search_space.items()}
    model_test_results=test_model(model, current_params, input_data, output_data)
    if model_test_results is None:
        print ('start algorithm again')
    else:
        current_energy = - model_test_results['test_' + objective]  # Minimize 'energy', so negate score if e.g. accuracy
    best_params, best_energy = current_params, current_energy
    best_model_results = model_test_results
    states_checked = {}
    states_checked[tuple(sorted(current_params.items()))] = current_energy
    #grid_scores = [(1, temperature, current_energy, current_params)]
    # create a model using these parameters
    iter_nr=1
    temperature = temperature_0
    while iter_nr <= iterations:
        ### generate new neighbouring point ###
        new_params = current_params.copy()
        # randomly choose a parameter to update
        rand_param_to_update = np.random.choice(param_names)
        print(rand_param_to_update)
        param_vals = list(search_space[rand_param_to_update])
        current_index = param_vals.index(current_params[rand_param_to_update])
        # if first of list of params, then select second param
        if current_index == 0:
            new_params[rand_param_to_update] = param_vals[1]
        # if last of list of params, then select second last param
        elif current_index == len(param_vals) - 1:
            new_params[rand_param_to_update] = param_vals[current_index - 1]
        # if neither, then select previous or next param randomly
        else:
            new_params[rand_param_to_update] = param_vals[current_index + np.random.choice([-1, 1])]
        print(f'param changed from {current_params[rand_param_to_update]} to {new_params[rand_param_to_update]}')
        ### check if state has already been checked ###
        # if yes, use the corresponding result
        try:
            new_energy = states_checked[tuple(sorted(new_params.items()))]
            print('new_params revisited')
        # if no, evaluate 'energy' of new state and save as checked state
        except:
            model_test_results=test_model(model, new_params, input_data, output_data)
            # continue if no result
            if model_test_results is None:
                print ('no result')
                continue
            else:
                new_energy = - model_test_results['test_' + objective]  # Minimize 'energy', so negate score if e.g. accuracy
            states_checked[tuple(sorted(new_params.items()))] = new_energy
            print('new_params not revisited')
            model_test_results_list = list(model_test_results[result_columns])
            # all info on this new (not revisited) paramter set
            info_full = [temperature] + list(new_params.values()) + model_test_results_list
            # add it to the all_results df
            all_results.loc[len(all_results.index)] = info_full
        # Accept the new solution if it is better or with a certain probability if it is worse
        print(f'new/current energy {new_energy}/{current_energy}')
        acceptance_criterion = np.exp((current_energy-new_energy) / temperature)
        print(f'acceptance criterion:{acceptance_criterion}')
        if new_energy < current_energy or np.random.uniform() < acceptance_criterion:
            current_params = new_params
            current_energy = new_energy
            print ('new params accepted!')
        else:
            print('new params NOT accepted!')
        if new_energy < best_energy:
            best_params, best_energy = new_params, new_energy
            best_model_results = model_test_results
            print ('new best params!')
        ### Annealing schedule ###
        if temp_schedule=='geom':
            # geometric cooling
            temperature *= cooling_alpha
        elif temp_schedule=='lin':
            # linear cooling
            cooling_beta = (temperature_0-0.001)/iterations
            temperature = temperature_0 - cooling_beta*iter_nr
        elif temp_schedule=='fast':
            # fast SA
            temperature=temperature_0/iter_nr
        energy_results.loc[len(energy_results.index)] = [iter_nr, temperature, -new_energy, -current_energy, -best_energy]
        print(f'temperature:{temperature}')
        print(f'Nr. of iterations: {iter_nr}, Current energy: {current_energy}, Best energy: {best_energy} \n')
        iter_nr+=1
    print(len(states_checked))
    print(f'Time elapsed:{time.time()-t0} seconds')
    # save the results
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    all_results.to_csv(f"data/results/SA_results_{data_title}_{model_name}_{temp_schedule}_t{temperature_0}_alpha{cooling_alpha}_iter{iterations}_{current_time}.csv")
    energy_results.to_csv(f"data/results/SA_energy_results_{data_title}_{model_name}_{temp_schedule}_t{temperature_0}_alpha{cooling_alpha}_iter{iterations}_{current_time}.csv")    
    return best_params, best_model_results, energy_results

data_title = "wine"
objective = "balanced_accuracy"
temp_schedule = 'geom' # geom, lin, fast
temperature_0 = 1.0
cooling_alpha = 0.95
iterations = 100
with open(f"data/{data_title}/{data_title}_x.pickle", "rb") as f:
    input_data = pickle.load(f)
with open(f"data/{data_title}/{data_title}_y.pickle", "rb") as f:
    output_data = pickle.load(f)

model = MLPClassifier
print("starting annealing")
best_params,best_model_results, energy_results =simulated_annealing(model, input_data, output_data, data_title=data_title, model_name="MLP",
                        break_after=None, print_progress=True, skip_til=None,n_neurons_steps=5,
                        temp_schedule=temp_schedule, temperature_0=temperature_0, cooling_alpha=cooling_alpha, iterations=iterations, objective=objective)
print(best_params)
#%%
ax = energy_results.plot(y=[f'new {objective}', f'current {objective}',f'best {objective}'], use_index=True)
ax.set_xlabel("iteration number")
ax.set_ylabel(objective)
ax.legend(loc='lower right')
ax.figure.savefig(f"data/results/SA_{data_title}_{temp_schedule}_t{temperature_0}_alpha{cooling_alpha}_iter{iterations}.png")