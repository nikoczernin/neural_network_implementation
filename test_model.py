import pickle
from pprint import pprint

from sklearn.model_selection import cross_validate
import pandas as pd
from sklearn.neural_network import MLPClassifier


def test_model(Model, params, X, y, k=5, mean=True, scoring=None):
    """
    Function to test a model with cross validation
    Parameters:
        Model: sklearn model
        params: dict of parameters for the model
        X: features
        y: target
        k: number of folds
        mean: if True, return the mean of the scores
        scoring: dict of scorers to use
    """
    if scoring is None:
        scoring = {
            'balanced_accuracy': 'balanced_accuracy',
            'f1_weighted': 'f1_weighted',
            'precision_weighted': 'precision_weighted',
            'recall_weighted': 'recall_weighted'}

    model = Model(**params)
    try:
        cv_results = cross_validate(model, X, y, cv=k, scoring=scoring, return_train_score=True)
    except ValueError as e:
        print()
        print("This model failed:")
        pprint(params)
        pprint(e)
        return None
    # to pandas df
    cv_results = pd.DataFrame(cv_results)
    if mean:
        cv_results = cv_results.mean()
    return cv_results


def main():
    with open("data/biomed/biomed_x.pickle", "rb") as f:
        x = pickle.load(f)
    with open("data/biomed/biomed_y.pickle", "rb") as f:
        y = pickle.load(f)

    print(x.shape)
    print(y.shape)

    # get the parameters
    # test the model
    res = test_model(MLPClassifier,
               {'activation': 'logistic',
                'alpha': 1.0002302850208247,
                'hidden_layer_sizes': (4,),
                'learning_rate': 'constant',
                'learning_rate_init': 1.0023052380778996,
                'solver': 'lbfgs'
                },
               x, y)
    print(res)


if __name__ == "__main__":
    main()
