from sklearn.model_selection import cross_validate
import pandas as pd



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
    cv_results = cross_validate(model, X, y, cv=k, scoring=scoring, return_train_score=True)
    # to pandas df
    cv_results = pd.DataFrame(cv_results)
    if mean:
        cv_results = cv_results.mean()
    return cv_results
