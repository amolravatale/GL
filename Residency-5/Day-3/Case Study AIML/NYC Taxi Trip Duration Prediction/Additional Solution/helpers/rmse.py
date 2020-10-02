import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.metrics.scorer import make_scorer

def rmse(actual, pred):
    assert len(actual) == len(pred)
    # We already performed log transformation, we will use plain mse instead
    # return np.sqrt(np.mean(np.square(np.log1p(actual) - np.log1p(pred))))
    return np.sqrt(mean_squared_error(actual, pred))

rmse_scorer = make_scorer(rmse, greater_is_better=False)