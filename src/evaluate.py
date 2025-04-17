import numpy as np
from sklearn.metrics import mean_squared_error

def calculate_rmse(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = calculate_rmse(y_test, y_pred)
    return rmse
