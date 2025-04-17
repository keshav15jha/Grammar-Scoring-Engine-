import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

class GrammarScoringModel:
    def __init__(self, model=None):
        self.model = model if model else RandomForestRegressor(random_state=42)
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        # Scale the features
        X_train_scaled = self.scaler.fit_transform(X_train)
        self.model.fit(X_train_scaled, y_train)

    def predict(self, X_test):
        # Scale the test features
        X_test_scaled = self.scaler.transform(X_test)
        return self.model.predict(X_test_scaled)

    def evaluate(self, X_test, y_test):
        predictions = self.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        return rmse
