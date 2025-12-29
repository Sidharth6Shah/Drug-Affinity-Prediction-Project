#Phase 6: Training baseline model (XGBoost)

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

X_train = np.load('data/final/X_train.npy')
Y_train = np.load('data/final/Y_train.npy')
X_val = np.load('data/final/X_val.npy')
Y_val = np.load('data/final/Y_val.npy')
X_test = np.load('data/final/X_test.npy')
Y_test = np.load('data/final/Y_test.npy')

#Set up the model
model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    n_jobs=-1,
    verbosity=1
    )

#Train
model.fit(X_train, Y_train)