#Phase 6: Training baseline model (XGBoost)

import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

X_train = np.load('data/final_stratified/X_train.npy')
Y_train = np.load('data/final_stratified/Y_train.npy')
X_val = np.load('data/final_stratified/X_val.npy')
Y_val = np.load('data/final_stratified/Y_val.npy')
X_test = np.load('data/final_stratified/X_test.npy')
Y_test = np.load('data/final_stratified/Y_test.npy')

#Model setup
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


#Validation
Y_val_pred = model.predict(X_val)
val_rmse = np.sqrt(mean_squared_error(Y_val, Y_val_pred))
val_r2 = r2_score(Y_val, Y_val_pred)

#Test
Y_test_pred = model.predict(X_test)
test_rmse = np.sqrt(mean_squared_error(Y_test, Y_test_pred))
test_r2 = r2_score(Y_test, Y_test_pred)

#Log results to file
import json
from datetime import datetime

results = {
    'model': 'XGBoost',
    'timestamp': datetime.now().isoformat(),
    'hyperparameters': {
        'n_estimators': 100,
        'max_depth': 6,
        'learning_rate': 0.1,
        'random_state': 42,
        'n_jobs': -1
    },
    'validation_metrics': {
        'rmse': float(val_rmse),
        'r2': float(val_r2)
    },
    'test_metrics': {
        'rmse': float(test_rmse),
        'r2': float(test_r2)
    }
}

with open('results/stratified_split/xgboost_results.json', 'w') as f:
    json.dump(results, f, indent=2)

#Save model for ensemble
pickle.dump(model, open('results/stratified_split/xgboost_model.pkl', 'wb'))