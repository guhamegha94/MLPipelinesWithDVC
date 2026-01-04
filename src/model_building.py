import numpy as np
import pandas as pd
import pickle

import xgboost as xgb

train_data = pd.read_csv('./data/features/train_bow.csv')

X_train = train_data.iloc[:,0:-1].values
y_train = train_data.iloc[:,-1].values

# Define and train the model
model = xgb.XGBClassifier(eval_metric='logloss',n_estimators=50, learning_rate=0.1, max_depth=5)
model.fit(X_train, y_train)

# Save the trained model
pickle.dump(model, open('xgb_model.pkl','wb'))