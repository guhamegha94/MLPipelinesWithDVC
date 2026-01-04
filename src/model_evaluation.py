import numpy as np
import pandas as pd

import pickle
import json

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, roc_auc_score

model = pickle.load(open('xgb_model.pkl','rb'))
test_data = pd.read_csv('./data/features/test_bow.csv')

X_test = test_data.iloc[:,0:-1].values
y_test = test_data.iloc[:,-1].values

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:,1]

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
metrics = {
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'roc_auc': roc_auc,
    'confusion_matrix': conf_matrix.tolist(),
    'classification_report': class_report
}

# Store evaluation metrics
with open('model_evaluation_metrics.json', 'w') as file:
    json.dump(metrics, file, indent=4)

