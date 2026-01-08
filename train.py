# train.py - Preprocess, tune and train an XGBoost model for loan eligibility
import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
from utils import build_preprocessor
import numpy as np

DATA_PATH = 'data/paisabazar_sample.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)
PIPE_PATH = os.path.join(MODEL_DIR, 'pipeline.joblib')

def load_data(path=DATA_PATH):
    df = pd.read_csv(path)
    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status'].map({'Y':1,'N':0})
    return X, y

def main():
    X, y = load_data()
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    preprocessor = build_preprocessor()
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_jobs=1)
    pipe = Pipeline(steps=[('pre', preprocessor), ('clf', xgb)])

    param_dist = {
        'clf__n_estimators': [50,100,200],
        'clf__max_depth': [3,4,6,8],
        'clf__learning_rate': [0.01,0.05,0.1],
        'clf__subsample': [0.6,0.8,1.0],
        'clf__colsample_bytree': [0.5,0.7,1.0]
    }

    rnd = RandomizedSearchCV(pipe, param_distributions=param_dist, n_iter=20, scoring='accuracy', cv=3, verbose=1, random_state=42)
    print('Starting RandomizedSearchCV...')
    rnd.fit(X_train, y_train)
    print('Best params:', rnd.best_params_)
    best = rnd.best_estimator_

    # Evaluate
    preds = best.predict(X_valid)
    print('\nValidation Accuracy:', accuracy_score(y_valid, preds))
    print('\nClassification Report:\n', classification_report(y_valid, preds))

    # Save pipeline
    joblib.dump(best, PIPE_PATH)
    print('Saved trained pipeline to', PIPE_PATH)

if __name__ == '__main__':
    main()
