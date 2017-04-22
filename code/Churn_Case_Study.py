import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, recall_score, precision_score, roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import GradientBoostingClassifier

def get_data(file_path):
    df = pd.read_csv(file_path, parse_dates=['last_trip_date', 'signup_date'])
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df['churn'] = (df.last_trip_date < pd.to_datetime('2014-06-01')).astype(int)
    return df

def get_features(df):
    city_dummies = pd.get_dummies(df.city).iloc[:,1:]
    df_new = pd.concat([df, city_dummies], axis=1)
    return df_new

def split_data(df_new):
    feature_cols = ['avg_dist','trips_in_first_30_days',"King's Landing",'Winterfell','luxury_car_user', 'weekday_pct',\
                    'luxury_car_user','surge_pct']
    X = df_new[feature_cols]
    y = df.churn
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)
    return X_train, X_test, y_train, y_test

def run_model(model, X_train, y_train):
    cv_score = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy').mean()
    return cv_score

def test_model(model, X_test, y_test):
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, model.predict(X_test))
    recall = recall_score(y_test, model.predict(X_test))
    precision = precision_score(y_test, model.predict(X_test))
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    return accuracy, recall, precision, roc_auc, y_pred_proba

def roc_curve_plot(y_test, y_pred_proba):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.plot(fpr, tpr)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('Roc Curve')
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.show()

if __name__ == '__main__':
    file_path = '../data/churn.csv'
    df = get_data(file_path)
    df_new = get_features(df)
    X_train, X_test, y_train, y_test = split_data(df_new)
    model = GradientBoostingClassifier(n_estimators=500, max_depth=8, subsample=0.5,
                                 max_features='auto', learning_rate=0.05)
    cv_score = run_model(model, X_train, y_train)
    print 'Cross_val_Score_training =', cv_score
    model.fit(X_train, y_train)
    accuracy_score, recall_score, precision_score, roc_auc_score, y_pred_proba = test_model(model, X_test, y_test)
    print 'Accuracy Score = {:.2f}\nRecall Score = {:.2f}\nPrecision Score = {:.2f}\nAuc Score = {:.2f}'.format(accuracy_score, recall_score, precision_score, roc_auc_score)
    roc_curve_plot(y_test, y_pred_proba)
