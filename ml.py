from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import signal_processing
from skfeature.function.similarity_based import fisher_score
from sklearn import svm
from sklearn.metrics import accuracy_score, mean_squared_error

import shap
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn import tree

def shapley_value(x_train:np.ndarray, x_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray, X:pd.DataFrame):
    shap.initjs()
    xgb_model = XGBRegressor(n_estimators=1000, max_depth=10, learning_rate=0.001, random_state=0)
    xgb_model.fit(x_train, y_train)
    y_predict = xgb_model.predict(x_test)
    print(y_predict)
    mean_squared_error(y_test, y_predict)**(0.5)
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(x_train)
    shap.summary_plot(shap_values, features=x_train, feature_names=X.columns)
    acc = accuracy_score(y_test, y_predict)
    print(acc)

def fisher_score_show(x_train:np.ndarray, x_test:np.ndarray, y_train:np.ndarray, y_test:np.ndarray):
    score = fisher_score.fisher_score(x_train, y_train)
    idx = fisher_score.feature_ranking(score)
    num_fea = 5
    
    plt.figure(layout="constrained")
    plt.bar(range(len(score)), score)
    plt.xlabel('Feature Index')
    plt.ylabel('Fisher Score')
    plt.title('Fisher Score for Each Feature')
    for i in range(num_fea):
        plt.annotate(idx[i], 
                     xy=(idx[i], score[idx[i]]), rotation=0, xycoords='data',
                     xytext=(0,0), textcoords='offset pixels')
    plt.show()
    
    selected_feature_train = x_train[:, idx[0:num_fea]]
    selected_feature_test = x_test[:, idx[0:num_fea]]
    clf = svm.LinearSVC(dual=False)
    clf.fit(selected_feature_train, y_train)
    y_predict = clf.predict(selected_feature_test)
    acc = accuracy_score(y_test, y_predict)
    print(acc)

if __name__ == '__main__':
    X = signal_processing.read_sheets('psd.xlsx', usecols=[0,1,2,3], combine=True)
    X = X.transpose()
    y = np.array([signal_processing.class_label(sample.split(' ')[-1].split('_')[0]) for sample in X.index])
    x_train, x_test, y_train, y_test = train_test_split(X.to_numpy(), y, test_size=0.2, random_state=40)
    
    