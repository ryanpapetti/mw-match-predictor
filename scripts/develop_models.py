'''
develop_models.py

January 3, 2022

The purpose of this script is to train and validate several models on existing data to prepare for deployment/testing 

'''

#we will do SVM (SVC), Logistic Regression, Decision Tree, Random Forest, and ensembling them all!

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier #think this will be the worst

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier #think these will work best

import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from data_exploration import load_data







def main():
    training_features, training_labels = load_data('training')
    validation_features, validation_labels = load_data('validation')


if __name__ == '__main__':
    main()