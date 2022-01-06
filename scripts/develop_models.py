'''
develop_models.py

January 3, 2022

The purpose of this script is to train and validate several models on existing data to prepare for deployment/testing 

For demonstration, I will tune one model via standard trial and error (with CV) and one with Grid Search. I hypothesize grid search will get a better score now but not with generalized data

'''

#we will do SVM (SVC), Logistic Regression, Decision Tree, Random Forest, and ensembling them all!

import pandas as pd




from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier #think this will be the worst

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier #think these will work best

from sklearn.model_selection import cross_validate,RandomizedSearchCV #for quicker parameter searching / comparison

from sklearn.metrics import f1_score, precision_recall_fscore_support, precision_recall_curve

from sklearn.preprocessing import MinMaxScaler

from data_exploration import load_data


def initialize_non_grid_search_models():
    model_dict = {'SVC':SVC(), 
    'LogisticRegression':LogisticRegression(),
    'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=7), #
    'GaussianNB':GaussianNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth = 7), #manually keeping some models smaller to avoid overfitting
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'RandomForestClassifier':RandomForestClassifier(max_depth = 5)}

    return model_dict



def optional_scale(model_type,training_data):
    features,targets = training_data
    models_need_scaling = ['SVC', 'LogisticRegression', 'KNeighborsClassifier']
    if model_type in models_need_scaling:
        feature_scaler = MinMaxScaler()
        new_features = feature_scaler.fit_transform(features)
        return new_features, targets
    return training_data



def train_via_cross_validation(model, training_data):
    cv_results = cross_validate(model, *training_data, scoring=f1_score) #default 5 fold validation
    return cv_results


def select_best_model_from_cross_validation(cv_results):
    #get the estimator at the index of the max score in test score
    return cv_results['estimator'][cv_results['test_score'].index(max(cv_results['test_score']))]



def train_via_grid_search(model,distributions_to_search,iterations_desired, training_data):
    classifier = RandomizedSearchCV(model, distributions_to_search, scoring=f1_score,n_iter=iterations_desired, verbose=2)
    search_results = classifier.fit(*training_data)
    return search_results.best_estimator_



def train_all_models(training_data):
    pass




def score_validate_model(fitted_model, validation_data):
    features, true_labels = validation_data
    predicted_labels = fitted_model.predict(features)
    precision, recall, fscore = precision_recall_fscore_support(true_labels, predicted_labels)[:3]
    return dict(precision=precision, recall=recall, fscore=fscore)



def rank_save_best_models(fitted_models, validation_data):
    validation_scores = pd.DataFrame({key:score_validate_model(model,validation_data) for key,model in fitted_models.items()})

    




def main():
    training_features, training_labels = load_data('training')
    validation_features, validation_labels = load_data('validation')


if __name__ == '__main__':
    main()