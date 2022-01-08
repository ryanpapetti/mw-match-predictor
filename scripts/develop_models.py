'''
develop_models.py

January 3, 2022

The purpose of this script is to train and validate several models on existing data to prepare for deployment/testing 

For demonstration, I will tune one model via standard trial and error (with CV) and one with Grid Search. I hypothesize grid search will get a better score now but not with generalized data

'''

#we will do SVM (SVC), Logistic Regression, Decision Tree, Random Forest, and ensembling them all!

import pandas as pd
import joblib, random, os
random.seed(420)

from sklearn.svm import SVC

from sklearn.linear_model import LogisticRegression

from sklearn.neighbors import KNeighborsClassifier #think this will be the worst

from sklearn.naive_bayes import GaussianNB

from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier ,RandomForestClassifier #think these will work best

from sklearn.model_selection import cross_validate,RandomizedSearchCV #for quicker parameter searching / comparison

from sklearn.metrics import precision_recall_fscore_support

from sklearn.preprocessing import MinMaxScaler

from data_exploration import load_data


def initialize_models():
    model_dict = {'SVC':SVC(), 
    'LogisticRegression':LogisticRegression(),
    'KNeighborsClassifier':KNeighborsClassifier(n_neighbors=7), #
    'GaussianNB':GaussianNB(),
    'DecisionTreeClassifier': DecisionTreeClassifier(max_depth = 7), #manually keeping some models smaller to avoid overfitting
    'GradientBoostingClassifier':GradientBoostingClassifier(),
    'RandomForestClassifier':RandomForestClassifier(max_depth = 5)}

    return model_dict



def onehotencode_features(features):
    nonencodable_features = features.drop(['map', 'mode'], axis=1)
    encodable_features = features[['map','mode']]
    encoded_features = pd.get_dummies(encodable_features).drop(['map', 'mode'], axis=1)
    return pd.concat([nonencodable_features,encoded_features],axis=1)
    




def optional_scale(model_type,scalable_data, prefit_scaler = None):
    features,targets = scalable_data
    models_need_scaling = ['SVC', 'LogisticRegression', 'KNeighborsClassifier']
    if model_type in models_need_scaling:
        if prefit_scaler is None:
            scaler_filename = '../data/training/training_data_scaler.sav'
            if os.path.exists(scaler_filename):
                feature_scaler = joblib.load('../data/training/training_data_scaler.sav')
                new_features = feature_scaler.transform(features)

            else:
                new_features, targets = scale_training_data(scalable_data)
        
        else:
            feature_scaler = prefit_scaler
            new_features = feature_scaler.transform(features)

        return new_features, targets
    return scalable_data




def scale_training_data(training_data):
    features, targets = training_data
    feature_scaler = MinMaxScaler()
    new_features = feature_scaler.fit_transform(features)
    #due to needing to also scale the test and validation data APPROPRIATELY, I am going to save the scaler
    scaler_filename = '../data/training/training_data_scaler.sav'

    joblib.dump(feature_scaler,scaler_filename)
    return new_features, targets




def scale_non_training_data(model_type, non_training_data):
    features, labels = non_training_data
    training_scaler = joblib.load('../data/training/training_data_scaler.sav')
    return training_scaler.transform(features), labels



def prepare_validation_test_data(model_type,non_training_data):
    features, labels = non_training_data
    encoded_features = onehotencode_features(features)
    scalable_data = encoded_features, labels
    training_scaler = joblib.load('../data/training/training_data_scaler.sav')
    return optional_scale(model_type=model_type, scalable_data=scalable_data, prefit_scaler = training_scaler)



def prepare_training_data(model_type, training_data):
    features, labels = training_data
    encoded_features = onehotencode_features(features)
    new_training_data = encoded_features, labels
    final_training_data = optional_scale(model_type=model_type, scalable_data = new_training_data)
    return final_training_data




def train_via_cross_validation(model, training_data):
    cv_results = cross_validate(model, *training_data, return_estimator=True) #default 5 fold validation
    return cv_results


def select_best_model_from_cross_validation(cv_results):
    #get the estimator at the index of the max score in test score
    return cv_results['estimator'][cv_results['test_score'].argmax()]



def train_via_grid_search(model,distributions_to_search,iterations_desired, training_data):
    classifier = RandomizedSearchCV(model, distributions_to_search,n_iter=iterations_desired, verbose=2)
    search_results = classifier.fit(*training_data)
    return search_results.best_estimator_



def train_all_models(model_dict, training_data):
    fitted_models = {}
    for model_type, model in model_dict.items():
        prepped_training_data = prepare_training_data(model_type=model_type, training_data = training_data) #this encodes and scales the data if necessary

        #train model with cross validation
        cross_validation_results = train_via_cross_validation(model,prepped_training_data)

        #find best model
        best_model = select_best_model_from_cross_validation(cross_validation_results)

        #add model to new_dict
        fitted_models[model_type] = best_model
    return fitted_models




def score_validate_model(fitted_model, validation_data):
    features, true_labels = validation_data
    predicted_labels = fitted_model.predict(features)
    precision, recall, fscore = precision_recall_fscore_support(true_labels, predicted_labels)[:3]
    return dict(precision=precision, recall=recall, fscore=fscore)



def rank_save_best_models(fitted_models, validation_data):
    validation_scores = {}
    for model_name, model in fitted_models.items():
        prepped_validation_data = prepare_validation_test_data(model_name,validation_data)
        validation_scores[model_name] =  score_validate_model(model,prepped_validation_data) 

    pd.DataFrame(validation_scores).T.to_json('../results/validation_model_scores.json')

    




def main():
    model_dict = initialize_models()
    training_data = load_data('training')
    trained_models = train_all_models(model_dict,training_data)
    
    validation_data = load_data('validation')
    
    rank_save_best_models(trained_models,validation_data)



if __name__ == '__main__':
    main()