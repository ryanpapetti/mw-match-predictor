'''
data_preparation.py

Jan 1 2022

this script downloads the user's match data from DynamoDb and sets it up for eventual exploration/modelling
'''
import boto3, pandas as pd, os, shutil
from sklearn.model_selection import train_test_split
from boto3.dynamodb.conditions import Attr



def authenticate_aws(local_profile_name, specific_region):

    return boto3.Session(profile_name=local_profile_name, region_name=specific_region)


def retrieve_all_data_dynamo(session, match_data_table, optional_dynamo_params={}):
    dynamodb = session.client('dynamodb')

    try:
        assert match_data_table in dynamodb.list_tables()['TableNames']
        return dynamodb.scan(TableName=match_data_table, **optional_dynamo_params)['Items']
    except AssertionError:
        raise AssertionError(f"The table({match_data_table}) was not found in {dynamodb.list_tables()['TableNames']}")



def filter_relevant_data(retrieved_data):
    numerical_features = {'suicides', 'headshots', 'scorePerMinute', 'deaths', 'percentTimeMoving', 'longestStreak', 'timePlayed', 'accuracy', 'kdRatio', 'kills', 'assists'}

    important_string_features = {'matchID', 'mode', 'result', 'map'}

    new_data = {}

    for match in retrieved_data:
        relevant_numerical_features = set(match.keys()) & numerical_features
        numerical_match_data = {feature:float(list(match[feature].values())[0]) for feature in relevant_numerical_features}
        match_demographic_data = {feature:list(match[feature].values())[0] for feature in important_string_features}
        numerical_match_data.update(match_demographic_data)
        match_id = numerical_match_data.pop('matchID')
        new_data[match_id] = numerical_match_data
    filtered_data = pd.DataFrame(new_data).T #this puts the match id as the index
    filtered_data.dropna(inplace=True)
    filtered_data = filtered_data[filtered_data['accuracy'] < 1.0] #get rid of "perfect accuracy" and no movement games
    filtered_data = filtered_data[filtered_data['percentTimeMoving'] > 20]

    #update target to only be 0/1
    filtered_data['result'] = (filtered_data['result'] == "win").astype(int) 

    #map must be in the hardcore shipment/shoot house options

    filtered_data = filtered_data[filtered_data['map'].isin(["mp_shipment","mp_m_speed"])]

    #get only hardcore tdm
    is_hardcore_mode = lambda mode: mode.endswith('war_hc')
    filtered_data = filtered_data[filtered_data['mode'].apply(is_hardcore_mode)]


    match_results = filtered_data.pop("result")
    return filtered_data, match_results



    
def split_save_training_validation_test_data(filtered_data, match_results, training_size, validation_size):
    X_train, X_validtest, y_train, y_validtest = train_test_split(filtered_data, match_results, train_size = training_size)
    X_valid, X_test, y_valid, y_test = train_test_split(X_validtest, y_validtest, train_size=validation_size)
    training_data = (X_train,y_train)
    validation_data = (X_valid, y_valid)
    testing_data = (X_test,y_test)
    relevant_data_in_order = [training_data,validation_data,testing_data]
    #now make a directory within 'data' for training and test data
    for data_type, relevant_data in zip(['training', 'validation', 'test'], relevant_data_in_order):
        if data_type in os.listdir('../data'):
            #remove dir
            shutil.rmtree(f'../data/{data_type}')
        os.mkdir(f'../data/{data_type}')
        features, target = relevant_data
        features.to_pickle(f"../data/{data_type}/features.pkl")
        target.to_pickle(f"../data/{data_type}/target.pkl")
        
    return True







def main():
    desired_profile = 'ryanschool'
    desired_region = 'us-west-2'
    authenticated_session = authenticate_aws(desired_profile,desired_region)
    # special_dynamo_filtering = {'FilterExpression': Attr("isPresentatEnd").eq('true')}
    scanned_data = retrieve_all_data_dynamo(authenticated_session, 'MWMatchData')
    filtered_data, match_results = filter_relevant_data(scanned_data)
    success_status = split_save_training_validation_test_data(filtered_data,match_results,training_size=0.6, validation_size=0.75)
    assert success_status

if __name__ == '__main__':
    main()






