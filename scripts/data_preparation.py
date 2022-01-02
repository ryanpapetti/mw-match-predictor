'''
data_preparation.py

Jan 1 2022

this script downloads the user's match data from DynamoDb and sets it up for eventual exploration/modelling
'''
import boto3



def authenticate_aws(local_profile_name, specific_region):

    return boto3.Session(profile_name=local_profile_name, region_name=specific_region)


def retrieve_all_data_dynamo(session, match_data_table, optional_dynamo_params={}):
    dynamodb = session.client('dynamodb')

    try:
        assert match_data_table in dynamodb.list_tables()['TableNames']
        return dynamodb.scan(TableName=match_data_table, **optional_dynamo_params)
    except AssertionError:
        raise AssertionError(f"The table({match_data_table}) was not found in {dynamodb.list_tables()['TableNames']}")



def filter_relevant_data(retrieved_data):
    numerical_features = {'suicides', 'damageDone', 'headshots', 'totalXp', 'scorePerMinute', 'score', 'shotsMissed', 'deaths', 'shotsFired', 'percentTimeMoving', 'longestStreak', 'damageTaken', 'utcEndSeconds', 'timePlayed', 'accuracy', 'kdRatio', 'kills', 'utcStartSeconds', 'executions', 'nearmisses', 'medalXp', 'matchXp', 'distanceTraveled', 'duration', 'wallBangs', 'shotsLanded', 'averageSpeedDuringMatch', 'miscXp', 'scoreXp', 'assists'}

    important_string_features = {'matchID', 'gameType', 'mode', 'result', 'map'}

    new_data = {}

    for match in retrieved_data:
        numerical_match_data = {feature:float(list(match[feature].values())[0]) for feature in numerical_features}
        match_demographic_data = {feature:list(match[feature].values())[0] for feature in important_string_features}
        numerical_match_data.update(match_demographic_data)
        match_id = numerical_match_data.pop('matchID')
        new_data[match_id] = numerical_match_data
    
    return new_data



    
def split_training_validation_test(filtered_data, training_size, validation_size, test_size):
    pass






def main():
    pass

if __name__ == '__main__':
    main()






