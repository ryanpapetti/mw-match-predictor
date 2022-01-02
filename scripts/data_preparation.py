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
    pass







def main():
    pass

if __name__ == '__main__':
    main()






