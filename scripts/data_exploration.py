'''
data_exploration.py

January 2, 2022

this script does basic data exploration prior to modelling

'''
import pandas as pd, seaborn as sns, matplotlib.pyplot as plt

def load_data(data_type):
    features = pd.read_pickle(f'../data/{data_type}/features.pkl')
    targets = pd.read_pickle(f'../data/{data_type}/target.pkl')
    return features, targets


def visualize_numerical_data(desired_features, desired_targets):
    combined_data = pd.concat([desired_features,desired_targets],axis=1).convert_dtypes()
    plottable_features = {'suicides', 'headshots', 'scorePerMinute', 'deaths', 'percentTimeMoving', 'longestStreak', 'timePlayed', 'accuracy', 'kdRatio', 'kills', 'assists', 'result'}
    plottable_data = combined_data[plottable_features]
    grid = sns.pairplot(plottable_data, corner=True, hue='result')



def main():
    features,targets = load_data(data_type='training')


if __name__ == '__main__':
    main()