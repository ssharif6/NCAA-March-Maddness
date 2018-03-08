import mm_predictor # stat-analysis library
from sklearn import cross_validation, linear_model
import csv
import random
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from tpot import TPOTClassifier
from sklearn.metrics import make_scorer

def getWinnersList(tourney_data):
    winners = []

    tourney_data_grouped = tourney_data[4:].groupby('WTeamID').size().reset_index(name='NumWins')
    for index, row in tourney_data_grouped.iterrows():
        teamName = team_id_map[row['WTeamID']]
        wins = row['NumWins']
        if len(winners) == 0:
            winners.append([])
        winners[0].append(teamName)
        if wins > 1:
            if len(winners) == 1:
                winners.append([])
            winners[1].append(teamName)
        if wins > 2:
            if len(winners) == 2:
                winners.append([])
            winners[2].append(teamName)
        if wins > 3:
            if len(winners) == 3:
                winners.append([])
            winners[3].append(teamName)
        if wins > 4:
            if len(winners) == 4:
                winners.append([])
            winners[4].append(teamName)
        if wins > 5:
            if len(winners) == 5:
                winners.append([])
            winners[5].append(teamName)
        if wins > 6:
            if len(winners) == 6:
                winners.append([])
            winners[6].append(teamName)
    
    return winners


# Method that takes in season's tournament data, id to name mapping of teams
def calcBracketScore(teamTeamWinChanceMap, tourney_data):
    
    winners = getWinnersList(tourney_data)
    
    # First four rows is 'first four' and not in actual bracket
    firstFour = tourney_data[:4]
    mainTourney = tourney_data[4:]
    
    tourney_seeds = pd.read_csv('./ncaa-data/TourneySeeds.csv')
    
    tourney = [-1] * max(tourney_seeds['Team'])

    # Look at first four teams
    for index, row in firstFour.iterrows():
        index1 = int(row['WTeamID'])
        index2 = int(row['LTeamID'])
        team1Name = team_id_map[index1]
        team2Name = team_id_map[index2]

        if (team1Name in teamTeamWinChanceMap) and (team2Name in teamTeamWinChanceMap[team1Name]):
            tourney[index2] = index1
        else:
            tourney[index1] = index2

    # MAIN TOURNAMENT 
    score = 0
    for index, row in mainTourney.iterrows():
        index1 = int(row['WTeamID'])
        index2 = int(row['LTeamID'])
        while tourney[index1] > 0:
            index1 = tourney[index1]
        while tourney[index2] > 0:
            index2 = tourney[index2]
        team1Name = team_id_map[index1]
        team2Name = team_id_map[index2]

        if (team1Name in teamTeamWinChanceMap) and (team2Name in teamTeamWinChanceMap[team1Name]): # team1 would win
            tourney[index2] = index1
            tourney[index1] = tourney[index1] - 1
            if team1Name in winners[abs(tourney[index1]) - 2]:
                score += 2**(abs(tourney[index1]) - 2) * 10
                print(team1Name + ' vs ' + team2Name + ', team 1 wins')
                print('Score ' + str(2**(abs(tourney[index1]) - 2) * 10))
            else: # Delete else statemenet when done debugging
                print('Incorrect: Predicted ' + team1Name + ' vs ' + team2Name + ', team1 wins')

        else: #team2 would win
            tourney[index1] = index2
            tourney[index2] = tourney[index2] - 1
            if team2Name in winners[abs(tourney[index2]) - 2]:
                score += 2**(abs(tourney[index2]) - 2) * 10
                print(team1Name + ' vs ' + team2Name + ', team 2 wins')
                print('Score ' + str(2**(abs(tourney[index1]) - 2) * 10))
            else: # Delete else statement when done debugging
                print('Incorrect: Predicted ' + team1Name + ' vs ' + team2Name + ', team2 wins')

    return score






if __name__ == '__main__':
    # intialize stat & elo dictionaries
    mm_predictor.init()

    # Load data
    season_data = pd.read_csv('./ncaa-data/RegularSeasonDetailedResults.csv')
    tourney_data = pd.read_csv('./ncaa-data/NCAATourneyDetailedResults.csv')
    tourney_data = tourney_data[tourney_data.Season != 2017]

    aggregated_data = pd.concat([season_data, tourney_data])

    X,Y = mm_predictor.analyze_teams_diff(aggregated_data)


    print("Fitting on " + str(len(X)) + " samples")

    # TODO Use TPOT or xgboost
    # model = XGBClassifier()
    # model = linear_model.LogisticRegression()
    tpot = TPOTClassifier(generations = 50, max_time_mins=720, verbosity=2, scoring='accuracy', population_size=30)
    tpot.fit(np.array(X), np.array(Y))
    tpot.export('tpot_difference_pipeline.py')
