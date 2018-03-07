import pandas as pd
import math
import csv
import random
import numpy

# Global Variables
NUM_GAMES = 8
TEAM_STATS = {}
YEAR = 2018
TEAM_ELOS = {}
BASE_ELO = 1600
stat_fields = ['score', 'fga', 'fgp', 'fga3', '3pp', 'ftp', 'or', 'dr', 'ast', 'to', 'stl', 'blk', 'pf']
POWER_CONFERENCES = ['acc', 'big_twelve', 'big_ten', 'sec', 'pac_twelve']

conferences = pd.read_csv('./ncaa-data/TeamConferences.csv')


def init():
    for year in range(1985, YEAR + 1):
        TEAM_STATS[year] = {}
        TEAM_ELOS[year] = {}


def get_elo(year, team):
    try:
        return TEAM_ELOS[year][team]
    except:
        try:
            # Return previous season
            TEAM_ELOS[year][team] = TEAM_ELOS[year - 1][team]
            return TEAM_ELOS[year][team]
        except:
            TEAM_ELOS[year][team] = BASE_ELO
            return TEAM_ELOS[year][team]


def get_stats(season, team, field):
    try:
        l = TEAM_STATS[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0


# Copied from https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/
def calc_elo(win_team, lose_team, season):
    winner_rank = get_elo(season, win_team)
    loser_rank = get_elo(season, lose_team)

    rank_diff = winner_rank - loser_rank
    exp = (rank_diff * -1) / 400
    odds = 1 / (1 + math.pow(10, exp))
    if winner_rank < 2100:
        k = 32
    elif 2100 <= winner_rank < 2400:
        k = 24
    else:
        k = 16
    new_winner_rank = round(winner_rank + (k * (1 - odds)))
    new_rank_diff = new_winner_rank - winner_rank
    new_loser_rank = loser_rank - new_rank_diff

    return new_winner_rank, new_loser_rank


def update_stats(season, team, fields):
    'Updates stats for a given team and season for provided field'
    if team not in TEAM_STATS[season]:
        TEAM_STATS[season][team] = {}
    for key, value in fields.items():
        # Make sure we have the field.
        if key not in TEAM_STATS[season][team]:
            TEAM_STATS[season][team][key] = []

        # Only want to traak the last 8 games of the season along with tourney results
        # Track last 8 games as it measures form going into the tournament
        if len(TEAM_STATS[season][team][key]) >= NUM_GAMES:
            TEAM_STATS[season][team][key].pop()
        TEAM_STATS[season][team][key].append(value)


def analyze_teams(agg_data):
    'Calculates elo for each game and each season via in-game stat analysis'
    stat_features = []  # X
    y = []
    # Iterate through each row
    for i, game in agg_data.iterrows():
        win_team_id = game['WTeamID']
        lose_team_id = game['LTeamID']
        curr_season = game['Season']
        # Get previous ELOs to update
        winning_team_elo = get_elo(curr_season, win_team_id)
        losing_team_elo = get_elo(curr_season, lose_team_id)
        # Part of ELO rules award 100 ELO to home team
        if game['WLoc'] == 'H':
            winning_team_elo += 100
        elif game['WLoc'] == 'A':
            losing_team_elo += 100

        # Else is neutral location, so don't add elo
        wTeam_feats = [winning_team_elo]
        lTeam_feats = [losing_team_elo]

        # BINARY value for if in power conference
        if get_is_power_conference(win_team_id, curr_season):
            wTeam_feats.append(1)
        else:
            wTeam_feats.append(0)

        if get_is_power_conference(lose_team_id, curr_season):
            lTeam_feats.append(1)
        else:
            lTeam_feats.append(0)

        validData = True
        for stat in stat_fields:
            w_stat = get_stats(game['Season'], game['WTeamID'], stat)
            l_stat = get_stats(game['Season'], game['LTeamID'], stat)
            if w_stat is not 0 and l_stat is not 0:
                wTeam_feats.append(w_stat)
                lTeam_feats.append(l_stat)
            else:
                validData = False


        # Randomly select left and right and 0 or 1 so we can train for multiple classes
        if validData:
            if random.random() > 0.5:
                stat_features.append(wTeam_feats + lTeam_feats)
                y.append(0)
            else:
                stat_features.append(lTeam_feats + wTeam_feats)
                y.append(1)

        if game['WFTA'] != 0 and game['LFTA'] != 0:
            winner_stats = {
                'score': game['WScore'],
                'fgp': game['WFGM'] / game['WFGA'] * 100,
                'fga': game['WFGA'],
                'fga3': game['WFGA3'],
                '3pp': game['WFGM3'] / game['WFGA3'] * 100,
                'ftp': game['WFTM'] / game['WFTA'] * 100,
                'or': game['WOR'],
                'dr': game['WDR'],
                'ast': game['WAst'],
                'to': game['WTO'],
                'stl': game['WStl'],
                'blk': game['WBlk'],
                'pf': game['WPF']
            }
            loser_stats = {
                'score': game['LScore'],
                'fgp': game['LFGM'] / game['LFGA'] * 100,
                'fga': game['LFGA'],
                'fga3': game['LFGA3'],
                '3pp': game['LFGM3'] / game['LFGA3'] * 100,
                'ftp': game['LFTM'] / game['LFTA'] * 100,
                'or': game['LOR'],
                'dr': game['LDR'],
                'ast': game['LAst'],
                'to': game['LTO'],
                'stl': game['LStl'],
                'blk': game['LBlk'],
                'pf': game['LPF']
            }
            update_stats(game['Season'], game['WTeamID'], winner_stats)
            update_stats(game['Season'], game['LTeamID'], loser_stats)

        # Update elo    
        new_winner_elo, new_loser_elo = calc_elo(
            game['WTeamID'], game['LTeamID'], game['Season'])
        TEAM_ELOS[game['Season']][game['WTeamID']] = new_winner_elo
        TEAM_ELOS[game['Season']][game['LTeamID']] = new_loser_elo
    return stat_features, y


def predict_winner(team_1, team_2, model, season, sf):
    features = []

    # Team 1
    features.append(get_elo(season, team_1))
    if get_is_power_conference(team_1, season):
        features.append(1)
    else:
        features.append(0)

    for stat in sf:
        features.append(get_stat(season, team_1, stat))

    # Team 2
    features.append(get_elo(season, team_2))
    if get_is_power_conference(team_2, season):
        features.append(1)
    else:
        features.append(0)

    for stat in sf:
        features.append(get_stat(season, team_2, stat))

    return model.predict_proba([features])


def get_is_power_conference(teamId, year):
    win_conf = conferences[conferences['TeamID'] == teamId]
    win_conf = win_conf[win_conf['Season'] == year]['ConfAbbrev'].tolist()[0]

    return win_conf in POWER_CONFERENCES

def get_stat(season, team, field):
    try:
        l = TEAM_STATS[season][team][field]
        return sum(l) / float(len(l))
    except:
        return 0
