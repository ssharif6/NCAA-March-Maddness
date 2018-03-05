import mm_predictor # stat-analysis library
from sklearn import cross_validation, linear_model
import csv
import random
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from tpot import TPOTClassifier

# intialize stat & elo dictionaries
mm_predictor.init()

# Load data
season_data = pd.read_csv('./ncaa-data/RegularSeasonDetailedResults.csv')
tourney_data = pd.read_csv('./ncaa-data/NCAATourneyDetailedResults.csv')
tourney_data = tourney_data[tourney_data.Season != 2017]

aggregated_data = pd.concat([season_data, tourney_data])

X,Y = mm_predictor.analyze_teams(aggregated_data)


print("Fitting on " + str(len(X)) + " samples")

# TODO Use TPOT or xgboost
# model = XGBClassifier()
# model = linear_model.LogisticRegression()
tpot = TPOTClassifier(generations = 100, max_time_mins=780, verbosity=2, scoring = 'neg_log_loss', population_size=30)
tpot.fit(np.array(X), np.array(Y))
tpot.export('tpot_secondTry_pipeline.py')
# Use Gridsearch here


# Check accuracy.
# print("Doing cross-validation.")
# print(cross_validation.cross_val_score(
#     model, numpy.array(X), numpy.array(Y), cv=10, scoring='neg_log_loss', n_jobs=-1
# ).mean())


# model.fit(X,Y)

# TODO Predict probability here