from sklearn.grid_search import GridSearchCV
from sklearn import svm
import numpy as np
import mm_predictor
import pandas as pd

mm_predictor.init()
season_data = pd.read_csv('./ncaa-data/RegularSeasonDetailedResults.csv')
tourney_data = pd.read_csv('./ncaa-data/NCAATourneyDetailedResults.csv')
tourney_data = tourney_data[tourney_data.Season != 2017]

aggregated_data = pd.concat([season_data, tourney_data])

X,Y = mm_predictor.analyze_teams_diff(aggregated_data)



parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}

svc = svm.SVC(probability=True)

clfA = GridSearchCV(svc, parameters, scoring='accuracy', n_jobs=1)

clfA.fit(np.array(X), np.array(Y))
print(clfA.best_params_)
