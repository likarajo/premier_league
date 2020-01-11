import os.path
from glob import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import os.path
import pickle
import time
import datetime as dt

######################################################################

DATA_DIR = 'data/'
DATA_COMBINED = os.path.join(DATA_DIR, 'combined/combined_data.csv')
DATA_FILES = glob(os.path.join(DATA_DIR, '*.csv'))
DATA_FILE_CURR = DATA_FILES[-1]
SAVE_DIR = 'models'
if not os.path.isdir(SAVE_DIR):
    os.makedirs(SAVE_DIR)

REQ = ['HomeTeam','AwayTeam','HTHG','HTAG','HTR','FTHG','FTAG','FTR',
       'HS','HST','AST','HC','AC','HF','AF','HY','AY','HR','AR']

HOME_TEAM = 'leicester'
AWAY_TEAM = 'arsenal'

######################################################################

def load_data(file=None):
    if file:
        data = pd.read_csv(file, usecols=REQ)
        data = data[pd.notnull(data['HomeTeam'])]
        return data
    datasets = []
    for f in DATA_FILES:
        d = pd.read_csv(f, usecols=REQ)
        d = d[pd.notnull(d['HomeTeam'])]
        datasets.append(d)
    datasets = pd.concat(datasets, sort=False)
    return datasets

def get_teams():
    df = load_data()
    teams = df['HomeTeam'].values.tolist() + df['AwayTeam'].values.tolist()
    return list(set(teams))

def get_index(teams, tag):
    tag = tag.title()
    indexes = [idx for idx,team in enumerate(teams) if team==tag]
    return indexes

def make_numeric(df):
    def convert(v):
        return numeric[v]
    for col in df.columns.values:
        numeric = {}
        if df[col].dtype != np.int64 and df[col].dtype != np.float64:
            vals = set(df[col].values.tolist())
            x = 0
            for v in vals:
                if v not in numeric:
                    numeric[v] = x
                    x += 1
            df[col] = list(map(convert, df[col]))
    return df

def make_features(home, away):
    df = load_data()
    df.drop(['FTR'], axis=1, inplace=True)
    home_teams = df['HomeTeam'].values
    away_teams = df['AwayTeam'].values
    df = make_numeric(df)
    home_index = get_index(home_teams.tolist(), home)
    away_index = get_index(away_teams.tolist(), away)
    home_data = df.values[home_index]
    away_data = df.values[away_index]
    home_data = np.average(home_data, axis=0)
    away_data = np.average(away_data, axis=0)
    return home_data, away_data 

def preprocess(file=None, test_size=None, train_size=None, saveCsv=False):
    data = load_data(file)
    X = data.drop(['FTR'], axis=1)
    X = make_numeric(X)
    X.fillna(0, inplace=True)
    if saveCsv:
        X.to_csv(DATA_COMBINED)
    y = data['FTR']
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                       test_size=test_size,
                                                       train_size=train_size,
                                                       random_state=42)
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    return X_train, X_test, y_train, y_test

def train(clf, X_train, y_train):
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    print('Training time {:.04f} secs'.format(end - start))
    return clf

def test(clf, X_test, y_test):
    accuracy = clf.score(X_test, y_test)
    return accuracy

def predict(clf, X):
    prediction = clf.predict(X)
    return prediction
    
def train_predict(clf, X_train, y_train, X):
    train(clf, X_train, y_train)
    return predict(clf, X)

def save_model(clf):
    now = dt.datetime.now()
    now = str(now.year)+'-'+str(now.month)+'-'+str(now.day)+'_'+str(now.hour)+':'+str(now.minute)
    name = str(clf)
    name = name[:name.index('(')]
    name = name+'_'+now
    file = os.path.join(SAVE_DIR, '{}.pkl'.format(name))
    f = open(file, 'wb')
    pickle.dump(clf, f)
    f.close()
    print('Model saved:', file)

######################################################################

def main():
    from sklearn.ensemble import AdaBoostClassifier
    
    all_teams = get_teams()
    i=1
    print('Teams:')
    for t in all_teams:
        print(i, t)
        i+=1
        
    X_train, X_test, y_train, y_test = preprocess(file=None, 
                                                 test_size=0.2,
                                                 saveCsv=True)
    print('Training:', X_train.shape, y_train.shape)
    print('Test:', X_test.shape, y_test.shape)
    
    home_team = HOME_TEAM
    away_team = AWAY_TEAM
    
    X = make_features(home=home_team, away=away_team)
    print('Predicting for {} vs {}'.format(home_team, away_team))
    
    try:
        clf = AdaBoostClassifier(n_estimators=500, learning_rate=1e-2)
        train(clf, X_train, y_train)
        accuracy = test(clf, X_test, y_test)
        print('Accuracy = {:.02%}'.format(accuracy))
        prediction = predict(clf, X)
        print('Prediction:', prediction)
        save_model(clf)
    except Exception as e:
        import sys
        sys.stderr.write(str(e))
        sys.stderr.flush()

######################################################################

if __name__ == '__main__':
    main()
