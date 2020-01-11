# Premier League

The Premier League, often referred to as the ***English Premier League*** or the EPL outside England, is the top level of the English football league system. It is Contested by 20 clubs every season.

## Goal

To predict the result of every fixture

## Dependencies

* Glob
* Pandas
* Seaborn
* Scikit-learn

`pip install -r requirments.txt`

## Dataset

* The dataset is taken from https://www.football-data.co.uk/englandm.php
  * Contains data for last 10 seasons of English Premier League, along with the data from the current season.

* Details: https://www.football-data.co.uk/notes.txt

* Fixture is taken from : https://fixturedownload.com/results/epl-2019

## Import Dataset

* Read all the data files from previous seasons
* Read only required columns
  * Detailed description: https://www.football-data.co.uk/notes.txt

```
* Date = Match Date (dd/mm/yy)
* HomeTeam = Home Team
* AwayTeam = Away Team
* HTHG = Half Time Home Team Goals
* HTAG = Half Time Away Team Goals
* HTR = Half Time Result (H=Home Win, D=Draw, A=Away Win)
* FTHG = Full Time Home Team Goals
* FTAG = Full Time Away Team Goals
* FTR = Full Time Result (H=Home Win, D=Draw, A=Away Win)
* HS = Home Team Shots
* AS = Away Team Shots
* HST = Home Team Shots on Target
* AST = Away Team Shots on Target
* HC = Home Team Corners
* AC = Away Team Corners
* HF = Home Team Fouls Committed
* AF = Away Team Fouls Committed
* HY = Home Team Yellow Cards
* AY = Away Team Yellow Cards
* HR = Home Team Red Cards
* AR = Away Team Red Cards
```

* Concatenate the data from the files together
* Import the concatenated data into a data frame

### Obtain team names

* Read the current season's data file
* Obtain the list of home teams from the records
* Find the unique teams from the home teams list

---

## Data Analysis

[data_analysis.ipynb](data_analysis.ipynb)

### Home Team Win Rate

* Total matches: 7430
* Total Home team wins: 3449
* Home wins rate: 46.42 %

### Team Statistics

* Games
* Wins
* Win Rate
* Losses
* Draws  
* Goals scored
* Scoring rate
* Goals conceded
* Shots
* Shots on Target
* Shots against
* Fouls
* Yellow Cards
* Red Cards

[team_stats.csv](data/team_stats.csv)

---

## Data Preprocessing

### Keep records of only the teams in the current season

### Prepare features and labels

* Features: all columns except 'Date', 'HTR', 'FTR'

```
18 Features
HomeTeam
AwayTeam
HTHG
HTAG
FTHG
FTAG
HS
AS
HST
AST
HC
AC
HF
AF
HY
AY
HR
AR
```

* Labels = 'FTR'

### Scale and standardise the features

* Center to the mean and component wise scale to unit variance.

### Handle categorical values

* Input data needs to be continous variables that are integers
* Convert to dummy variables

### Split data to Training and Test sets

* Test data: 50

## Create models

Classifiers:

* Logistic Regression
* Support Vector Classifier
* K-Nearest Neighbors

## Train and evaluate the models

* Train the model
* Test based on the F1 score and Accuarcy
  * F1 score considers both the precision and the recall of the test to compute the score
  * The F1 score can be interpreted as a weighted average of the precision and recall
  * F1 score reaches its best value at 1 and worst at 0.
  * Accuracy is the ratio of correct predictions to the total predictions

## Use the best model for making predictions

* Set the model
* Train the model with training dataset
* Make predictions
* Predict the probability of results (Away team win, draw, Home team win)

## Incorporate the result probabilities into the fixture

* Import the fixture
* Set up dataframe to include the fixture along with the features = Test data
* Preprocess the set
* Make predictions






