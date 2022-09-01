from audioop import avg
from unicodedata import name
import pandas as pd
import numpy as np


titanic_data = pd.read_csv("titanic dataset/train.csv")
titanic_data.head()
titanic_data.columns

subset = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch']
titanic_subset = titanic_data[subset]
titanic_subset.head()
titanic_subset.shape
titanic_subset.describe()


titanic_df = pd.DataFrame(titanic_subset)
titanic_df.head()
titanic_df.isnull().sum() ##177 missing values in Age
titanic_df.Pclass.unique() ## No inconsistency, as desired
titanic_df.Sex.unique() ## No inconsistency, as desired
titanic_df.SibSp.unique() ## No inconsistency, as desired
titanic_df.Parch.unique() ## No inconsistency, as desired

## Fill out missing ages
null_rows = np.where([titanic_df.Age.isnull()])[1] # Get rows with missing ages
filled_rows = list(set(titanic_df.index )- set(null_rows))
len(filled_rows), len(null_rows)

## Get the average ages by gender by classes
titanic_df_NoNull = titanic_df.loc[filled_rows]
avg_age_by_sex_class = titanic_df_NoNull.groupby(['Sex', 'Pclass']).Age.mean()
avg_age_by_sex_class = round(avg_age_by_sex_class, 2)
avg_age_by_sex_class
# female_1st_age, female_2nd_age, female_3rd_age = avg_age_by_sex_class.loc[('female',)]
# male_1st_age, male_2nd_age, male_3rd_age = avg_age_by_sex_class.loc[('male',)]

## Automatically estimate passenger age given class and gender
def estAge(df, row_indx):
    passenger = df.loc[row_indx]
    passenger_class = passenger.Pclass
    passenger_sex = passenger.Sex
    titanic_df.loc[row_indx, 'Age'] = avg_age_by_sex_class.loc[
        (passenger_sex, passenger_class)] # Mutate the record directly
    print("Finish filling")

titanic_df.loc[5, 'Age'] = 26.5
titanic_df.loc[5, 'Age']


## Fill out missing ages by class-gender age averages
for indx in null_rows:
    estAge(titanic_df, indx)

titanic_df.loc[null_rows[:10],] ## Filling complete
titanic_df.isnull().sum()

titanic_df.to_csv('titanic-age-filled.csv')

