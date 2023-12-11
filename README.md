# Power-Outages

By Zhiheng Wu (zhw010@ucds.edu)

This assignment accepts some great advice from ChatGPT, and some operations refer to the example provided by the course.

## Firstly clean the data like we did before in project03

file_path = '/Users/zhihengwu/zw/Academics/dsc80-fall2023/dsc80-2023-fa/projects/03-topic/outage.xlsx'

data = pd.read_excel(file_path, skiprows=5, usecols='B:BE')[1:].drop('OBS', axis = 1)
data_cleaned = data.copy()

data_cleaned['YEAR'] = pd.to_numeric(data_cleaned['YEAR'], errors='coerce').astype('Int64')

data_cleaned['MONTH'] = pd.to_numeric(data_cleaned['MONTH'], errors='coerce').astype('Int64')

data_cleaned = data_cleaned.drop(data_cleaned.index[0])

data_cleaned.reset_index(drop=True, inplace=True)

missing_data = data_cleaned.isnull().sum()

data_cleaned

## Framing the Problem

Power outages are terrible to most residents, because it severely impact the normal process of daily work and life.

It can be measured precisely bu can be never controlled or avoided.

Therefore, it is important to know some patterns between the external factors and the characteristics of outages.

Selecting two variables among the dataframe, we find one question most intriguing:

How do different climates affect the length of outages?

If we can have some insights with this question, then it will be more comfortable to deal with the situation.

To better explain variation in outage duration, we can select R^2 as the metric for evaluation.

## Baseline Model

from sklearn.preprocessing import OneHotEncoder

from sklearn.pipeline import Pipeline

from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

mean_duration_by_state = data_cleaned.groupby('CLIMATE.REGION')['OUTAGE.DURATION'].mean()

data_cleaned['OUTAGE.DURATION'].fillna(data_cleaned['CLIMATE.REGION'].map(mean_duration_by_state), inplace=True)

data_cleaned['CLIMATE.REGION'] = data_cleaned['CLIMATE.REGION'].astype(int)

X_train = data_cleaned[['NERC.REGION', 'CAUSE.CATEGORY', 'OUTAGE.DURATION']]

y_train = data_cleaned['CLIMATE.REGION']

X_test = data_cleaned[['NERC.REGION', 'CAUSE.CATEGORY', 'OUTAGE.DURATION']]

y_test = data_cleaned['CLIMATE.REGION']

pipeline = Pipeline([

    ('preprocessing', ColumnTransformer([

        ('onehot', OneHotEncoder(), ['NERC.REGION', 'CAUSE.CATEGORY'])

    ], remainder='passthrough')),

    ('model', LogisticRegression())

])

pipeline.fit(X_train, y_train)

predictions = pipeline.predict(X_test)

accuracy_train = pipeline.score(X_train, y_train)

print("Training Accuracy:", accuracy_train)

print("Testing Accuracy:", pipeline.score(X_test, y_test))



The model achieved a training accuracy of 93.71% and a testing accuracy of 91.16%. 

The precision and recall scores for the model are 0.661 and 0.672, respectively.


## Final Model



## Fairness Analysis