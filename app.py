import seaborn as sns
import numpy as np
import pandas as pd
!pip install dash
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

import plotly.graph_objects as go
import plotly.express as px

import dash
from dash import html, dcc
import dash_daq as daq
from dash.dependencies import Input, Output, State

# ********************* DATA PREPARATION *********************
# Load data
df = sns.load_dataset('titanic').drop(columns=['pclass', 'embarked', 'alive'])

# Format data for dashboard
df.columns = df.columns.str.capitalize().str.replace('_', ' ')
df.rename(columns={'Sex': 'Gender'}, inplace=True)
for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.capitalize()

# Partition into train and test splits
TARGET = 'Survived'
y = df[TARGET]
X = df.drop(columns=TARGET)

numerical = X.select_dtypes(include=['number', 'boolean']).columns
categorical = X.select_dtypes(exclude=['number', 'boolean']).columns
X[categorical] = X[categorical].astype('object')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, random_state=42, 
                                                    stratify=y)
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

# Build pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(transformers=[
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
            ('encoder', OneHotEncoder(sparse=False))
            
        ]), categorical),
        ('num', SimpleImputer(strategy='mean'), numerical)
    ])),
    ('model', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)

# Add predicted probabilities
test['Probability'] = pipeline.predict_proba(X_test)[:,1]
test['Target'] = test[TARGET]
test[TARGET] = test[TARGET].map({0: 'No', 1: 'Yes'})

labels = []
for i, x in enumerate(np.arange(0, 101, 10)):
    if i>0:
        labels.append(f"{previous_x}% to <{x}%")
    previous_x = x
test['Binned probability'] = pd.cut(test['Probability'], len(labels), labels=labels, 
                                    right=False)

# Helper functions for dropdowns and slider
def create_dropdown_options(series):
    options = [{'label': i, 'value': i} for i in series.sort_values().unique()]
    return options
def create_dropdown_value(series):
    value = series.sort_values().unique().tolist()
    return value
def create_slider_marks(values):
    marks = {i: {'label': str(i)} for i in values}
    return marks