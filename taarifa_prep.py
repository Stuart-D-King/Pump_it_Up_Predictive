
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder
from sklearn.model_selection import train_test_split
import pickle

def read_data():
    df_main = pd.read_csv('Taarifa Main Dataset.csv')
    df_status = pd.read_csv('Taarifa Status Dataset.csv')
    df_status = df_status.drop('id', axis=1)
    df = pd.concat([df_main, df_status], axis=1)

    return df

def clean_df(df):
    # select whick columns to use
    cols_to_use = ['basin', 'gps_height', 'region', 'scheme_management', 'construction_year', 'extraction_type_class', 'management_group', 'payment_type', 'quality_group', 'quantity_group', 'source_class', 'waterpoint_type_group', 'status_group']

    # replace non-values with np.nan
    df = df[cols_to_use]
    cols = df.columns.tolist()
    for col in cols:
        df[col] = df[col].replace([0, 'None', 'none', 'Unknown', 'unknown'], np.nan)

    # encode categorical columns
    for col in cols:
        if col != 'construction_year' and col != 'gps_height' and col != 'status_group':
            df[col] = df[col].apply(lambda x: str(x))
            le = LabelEncoder()
            le.fit(df[col])
            df[col] = le.transform(df[col])

        if col == 'status_group':
            df[col] = df[col].apply(lambda x: str(x))
            le = LabelEncoder()
            le.fit(df[col])
            class_names = le.classes_
            y_transformer = le
            df[col] = le.transform(df[col])

    # fill missing dates with mode if discrete, mean if continuous
    for col in cols:
        if col != 'gps_height':
            imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0)
            imp.fit(df[[col]])
            df[col] = imp.transform(df[[col]]).ravel()

        if col == 'gps_height':
            imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
            imp.fit(df[[col]])
            df[col] = imp.transform(df[[col]]).ravel()

    # add 'pump age' feature and pop construction year
    df['pump_age'] = df['construction_year'].apply(lambda x: 2013 - x)
    df.pop('construction_year')

    df['status_group'] = df['status_group'].astype(int)

    return df, y_transformer

def make_train_test():
    df = read_data()
    df, y_transformer = clean_df(df)
    seed = 1337

    filename = 'y_transformer.sav'
    pickle.dump(y_transformer, open(filename, 'wb'))

    y = df.pop('status_group').values
    columns = df.columns.tolist()
    X = df.values

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)
    X_train_scaled = StandardScaler().fit_transform(X_train)
    X_test_scaled = StandardScaler().fit_transform(X_test)

    directory = 'data'
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.savetxt(directory + '/X_train.csv', X_train_scaled)
    np.savetxt(directory + '/X_test.csv', X_test_scaled)
    np.savetxt(directory + '/y_train.csv', y_train)
    np.savetxt(directory + '/y_test.csv', y_test)
