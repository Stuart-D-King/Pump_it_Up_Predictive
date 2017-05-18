import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, Imputer, LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, recall_score, precision_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.decomposition import PCA
from keras.utils import np_utils
from taarifa_mlp import create_mlp, grid_search_mlp

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

    # convert construction_year to datetime object
    # df['construction_year'] = pd.to_datetime(df['construction_year']).dt.year

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

        # impute_subset = df[col].values
        # X_subset = df[[c for c in cols if c != col]].values
        # X_subset = StandardScaler().fit_transform(X_subset.astype(float))
        # missing = np.isnan(impute_subset)
        # model = KNeighborsClassifier()
        # model.fit(X_subset[~missing], impute_subset[~missing])
        # predicted = model.predict(X_subset[missing])
        # impute_subset[missing] = predicted
        # df[col] = impute_subset

    # add 'pump age' feature and pop construction year
    df['pump_age'] = df['construction_year'].apply(lambda x: 2013 - x)
    df.pop('construction_year')

    df['status_group'] = df['status_group'].astype(int)

    return df, class_names, y_transformer

def scale_data(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def make_histogram(data, norm=False):
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.hist(data, bins=50, edgecolor='k', normed=norm)
    ax.set_ylabel('Freqency')
    plt.show()

def cv_tuning(X, y, clf):
    n_folds=5
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=1337)

    acc_scores = np.zeros(n_folds)
    prec_scores = np.zeros(n_folds)
    rec_scores = np.zeros(n_folds)
    f1_scores = np.zeros(n_folds)

    for i, (train_inds, val_inds) in enumerate(kf.split(X)):
        X_train = X[train_inds, :]
        y_train = y[train_inds]
        X_val = X[val_inds, :]
        y_val = y[val_inds]

        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)

        acc_score = accuracy_score(y_val, y_pred)
        prec_score = precision_score(y_val, y_pred, average='weighted', labels=[0,1,2])
        rec_score = recall_score(y_val, y_pred, average='weighted', labels=[0,1,2])
        f1_sc = f1_score(y_val, y_pred, average='weighted', labels=[0,1,2])

        acc_scores[i] = acc_score
        prec_scores[i] = prec_score
        rec_scores[i] = rec_score
        f1_scores[i] = f1_sc

    print('Accuracy: {0:02f}'.format(np.mean(acc_scores)))
    print('Precision: {0:02f}'.format(np.mean(prec_scores)))
    print('Recall: {0:02f}'.format(np.mean(rec_scores)))
    print('F1: {0:02f}'.format(np.mean(f1_scores)))

def grid_search(clf, params, X, y):
    gs = GridSearchCV(estimator=clf, param_grid=params, n_jobs=-1, verbose=True)
    gs.fit(X, y)
    best_params = gs.best_params_
    best_model = gs.best_estimator_
    print('Best parameters:')
    print(best_params)
    print('Best model:')
    print(best_model)

def test_models(X, y, models_lst):
    for model in models_lst:
        print('{} scores:'.format(model.__class__.__name__))
        cv_tuning(X, y, model)
        print('-----------')

def confusion_mtrx(X_train, X_test, y_train, y_test, clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mtrx = confusion_matrix(y_test, y_pred, labels=[0,1,2])
    return mtrx

def ROC_curve(X_train, X_test, y_train, y_test, clf, labels):
    # Binarize the output
    y_train = label_binarize(y_train, classes=[0,1,2])
    y_test = label_binarize(y_test, classes=[0,1,2])
    n_classes = y_train.shape[1]

    # Learn to predict each class against the other
    ovr_clf = OneVsRestClassifier(estimator=clf, n_jobs=-1)
    y_score = ovr_clf.fit(X_train, y_train).predict_proba(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr['micro'], tpr['micro'], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc['micro'] = auc(fpr['micro'], tpr['micro'])

    plot_ROC(fpr, tpr, roc_auc, n_classes, labels)

def plot_ROC(fpr, tpr, roc_auc, n_classes, labels):
    # Compute macro-average ROC curve and ROC area
    # Aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Interpolate all ROC curves at this point(s)
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Average and compute AUC
    mean_tpr /= n_classes
    fpr['macro'] = all_fpr
    tpr['macro'] = mean_tpr
    roc_auc['macro'] = auc(fpr['macro'], tpr['macro'])

    # Plot all ROC curves
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.plot(fpr['micro'], tpr['micro'], label='micro-average ROC curve (area = {0:0.2f})'.format(roc_auc['micro']), color='deeppink', linestyle=':', linewidth=2)

    ax.plot(fpr['macro'], tpr['macro'], label='macro-average ROC curve (area = {0:0.2f})'.format(roc_auc['macro']), color='navy', linestyle=':', linewidth=2)

    colors = ['aqua', 'darkorange', 'cornflowerblue']
    for i, color, label in zip(range(n_classes), colors, labels):
        ax.plot(fpr[i], tpr[i], color=color, lw=2, label="ROC curve of class '{0}' (area = {1:0.2f})".format(label, roc_auc[i]))

    ax.plot([0, 1], [0, 1], 'k--', lw=2)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Multi-class Receiver Operating Characteristic (ROC) curve')

    plt.legend(loc='lower right')
    plt.savefig('images/ROC.png')

def leave_one_out_fi(X, y, columns, clf):
    for i, col in zip(range(X.shape[1]), columns):
        cols_to_plot = [c for c in range(X.shape[1]) if c != i]
        feats = X[:, cols_to_plot]
        rf = clf.fit(feats, y)
        scores = rf.feature_importances_

        fig = plt.figure(figsize=(8,8))
        ax = fig.add_subplot(111)
        ax.bar(range(len(columns) - 1), scores)
        ax.set_xticks(range(len(columns) - 1))
        col_names = [columns[i] for i in cols_to_plot]
        ax.set_xticklabels(col_names, rotation='vertical')

        plt.subplots_adjust(bottom=0.25)
        plt.savefig('images/left_out_{}.png'.format(col))
        plt.close('all')

def leave_one_out(X, y, columns, clf):
    base_mod = clf.fit(X, y)
    base_score = base_mod.oob_score_
    diff_from_base = []
    for col in range(X.shape[1]):
        X = X[:,[c for c in range(X.shape[1]) if c != col]]
        clf.fit(X, y)
        diff_from_base.append(abs(base_score - clf.oob_score_))

    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.bar(range(len(diff_from_base)), diff_from_base)
    ax.set_xticks(range(len(diff_from_base)))
    ax.set_xticklabels(columns, rotation='vertical')
    ax.set_title('Leave One Out')
    plt.savefig('images/loo_oob.png')
    plt.close('all)')

def pca_fit_transform(X, n_comps=10):
    pca = PCA(n_components=n_comps, random_state=1)
    pca.fit(X)
    X_transformed = pca.transform(X)
    return pca, X_transformed

def scree_plot(pca, title=None):
    num_components = pca.n_components_
    ind = np.arange(num_components)
    vals = pca.explained_variance_ratio_
    plt.figure(figsize=(10, 6), dpi=250)
    ax = plt.subplot(111)
    ax.bar(ind, vals, 0.35,
        color=[(0.949, 0.718, 0.004),
               (0.898, 0.49, 0.016),
               (0.863, 0, 0.188),
               (0.694, 0, 0.345),
               (0.486, 0.216, 0.541),
               (0.204, 0.396, 0.667),
               (0.035, 0.635, 0.459),
               (0.486, 0.722, 0.329),
              ])

    for i in range(num_components):
        ax.annotate(r'%s%%' % ((str(vals[i]*100)[:4])), (ind[i]+0.2, vals[i]), va="bottom", ha="center", fontsize=12)

    ax.set_xticklabels(ind,
                    fontsize=12)

    ax.set_ylim(0, max(vals)+0.05)
    ax.set_xlim(0-0.45, 8+0.45)

    ax.xaxis.set_tick_params(width=0)
    ax.yaxis.set_tick_params(width=2, length=12)

    ax.set_xlabel('Principal Component', fontsize=12)
    ax.set_ylabel('Variance Explained (%)', fontsize=12)

    if title is not None:
        plt.title(title, fontsize=16)

    plt.savefig('images/scree_plt.png')
    plt.close()

def fit_mlp(X, y):
    params = {'num_neurons_in_layer': [20, 50, 100, 250], 'activation': ['sigmoid', 'relu'], 'kernel_initializer': ['uniform', 'zeros'], 'epochs': [2, 4], 'batch_size': [100, 200, 500]}

    grid_search_mlp(X, y, params)


if __name__ == '__main__':
    plt.close('all')
    seed = 1337
    # create 'images' directory if it does not already exist - repository for created plots
    directory = 'images'
    if not os.path.exists(directory):
        os.makedirs(directory)

    # df = read_data()
    # df, labels, y_transformer = clean_df(df)
    # df.to_pickle('df_taarifa.pkl')

    labels = np.array(['functional', 'functional needs repair', 'non functional'])

    df = pd.read_pickle('df_taarifa.pkl')
    y = df.pop('status_group').values
    columns = df.columns.tolist()
    X = df.values

    # Binarize the output
    # y = label_binarize(y, classes=[0, 1, 2])
    # n_classes = y.shape[1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed)

    # y_train_ohe = np_utils.to_categorical(y_train)
    # y_test_ohe = np_utils.to_categorical(y_test)
    X_train_scaled = scale_data(X_train)
    X_test_scaled = scale_data(X_test)

    # pca_train, X_train_transformed = pca_fit_transform(X_train_scaled, n_comps=10)
    # pca_test, X_test_transformed = pca_fit_transform(X_test_scaled, n_comps=10)

    # ---TEST DIFFERENT MODELS---
    # models = [RandomForestClassifier(), KNeighborsClassifier(), XGBClassifier(), GradientBoostingClassifier(), AdaBoostClassifier(), LogisticRegression()]
    # test_models(X_train_transformed, y_train, models_lst=models)

    # ---RANDOM FOREST GRID SEARCH---
    # random_forest_grid = {'max_depth': [4, 6, None],
    #     'max_features': ['sqrt', 'log2', None],
    #     'class_weight': ['balanced'],
    #     'min_samples_split': [2, 4, 6],
    #     'min_samples_leaf': [1, 2, 4],
    #     'n_estimators': [100, 200],
    #     'random_state': [1]}
    # clf = RandomForestClassifier()
    # grid_search(clf, random_forest_grid, X_train_transformed, y_train)

    # Best parameters:
    # {'bootstrap': True, 'class_weight': 'balanced', 'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 4, 'n_estimators': 40, 'random_state': 1}

    # clf = RandomForestClassifier(n_estimators=40, min_samples_split=4, oob_score=True, class_weight='balanced', random_state=1)

    # test_models(X_train_scaled, y_train, models_lst=[clf])

    # ---CREATE CONFUSION MATRIX---
    # mtrx = confusion_mtrx(X_train_scaled, X_test_scaled, y_train, y_test, clf)

    # ---CREATE ROC PLOT---
    # ROC_curve(X_train_scaled, X_test_scaled, y_train, y_test, clf, labels)

    # ---LEAVE ONE OUT (OOB SCORES)---
    # leave_one_out(X_train_scaled, y_train, columns, clf)

    # ---LEAVE ONE OUT FEATURE IMPORTANCE---
    # leave_one_out_fi(X_train_scaled, y_train, columns)

    # ---MULTI-LAYER PERCEPTRON---
    # fit_mlp(X_train_scaled, y_train)

    # --PCA SCREE PLOT---
    # scree_plot(pca_train, title='Taarifa PCA Scree Plot')
