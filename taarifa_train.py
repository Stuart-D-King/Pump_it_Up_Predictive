
import pickle
import numpy as np
from taarifa_prep import make_train_test
from sklearn.ensemble import RandomForestClassifier

def train_model():
    make_train_test()
    directory = 'data'
    X_train = np.loadtxt(directory + '/X_train.csv')
    y_train = np.loadtxt(directory + '/y_train.csv')

    model = RandomForestClassifier(n_estimators=40, min_samples_split=4, oob_score=True, class_weight='balanced', random_state=1)

    model.fit(X_train, y_train)

    filename = 'trained_model.sav'
    pickle.dump(model, open(filename, 'wb'))

if __name__ == '__main__':
    train_model()
