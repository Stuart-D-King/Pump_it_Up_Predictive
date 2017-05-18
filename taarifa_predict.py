
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

def predict():
    directory = 'data'
    X_test = np.loadtxt(directory + '/X_test.csv')
    y_test = np.loadtxt(directory + '/y_test.csv')

    y_transformer = pickle.load(open('y_transformer.sav', 'rb'))

    model = pickle.load(open('trained_model.sav', 'rb'))
    result = model.score(X_test, y_test)
    predictions = model.predict(X_test).astype(int)
    transformed_predictions = y_transformer.inverse_transform(predictions)
    np.savetxt('predict_results.csv', transformed_predictions, fmt='%s')

    print('Accuracy: {0:02f}'.format(result))
    print('Predictions:')
    print(transformed_predictions)

if __name__ == '__main__':
    predict()
