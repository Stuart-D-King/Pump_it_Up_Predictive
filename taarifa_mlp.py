import numpy as np
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import SGD, Adagrad
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import theano
import pdb

def create_mlp(num_classes=3, num_feats=12, activation=None, num_neurons_in_layer=None, kernel_initializer=None):
    model = Sequential()
    # pdb.set_trace()
    model.add(Dense(input_shape=(num_feats,),
                     units=num_neurons_in_layer,
                     kernel_initializer=kernel_initializer, activation=activation))
    model.add(Dense(units=num_classes,
                     kernel_initializer=kernel_initializer,
                     activation='softmax'))
    sgd = SGD(lr=0.001, decay=1e-7, momentum=.9) # (keep)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=["accuracy"]) # (keep)
    return model

def grid_search_mlp(X, y, params):
    y_ohe = np_utils.to_categorical(y)
    model = KerasClassifier(build_fn=create_mlp, epochs=None, batch_size=None, verbose=0)

    grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1)

    grid_result = grid.fit(X, y_ohe)

    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    #
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == '__main__':
    rng_seed = 2
    np.random.seed(rng_seed)
