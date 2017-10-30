"""
Baseline which uses a simple sklearn NN

X vectors is filled with data from a specific hurricane at a specific time,
corresponding y vector entries are filled with data from the next time
"""

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from hurricane_prediction import HurricancePrediction

NUM_SPLITS = 2
MAX_ITERATIONS = 400
HIDDEN_LAYER_SIZES = (100, 100)

kf = KFold(n_splits=NUM_SPLITS, random_state=42, shuffle=True)

hurricane_predictor = HurricancePrediction()
hurricane_dict = hurricane_predictor.read('data/atlantic.csv')
X,y = hurricane_predictor.to_Xy(hurricane_dict) #Needs to be fixed so that it misses the last entry

neural_network = MLPRegressor(solver='lbfgs',
                              random_state=1,
                              activation='tanh',
                              hidden_layer_sizes=HIDDEN_LAYER_SIZES,
                              tol=1e-7)
neural_network.fit(X, y)
print(neural_network.score(X,y))

pred = neural_network.predict(X)

# for i in range(0,10):
#     if prediction[i][0] != 914.63033598:
#         print("NEXT>>>")
#         print(y[i])
#         print(prediction[i])

# for train_index, test_index in kf.split(X):
#     # neural_network = MLPRegressor(solver='adam',
#     #                               activation='tanh',
#     #                               alpha=1e-4,
#     #                               hidden_layer_sizes=HIDDEN_LAYER_SIZES,
#     #                               random_state=1,
#     #                               max_iter=200,
#     #                               tol=1e-4)
#     neural_network = MLPRegressor()
#
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]
#
#     neural_network.fit(X_train, y_train)
#     score = neural_network.score(X_test, y_test)
#     print(score)
#     # #
#     prediction = neural_network.predict(X_test)
#     print(np.subtract(y_test, prediction))
#     # # # for i in range(0, len(X_test)):
#     # #

