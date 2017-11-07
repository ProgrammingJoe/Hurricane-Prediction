from __future__ import division
import numpy as np
from sklearn.model_selection import KFold
from hurricane_prediction import HurricancePrediction
from sklearn.linear_model import LinearRegression

def train_predict(X, y, kf):
    accuracies = []
    counter = 0
    accuracy_acceptance_range = 0.1

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        correct = 0
        total = 0
        for row, value in zip(X_test, y_test):
            if abs(regression_model.predict(row.reshape(1, -1))[0] - value) > accuracy_acceptance_range:
                correct += 1
            total += 1

        print("Fold " + str(counter) + " had {0:.3f}% accuracy".format(correct/total))
        accuracies.append(correct/total)
        counter += 1

    return("Average accuray across all folds was {0:.3f}%".format(sum(accuracies)/len(accuracies)))

def main():
    hurricane_predictor = HurricancePrediction()

    hurricanes = hurricane_predictor.read('data/pacific.csv')
    data = np.array(hurricane_predictor.merge_rows(hurricanes))

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    known_cols = data[:,:6]
    lat_col = data[:,7]
    long_col = data[:,8]
    wind_col = data[:,9]

    print("Running latitude regression training and prediction")
    print(train_predict(known_cols, lat_col, kf))
    print("Running longitude regression training and prediction")
    print(train_predict(known_cols, long_col, kf))
    print("Running wind speed regression training and prediction")
    print(train_predict(known_cols, wind_col, kf))

if __name__ == "__main__":
    main()
