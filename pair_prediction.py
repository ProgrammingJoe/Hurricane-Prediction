from __future__ import division
import numpy as np
from sklearn.model_selection import KFold
from hurricane_prediction import HurricancePrediction
from sklearn.linear_model import LinearRegression
from gpxpy.geo import haversine_distance

LATITUDE_ALPHA = 0.1 # degrees (NOTE 1 degree of latitude is ~111km)
LONGITUDE_ALPHA = 0.2 # degrees (Getting legnth of 1 degree of longitude depends on what latitude you are at)
SURFACE_DISTANCE_ALPHA = 10 # km
MAXWIND_ALPHA = 1.0 # knots?

def train_predict(X, y, kf, accuracy_acceptance_range=0.1, use_surface_distance=False):
    accuracies = []
    counter = 0

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        correct = 0
        total = 0
        for X_row, y_true_row in zip(X_test, y_test):

            y_pred_row = regression_model.predict(X_row.reshape(1, -1))[0]

            # IF we are using longitude and latitude together then we can predict using haversine distance
            if use_surface_distance == True:
                distance = haversine_distance(y_pred_row[0], y_pred_row[1], y_true_row[0], y_true_row[1])/1000 #gives of km
                if distance < accuracy_acceptance_range:
                    correct += 1
            # IF we are predicting a single value we can use this
            elif abs(y_pred_row - y_true_row) < accuracy_acceptance_range:
                correct += 1
            total += 1

        print("Fold " + str(counter) + " had {0:.3f}% accuracy".format(correct/total*100))
        accuracies.append(correct/total)
        counter += 1

    return("Average accuracy across all folds was {0:.3f}%".format(sum(accuracies)/len(accuracies)*100))

def print_avg_error_distance(X, y_latlong, kf):
    ### Get the avg distance our guess was away from the actual spot ###
    print("Running lat,long regression and calculating the average distance between our predictions and the ground truth")
    y = y_latlong
    counter = 0
    all_avg_errors_in_distance = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        errors_in_distance = []
        for X_row, y_true_row in zip(X_test, y_test):
            y_pred_row = regression_model.predict(X_row.reshape(1, -1))[0]
            distance = haversine_distance(y_pred_row[0], y_pred_row[1], y_true_row[0], y_true_row[1])/1000 #gives of km
            errors_in_distance.append(distance)

        avg_error_in_distance = np.average(errors_in_distance)
        print("Fold " + str(counter) + " had an average error in distance of %.3f km" % avg_error_in_distance )
        all_avg_errors_in_distance.append(avg_error_in_distance)
        counter += 1
    print("Average error across all folds was %.3f km\n" % np.average(all_avg_errors_in_distance))

def print_avg_error_wind(X, y_wind, kf):
    ### Get the avg error our guess was away from the actual speed ###
    print("Running lat,long regression and calculating the average distance between our predictions and the ground truth")
    y = y_wind
    counter = 0
    all_avg_errors_in_windspeed = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        regression_model = LinearRegression()
        regression_model.fit(X_train, y_train)

        errors_in_windspeed = []
        for X_row, y_true_row in zip(X_test, y_test):
            y_pred_row = regression_model.predict(X_row.reshape(1, -1))[0]
            error = np.abs(y_true_row - y_pred_row)
            errors_in_windspeed.append(error)

        avg_error_in_windspeed = np.average(errors_in_windspeed)
        print("Fold " + str(counter) + " had an average error in windspeed of %.3f knots" % avg_error_in_windspeed )
        all_avg_errors_in_windspeed.append(avg_error_in_windspeed)
        counter += 1
    print("Average error across all folds was %.3f knots\n" % np.average(all_avg_errors_in_windspeed))


def main():
    hurricane_predictor = HurricancePrediction()

    hurricanes = hurricane_predictor.read('data/pacific.csv')
    data = np.array(hurricane_predictor.merge_rows(hurricanes))

    kf = KFold(n_splits=5, random_state=42, shuffle=True)

    known_cols = data[:,:6]
    lat_col = data[:,7]
    long_col = data[:,8]
    lat_long = np.array([[i,j] for i,j in zip(lat_col,long_col)])
    wind_col = data[:,9]

    print()
    print("Running latitude regression training and prediction with alpha %.3f (degrees)" % LATITUDE_ALPHA)
    print(train_predict(known_cols, lat_col, kf, accuracy_acceptance_range=LATITUDE_ALPHA), "\n")

    print("Running longitude regression training and prediction with alpha %.3f (degrees)" % LONGITUDE_ALPHA)
    print(train_predict(known_cols, long_col, kf, accuracy_acceptance_range=LONGITUDE_ALPHA), "\n")

    print("Running lat,long regression training and prediction with alpha %.3f (km on surface of earth)" % SURFACE_DISTANCE_ALPHA)
    print(train_predict(known_cols,
                        lat_long,
                        kf, accuracy_acceptance_range=SURFACE_DISTANCE_ALPHA,
                        use_surface_distance=True), "\n")

    print("Running wind speed regression training and prediction with alpha %.3f (knots)" % MAXWIND_ALPHA)
    print(train_predict(known_cols, wind_col, kf, accuracy_acceptance_range=MAXWIND_ALPHA), "\n")

    print_avg_error_distance(known_cols, lat_long, kf)
    print_avg_error_wind(known_cols, wind_col, kf)

if __name__ == "__main__":
    main()
