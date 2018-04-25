import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ZIPCODE = "zipcode"

LONG = "long"

LAT = "lat"

YEAR_RENOVATED = "yr_renovated"

FLOORS = "floors"

YEAR_BUILT = "yr_built"

CONDITION = "condition"

VIEW = "view"

WATERFRONT = "waterfront"

DATE = "date"

ID = "id"

SQFT_LIVING = "sqft_living"

PRICE = "price"


def main():
    # getting the data and clean it
    row_data = get_row_data()
    processed_data = process_data(row_data)
    train_error = []
    test_error = []
    for x in range(1, 100):
        training_data = processed_data.sample(frac=(x/100))
        testing_data = processed_data.drop(training_data.index)

        data_matrix, price_vector = get_data_matrix_and_price_vec(training_data)
        testing_data_matrix, testing_price_vector = get_data_matrix_and_price_vec(testing_data)

        w = np.dot(data_matrix, price_vector)
        prediction_value_vector = get_predict_vec(data_matrix, w)
        testing_prediction_value_vector = np.dot(prediction_value_vector, testing_price_vector)

        train_mse = get_mse(prediction_value_vector, price_vector)
        testing_mse = get_mse(testing_prediction_value_vector, testing_price_vector)
        test_error.append(testing_mse)
        train_error.append(train_mse)

    show_graph(train_error, test_error)


def show_graph(input_array1, test_error):
    plt.plot(input_array1, 'g', test_error, 'r')
    plt.show()

def get_mse(y, y_roof):
    return np.mean((y - y_roof) ** 2)


def get_predict_vec(matrix, vector):
    return np.dot(np.transpose(matrix), vector)


def get_data_matrix_and_price_vec(training_data):
    panda_price_vector = training_data[[PRICE]]
    data = training_data.drop([PRICE], axis=1)
    return np.linalg.pinv(data), panda_price_vector.values


def process_data(row_data):
    temp_processed_data = clean_unwanted_columns(row_data)
    temp_processed_data = temp_processed_data[temp_processed_data[PRICE] > 0]
    remove_data(temp_processed_data)
    temp_processed_data = temp_processed_data[temp_processed_data[SQFT_LIVING] > 250]
    temp_processed_data = pd.get_dummies(temp_processed_data, columns=[ZIPCODE])
    processed_data = temp_processed_data.dropna(axis=0, how="any")
    return processed_data

def remove_data(df):
    df.drop(df[(df.price <= 0) | (df.bedrooms <= 0) |
               (df.bathrooms <= 0) | (df.sqft_living <= 0) |
               (df.sqft_lot <= 0) | (df.grade <= 0) |
               (df.sqft_above <= 0) | (df.sqft_basement <= 0)].index,
            inplace=True)
    df.dropna(inplace=True)

def clean_unwanted_columns(row_data):
    return row_data.drop(
        [ID, DATE, WATERFRONT, VIEW, CONDITION, YEAR_BUILT, FLOORS, YEAR_RENOVATED, LAT, LONG], axis=1)


def get_row_data():
    return pd.read_csv("kc_house_data.csv")


if __name__ == '__main__':
    main()
