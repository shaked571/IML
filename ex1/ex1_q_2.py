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
        rows = np.random.rand(len(processed_data)) < x / 100
        training_data = processed_data[rows]
        test_data = processed_data[~rows]

        price_vector = training_data[[PRICE]]
        data_matrix = training_data.drop([PRICE], axis=1)

        testing_price_vector = test_data[[PRICE]]
        testing_data_matrix = test_data.drop([PRICE], axis=1)

        prediction_w = np.dot(np.linalg.pinv(data_matrix), price_vector)

        train_vec = np.dot(data_matrix, prediction_w)
        test_vec = np.dot(testing_data_matrix, prediction_w)

        testing_mse = get_mse(test_vec, testing_price_vector)
        test_error.append(testing_mse)

        train_mse = get_mse(train_vec, price_vector)
        train_error.append(train_mse)

    show_graph(train_error, test_error)


def show_graph(train_error, test_error):
    plt.plot(train_error, 'g', test_error, 'r')
    plt.title("train error(g) vs test error(r) ")
    plt.show()

def get_mse(y, y_roof):
    return np.mean((y - y_roof) ** 2)


def get_predict_vec(matrix, vector):
    return np.dot(matrix, np.transpose(vector))





def process_data(row_data):
    temp_processed_data = clean_unwanted_columns(row_data)
    remove_data(temp_processed_data)
    temp_processed_data = temp_processed_data[temp_processed_data[SQFT_LIVING] > 250]
    temp_processed_data = pd.get_dummies(temp_processed_data, columns=[ZIPCODE])
    temp_processed_data.insert(0, 'w_0', 1)
    processed_data = temp_processed_data.dropna(axis=0, how="any")
    return processed_data

def remove_data(origin_matrix):
    origin_matrix.drop(origin_matrix[(origin_matrix.price <= 0) |
                                     (origin_matrix.bedrooms <= 0) |
                                     (origin_matrix.bathrooms <= 0) |
                                     (origin_matrix.sqft_living <= 0) |
                                     (origin_matrix.sqft_lot <= 0) |
                                     (origin_matrix.grade <= 0) |
                                     (origin_matrix.sqft_above <= 0) |
                                     (origin_matrix.sqft_basement <= 0)].index,
                       inplace=True)

def clean_unwanted_columns(row_data):
    return row_data.drop(
        [ID, DATE, WATERFRONT, VIEW, LAT, LONG], axis=1)


def get_row_data():
    return pd.read_csv("kc_house_data.csv")


if __name__ == '__main__':
    main()
