import numpy
import matplotlib.pyplot as plt
import math

def generator():
    data = numpy.random.binomial(1, 0.25, (100000, 1000))
    epsilon_arr = [0.5, 0.25, 0.1, 0.01, 0.001]
    plot_five(data)
    chev_data, hoef_data = calc_chevichev_hoeffding(epsilon_arr)
    percenteges = calc_deviators(data, epsilon_arr)
    for epsilon in range(5):
        plt.title("Epsilon is - " + str(epsilon_arr[epsilon]))
        plt.ylabel("Delta")
        plt.xlabel("m - number of coin flips")
        plt.plot(chev_data[epsilon], label='Chevichev Upper Bound')
        plt.plot(hoef_data[epsilon], label='Hoeffding Upper Bound')
        plt.plot(percenteges[epsilon], label="P' of |Xm-E[X]|>=e")
        plt.legend()
        plt.show()


def plot_five(data):
    rows_sum = [[0] * 1000, [0] * 1000, [0] * 1000, [0] * 1000, [0] * 1000]
    plt.ylabel("Number of Heads")
    plt.xlabel("m - number of coin flips")

    for m in range(1, 1001):
        for i in range(m):
            rows_sum[0][m - 1] += data[0][i]
            rows_sum[1][m - 1] += data[1][i]
            rows_sum[2][m - 1] += data[2][i]
            rows_sum[3][m - 1] += data[3][i]
            rows_sum[4][m - 1] += data[4][i]
        rows_sum[0][m - 1] /= float(m)
        rows_sum[1][m - 1] /= float(m)
        rows_sum[2][m - 1] /= float(m)
        rows_sum[3][m - 1] /= float(m)
        rows_sum[4][m - 1] /= float(m)
    plt.plot(rows_sum[0], label='Data line 1')
    plt.plot(rows_sum[1], label='Data line 2')
    plt.plot(rows_sum[2], label='Data line 3')
    plt.plot(rows_sum[3], label='Data line 4')
    plt.plot(rows_sum[4], label='Data line 5')
    plt.legend()
    plt.show()

def calc_chevichev_hoeffding(epsilon_arr):

    upper_bounds_hoef = [[0]*1000, [0]*1000, [0]*1000, [0]*1000, [0]*1000]
    upper_bounds_chev = [[0]*1000, [0]*1000, [0]*1000, [0]*1000, [0]*1000]

    for m in range(1, 1001):
        for i in range(5):
            chev_res = float(1 / float(4 * m * epsilon_arr[i] * epsilon_arr[i]))
            hoef_res = 2 * float(math.exp(-2 * m * epsilon_arr[i] * epsilon_arr[i]))
            if chev_res > 1:
                chev_res = 1
            if hoef_res > 1:
                hoef_res = 1
            upper_bounds_chev[i][m - 1] = chev_res
            upper_bounds_hoef[i][m - 1] = hoef_res
    return upper_bounds_chev, upper_bounds_hoef




def calc_deviators(data, epsilon_arr):
    p = 0.25
    cum_data = numpy.cumsum(data, axis=1)
    total_data = [[0]*1000, [0]*1000, [0]*1000, [0]*1000, [0]*1000]
    for curr_epsilon in range(len(epsilon_arr)):
        for m in range(1, 1001):
            bad_lines_counter = 0
            for i in range(len(data)):
                if math.fabs((cum_data[i][m - 1]/float(m)) - p) >= epsilon_arr[curr_epsilon]:
                    bad_lines_counter += 1
            total_data[curr_epsilon][m - 1] = float(bad_lines_counter/float(len(data)))
        print("Finished epsilon number " + str(curr_epsilon))
    return total_data

if __name__ == '__main__':
    generator()
