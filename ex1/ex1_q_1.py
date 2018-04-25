import scipy.misc as misc
import matplotlib.pyplot as plt
from numpy import diag
from numpy.linalg import svd, norm
from matplotlib.pyplot import *
import copy

photo = misc.ascent()

np.set_printoptions(suppress=True, precision=4)
is_debug_mode = getattr(sys, 'gettrace', None)


def zero_singular_values(in_sigma, k):
    sigma = copy.deepcopy(in_sigma)
    for i in range(k, len(sigma)):
        sigma[i] = 0
    return sigma


def get_mk(matrix_u, matrix_sk, matrix_vt):
    u_sk = np.dot(matrix_u, matrix_sk)
    return np.dot(u_sk, matrix_vt)


def calculate_compression_ratio(k):
    n = len(photo)
    rank = np.linalg.matrix_rank(photo)
    a = (2 * k * n) + k
    b = (2 * n * rank) + rank
    return a / b


def show_graph(input_array1, label1, input_array2, label2):
    f, axarr = plt.subplots(1, 2)
    axarr[0].plot(input_array1)
    axarr[0].set_title(label1)
    axarr[1].plot(input_array2)
    axarr[1].set_title(label2)
    f.subplots_adjust(hspace=0.3)
    plt.show()


def get_images_to_ret():
    return [20, 30, 100, 300, 500]


def get_all_graph():
    u, sigma, v_t = svd(photo)
    forb_distance_graph = []
    compression_ratio_graph = []
    ret_images = []
    for i in range(0, len(sigma)):
        zero_k_sigma = zero_singular_values(sigma, i)
        m_k = get_mk(u, diag(zero_k_sigma), v_t)
        img_to_ret = get_images_to_ret()
        comp_k = calculate_compression_ratio(i)
        if i in img_to_ret:
            ret_images.append(m_k)
        compression_ratio_graph.append(comp_k)
        norm_k = norm(photo - m_k)
        forb_distance_graph.append(norm_k)
    return forb_distance_graph, compression_ratio_graph, ret_images


def plot_figures(figures, nrows=1, ncols=1):
    fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
    for ind, title in zip(range(len(figures)), figures):
        axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
        axeslist.ravel()[ind].set_title(title)
        axeslist.ravel()[ind].set_axis_off()
    plt.show()


def show_images(images):
    comperssions = []
    normas_dif = []
    for i in range(0, len(images)):
        comperssions.append(calculate_compression_ratio(get_images_to_ret()[i]))
        normas_dif.append(round(norm(photo - images[i]), 4))

    number_of_img = 5
    figures = {image_tilte(comperssions, normas_dif, i): images[i] for i in range(number_of_img)}
    plot_figures(figures, 1, 5)



def image_tilte(comperssions, normas_dif, i):
    return "{0}.comp: {1}\n norma: {2}\n k: {3}".format(i, comperssions[i], normas_dif[i], get_images_to_ret()[i])


def main():
    forb_distance_graph, compression_ratio_graph, images = get_all_graph()

    # ########      plotting the Frobenius distance graph     #######
    show_graph(forb_distance_graph, "forb_distance", compression_ratio_graph, "compression_ratio")

    # ########      plotting the Frobenius distance graph     #######
    show_images(images)


if __name__ == '__main__':
    main()
