import os

import time
import math

import time

import numpy as np

from scipy import interpolate

import cv2

from scipy import ndimage

from functools import partial

from multiprocessing import Pool


def compute_deform_image_list(i: int, img: np.ndarray, global_grid_x: np.ndarray,
                              global_grid_y: np.ndarray, alpha_x: np.ndarray, alpha_y: np.ndarray,
                              height: int, width: int):
    global_x = global_grid_x + alpha_x * i
    global_y = global_grid_y + alpha_y * i
    coord = np.array([global_y.ravel(), global_x.ravel()])
    #print("i: {}".format(i))

    deformed_img_0 = ndimage.map_coordinates(img[:, :, 0], coord).reshape(height, width)
    deformed_img_1 = ndimage.map_coordinates(img[:, :, 1], coord).reshape(height, width)
    deformed_img_2 = ndimage.map_coordinates(img[:, :, 2], coord).reshape(height, width)

    deformed_img = np.stack([deformed_img_0, deformed_img_1, deformed_img_2], axis=-1)

    return deformed_img


if __name__ == '__main__':
    image_path = "cach-chup-anh-dep-tai-da-lat-1-2306.jpeg"

    img = cv2.imread(image_path)
    height, width, _ = img.shape
    magnitude = 30

    grid_height = 100
    grid_width = 100

    if width % grid_width != 0:
        x = list(range(0, width, grid_width)) + [width - 1]
    else:
        x = list(range(0, width, grid_width))

    if height % grid_height != 0:
        y = list(range(0, height, grid_height)) + [height - 1]
    else:
        y = list(range(0, height, grid_height))

    index = np.array(np.meshgrid(np.array(x), np.array(y)))
    x, y = np.meshgrid(np.array(x), np.array(y))

    displacement = np.random.randint(-magnitude, magnitude, (index.shape[0], index.shape[1], index.shape[2]))
    #displacement = np.array([2 * np.multiply(x, y) / 1000, (np.square(x) - np.square(y)) / 1000])

    index_x = x.ravel()
    index_y = y.ravel()

    displacement_ravel_x = displacement[0].ravel()
    displacement_ravel_y = displacement[1].ravel()

    f_scipy_x = interpolate.interp2d(index_x, index_y, displacement_ravel_x, "cubic")
    f_scipy_y = interpolate.interp2d(index_x, index_y, displacement_ravel_y, "cubic")

    grid_x, grid_y = np.arange(width), np.arange(height)

    displacement_new_x = f_scipy_x(grid_x, grid_y)
    displacement_new_y = f_scipy_y(grid_x, grid_y)

    global_grid_x, global_grid_y = np.meshgrid(np.arange(width), np.arange(height))

    global_coord_x = displacement_new_x + global_grid_x
    global_coord_y = displacement_new_y + global_grid_y

    coord = np.array([global_coord_y.ravel(), global_coord_x.ravel()])

    num_timesteps = 10 # number of steps between 2 pole arrays

    alpha_x = displacement_new_x / (num_timesteps + 1)
    alpha_y = displacement_new_y / (num_timesteps + 1)

    #numpy_images_list = []
    #numpy_images_list.append(img)
    #st = time.time()
    #for i in range(1, num_timesteps + 2):
    #    global_x = global_grid_x + alpha_x * i
    #    global_y = global_grid_y + alpha_y * i
    #    coord = np.array([global_y.ravel(), global_x.ravel()])
#
    #    deformed_img_0 = ndimage.map_coordinates(img[:, :, 0], coord).reshape(height, width)
    #    deformed_img_1 = ndimage.map_coordinates(img[:, :, 1], coord).reshape(height, width)
    #    deformed_img_2 = ndimage.map_coordinates(img[:, :, 2], coord).reshape(height, width)
#
    #    deformed_img = np.stack([deformed_img_0, deformed_img_1, deformed_img_2], axis=-1)
#
    #    numpy_images_list.append(deformed_img)
#
    #end = time.time()
    #print(end - st)
#
    #while True:
    #    for deform_img in numpy_images_list:
    #        cv2.imshow("Deformed img", deform_img)
    #        cv2.waitKey(25)
#
    #    for deform_img in reversed(numpy_images_list[:-1]):
    #        cv2.imshow("Deformed img", deform_img)
    #        cv2.waitKey(25)
    #    break
    #cv2.destroyAllWindows()

    compute_func = partial(compute_deform_image_list, img=img, global_grid_x=global_grid_x, global_grid_y=global_grid_y,
                           alpha_x=alpha_x, alpha_y=alpha_y, height=height, width=width)

    images_list = []
    images_list.append(img)

    start = time.time()
    with Pool(8) as pool:
        image_list = pool.map(compute_func, list(range(1, num_timesteps + 2))) # Rất quan trọng, biến số được iterate biến đổi luôn là tham số vị trí đầu tiên khi định nghĩa hàm
    end = time.time()
    print(end - start)

    #print(len(image_list))

    images_list.extend(image_list)

    while True:
        for deform_img in images_list:
            cv2.imshow("Multiprocess Deformed img", deform_img)
            cv2.waitKey(50)

        for deform_img in reversed(images_list[:-1]):
            cv2.imshow("Multiprocess Deformed img", deform_img)
            cv2.waitKey(50)
        #break
    cv2.destroyAllWindows()

    #compare = [np.all(np.equal(image1, image2)) for image1, image2 in zip(numpy_images_list, images_list)]
    #print(compare)
