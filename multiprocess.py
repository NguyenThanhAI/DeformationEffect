import os

import argparse

import math

import time

import imageio

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

    deformed_img = np.stack([deformed_img_2, deformed_img_1, deformed_img_0], axis=-1)

    return deformed_img


def create_images_list(img: np.ndarray, magnitude: int, grid_height: int, grid_width: int,
                       num_timesteps: int, n_proc: int, show_image: bool, transform_type: str) -> list:
    height, width, _ = img.shape

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

    if transform_type == "random":
        displacement = np.random.randint(-magnitude, magnitude, (index.shape[0], index.shape[1], index.shape[2]))
    elif transform_type == "sin":
        displacement = np.array([25. * np.sin(2 * np.pi * x / 180), 25. * np.sin(2 * np.pi * y / 180)])
    else:
        raise ValueError("Not in supported transform type")

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

    alpha_x = displacement_new_x / (num_timesteps + 1)
    alpha_y = displacement_new_y / (num_timesteps + 1)

    compute_func = partial(compute_deform_image_list, img=img, global_grid_x=global_grid_x, global_grid_y=global_grid_y,
                           alpha_x=alpha_x, alpha_y=alpha_y, height=height, width=width)

    images_list = []
    images_list.append(img[:, :, ::-1])

    #start = time.time()
    with Pool(n_proc) as pool:
        image_list = pool.map(compute_func, list(range(1, num_timesteps + 2))) # Rất quan trọng, biến số được iterate biến đổi luôn là tham số vị trí đầu tiên khi định nghĩa hàm
    #end = time.time()
    #print(end - start)

    #print(len(image_list))

    images_list.extend(image_list)

    if show_image:
        while True:
            for deform_img in images_list:
                cv2.imshow("Multiprocess Deformed img", deform_img[:, :, ::-1])
                cv2.waitKey(50)

            for deform_img in reversed(images_list[:-1]):
                cv2.imshow("Multiprocess Deformed img", deform_img[:, :, ::-1])
                cv2.waitKey(50)
            break
        cv2.destroyAllWindows()

    images_list.extend(list(reversed(images_list[:-1])))

    return images_list


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_path", type=str, default=None)
    parser.add_argument("--magnitude", type=int, default=30)
    parser.add_argument("--grid_height", type=int, default=100)
    parser.add_argument("--grid_width", type=int, default=100)
    parser.add_argument("--num_timesteps", type=int, default=10)
    parser.add_argument("--n_proc", type=int, default=6)
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--show_image", type=str2bool, default=False)
    parser.add_argument("--transform_type", type=str, choices=["random", "sin"], default="random")

    args = parser.parse_args()

    return args


if __name__ == '__main__':

    args = get_args()
    image_path = args.image_path

    img = cv2.imread(image_path)
    magnitude = args.magnitude

    grid_height = args.grid_height
    grid_width = args.grid_width

    num_timesteps = args.num_timesteps # number of steps between 2 pole arrays

    n_proc = args.n_proc

    show_image = args.show_image

    transform_type = args.transform_type

    images_list = create_images_list(img=img, magnitude=magnitude, grid_height=grid_height,
                                     grid_width=grid_width, num_timesteps=num_timesteps, n_proc=n_proc,
                                     show_image=show_image, transform_type=transform_type)


    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
#
    imageio.mimsave(os.path.join(args.save_dir, os.path.splitext(os.path.basename(image_path))[0] + ".gif"),
                    images_list)
