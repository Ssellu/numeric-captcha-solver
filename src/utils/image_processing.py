import os
import glob
from PIL import Image

import cv2
import numpy as np
import matplotlib.pyplot as plt


def save_image(save_dir: str, save_filename: str, array: np.ndarray):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    im = Image.fromarray(array)
    im.save(f"{save_filename}.png")


def remove_border(origin_img: np.ndarray):
    print(origin_img)
    for iy, ix in np.ndindex(origin_img[1:-1, 1:-1].shape):
        if origin_img[iy, ix] == 0 and origin_img[iy-1, ix] and origin_img[iy+1, ix]:
            origin_img[iy, ix] = 254
        elif origin_img[iy, ix] == 0 and origin_img[iy, ix-1] and origin_img[iy, ix+1]:
            origin_img[iy, ix] = 254
    print('=' * 100)
    print(origin_img)
    return origin_img

# Zerorize all non-black pixels (only keep black color lines)
def convert_binary(origin_img: np.ndarray):
    _, im = cv2.threshold(origin_img, 0, 254, cv2.THRESH_BINARY)
    return im


def undistort_center(origin_img: np.ndarray):
    origin_img = np.pad(origin_img, ((55, 55), (0, 0)),
                        'constant', constant_values=255)
    rows, cols = origin_img.shape

    # ---① 설정 값 셋팅
    exp = 0.8       # 볼록, 오목 지수 (오목 : 0.1 ~ 1, 볼록 : 1.1~)
    scale = 0.7           # 변환 영역 크기 (0 ~ 1)

    # 매핑 배열 생성 ---②
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)

    # 좌상단 기준좌표에서 -1~1로 정규화된 중심점 기준 좌표로 변경 ---③
    mapx = 2*mapx/(cols-1)-1
    mapy = 2*mapy/(rows-1)-1

    # 직교좌표를 극 좌표로 변환 ---④
    r, theta = cv2.cartToPolar(mapx, mapy)

    # 왜곡 영역만 중심확대/축소 지수 적용 ---⑤
    r[r < scale] = r[r < scale] ** exp

    # 극 좌표를 직교좌표로 변환 ---⑥
    mapx, mapy = cv2.polarToCart(r, theta)

    # 중심점 기준에서 좌상단 기준으로 변경 ---⑦
    mapx = ((mapx + 1)*cols-1)/2
    mapy = ((mapy + 1)*rows-1)/2
    # 재매핑 변환
    distorted = cv2.remap(origin_img, mapx, mapy, cv2.INTER_LINEAR)
    return distorted

def main(src_dir: str, save_dir: str):
    filenames = glob.glob(os.path.join(src_dir, '*.png'))
    t = 3

    for filename in filenames:
        img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        origin_img = img.copy()

        img = convert_binary(img)
        img = remove_border(img)
        img = undistort_center(img)

        t -= 1
        if t == 0:
            break
        cv2.imshow('origin', origin_img)
        cv2.imshow('distorted', img)
        cv2.waitKey()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main('./data/origin_dataset', './data/result_dataset')
