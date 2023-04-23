import os
import glob
import re

from PIL import ImageFont, ImageDraw, Image

import cv2
import numpy as np

WIDTH = 160
HEIGHT = 50
X_INTERVAL = WIDTH // 3
Y_INTERVAL = HEIGHT // 10
IMAGE_DIR = '.\\data\\images'
ANNO_DIR = '.\\data\\annotations'
filename = 0

def get_lastest_filename(dir: str):
    previous_files = glob.glob(f"{dir}/*")
    if previous_files:
        numeric_sorted = sorted(previous_files, reverse=True, key=lambda file_name: int(re.compile(r"\d+").search(file_name).group(0)))[0]
        return int(os.path.splitext(os.path.basename(numeric_sorted))[0])
    return 0


def to_yolo_format(no: str, bbox: tuple):
    """ Convert (left, top, right, bottom) into (cls_no, center_x, center_y, w, h) """
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    return no, center_x, center_y, width, height


def save_dataset(image: np.ndarray, annotations: tuple):
    global filename
    filename += 1
    print(f"new file name : {filename}")
    if not os.path.exists(IMAGE_DIR):
        os.makedirs(IMAGE_DIR)
    image_path = os.path.join(IMAGE_DIR, f"{filename}.png")
    im = Image.fromarray(image)
    im.save(image_path)

    if not os.path.exists(ANNO_DIR):
        os.makedirs(ANNO_DIR)
    anno_path = os.path.join(ANNO_DIR, f"{filename}.txt")
    with open(anno_path, 'w') as f:
        f.write("\n".join([" ".join(list(map(str, tu)))
                for tu in annotations]))


def put_numbers(image: np.ndarray):

    CYCLE = 6
    FONT_PATH = ".\\font\\Lucida Console ANSI Regular\\Lucida Console ANSI Regular.ttf"
    TEXT_SIZE = 53
    TEXT_ITEM_WIDTH = 32
    TEXT_ITEM_HEIGHT = 28
    PAD_MEAN = 10

    font = ImageFont.truetype(FONT_PATH, size=TEXT_SIZE)
    x_scales = np.random.normal(loc=PAD_MEAN, scale=1.75, size=CYCLE)
    y_scales = np.random.normal(loc=PAD_MEAN, scale=0.75, size=CYCLE)

    x = (WIDTH - (TEXT_ITEM_WIDTH*CYCLE - sum(x_scales))) // 2

    annotations = []

    for x_scale, y_scale in zip(x_scales, y_scales):

        r = str(np.random.randint(9) + 1)
        y = (HEIGHT - TEXT_ITEM_HEIGHT) // 2 - y_scale

        img_pil = Image.fromarray(image)
        draw = ImageDraw.Draw(img_pil)
        bbox = draw.textbbox((x, y), r)
        annotations.append(to_yolo_format(r, bbox=bbox))

        draw.text((x, y), r, font=font, fill=(255))
        image = np.array(img_pil)

        x += TEXT_ITEM_WIDTH - x_scale

    return image, annotations


def draw_randomline(image: np.ndarray):
    # choose a random starting point
    start_point = (0, np.random.randint(0, HEIGHT // 2))

    # loop to generate random points and connect them with a line
    while start_point[0] < WIDTH and start_point[1] < HEIGHT:
        # generate random offset values for x and y
        x_offset = abs(int(np.random.normal(
            loc=X_INTERVAL // 2, scale=X_INTERVAL)))
        y_offset = abs(int(np.random.normal(
            loc=Y_INTERVAL // 2, scale=Y_INTERVAL)))

        # calculate the new point coordinates
        new_point = (start_point[0]+x_offset, start_point[1]+y_offset)
        # new_point = (start_point[1]+y_offset, start_point[0]+x_offset)

        # draw a line from the previous point to the new point
        cv2.line(image, start_point, new_point, (255), 2)

        # update the starting point for the next iteration
        start_point = new_point

    return image


def create_dataset():
    # create black canvas of size 500x500
    img = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    img = draw_randomline(img)
    img = draw_randomline(img)

    img = cv2.GaussianBlur(img, (1, 1), 0.55)
    img, annotations = put_numbers(img)

    img = cv2.GaussianBlur(img, (0, 0),  0.55)

    # 구조화 요소 커널, 사각형 (3x3) 생성 ---①
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # 팽창 연산 적용 ---②
    img = cv2.dilate(img, k)

    img = ~img

    return img, annotations


def show_dataset(image: np.ndarray):
    # display the image and wait for a key event
    cv2.imshow("Result", image)
    cv2.waitKey(0)

    # close the window
    cv2.destroyAllWindows()


if __name__ == '__main__':
    filename = get_lastest_filename(IMAGE_DIR)
    print(filename)
    for n in range(10):
        img, annotations = create_dataset()
        save_dataset(image=img, annotations=annotations)
        # show_dataset(img)
