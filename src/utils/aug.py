import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import os

import cv2
import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa


# TODO Y_OFFSET도 글자마다 다르게해볼까?

def aa(image:np.ndarray):
    ia.seed(4)

    rotate = iaa.Affine(rotate=(-25, 25))
    image_aug = rotate(image=image)

    print("Augmented:")
    ia.imshow(image_aug, backend='cv2')


WIDTH = 160
HEIGHT = 50
X_INTERVAL = WIDTH // 3
Y_INTERVAL = HEIGHT // 10




def put_random_numbers(img: np.ndarray):
    
    CYCLE = 6
    FONT_PATH = ".\\font\\Lucida Console ANSI Regular\\Lucida Console ANSI Regular.ttf"
    TEXT_SIZE = 55
    TEXT_HEIGHT = 40
    TEXT_ITEM_SIZE = 33
    PAD_MEAN = 10
    
    font = ImageFont.truetype(FONT_PATH, size=TEXT_SIZE)
    scales = np.random.normal(loc=PAD_MEAN, scale=1.5, size=CYCLE) 
    x = (WIDTH - (TEXT_ITEM_SIZE*CYCLE - sum(scales))) // 2
    print(f'x offset : {x}')
    for n in scales:
        y = (HEIGHT-TEXT_HEIGHT) // 2
        r = str(np.random.randint(9) + 1)
        
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        draw.text((x, y), r, font=font, fill=(255))
        img = np.array(img_pil)
        print(f'x : {x} y : {y}')
        x += TEXT_ITEM_SIZE - n
    return img
        

def draw_randomline(canvas:np.ndarray):
    # choose a random starting point
    start_point = (0, np.random.randint(0, HEIGHT // 2))

    # loop to generate random points and connect them with a line
    while start_point[0] < WIDTH and start_point[1] < HEIGHT:
        # generate random offset values for x and y
        x_offset = abs(int(np.random.normal(
            loc=X_INTERVAL // 2, scale=X_INTERVAL)))
        y_offset = abs(int(np.random.normal(
            loc=Y_INTERVAL // 2, scale=Y_INTERVAL)))
        print(f'x offset : {x_offset}')
        print(f'y offset : {y_offset}')
        print('======================')
        # calculate the new point coordinates
        new_point = (start_point[0]+x_offset, start_point[1]+y_offset)
        # new_point = (start_point[1]+y_offset, start_point[0]+x_offset)

        # draw a line from the previous point to the new point
        cv2.line(canvas, start_point, new_point, (255), 2)

        # update the starting point for the next iteration
        start_point = new_point

    return canvas

# create black canvas of size 500x500
canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
canvas = draw_randomline(canvas)
canvas = draw_randomline(canvas)

canvas = cv2.GaussianBlur(canvas, (1, 1), 0.55)
canvas = put_random_numbers(canvas)

canvas = cv2.GaussianBlur(canvas, (0, 0),  0.55)


# 구조화 요소 커널, 사각형 (3x3) 생성 ---①
k = cv2.getStructuringElement(cv2.MORPH_RECT, (2,2))
# 팽창 연산 적용 ---②
canvas = cv2.dilate(canvas, k)


canvas = ~canvas

# display the image and wait for a key event
cv2.imshow("Random curve", canvas)
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()
