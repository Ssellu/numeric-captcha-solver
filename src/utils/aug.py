import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image
import os

import cv2
import imageio
import imgaug as ia
import numpy as np
from imgaug import augmenters as iaa


def aa():
    image = cv2.imread(os.path.join(
        './data/sample_dataset/e2/original_images', '001.png'))

    ia.seed(4)

    rotate = iaa.Affine(rotate=(-25, 25))
    image_aug = rotate(image=image)

    print("Augmented:")
    ia.imshow(image_aug, backend='cv2')


WIDTH = 640
HEIGHT = 480
X_INTERVAL = WIDTH // 3
Y_INTERVAL = HEIGHT // 10


def write_numbers(img: np.ndarray, text: str):
    # Step 1: Load custom font file
    font_path = ".\\font\\Lucida Console ANSI Regular\\Lucida Console ANSI Regular.ttf"
    font = ImageFont.truetype(font_path, size=40, kerning=True)

    # Step 2: Add text to image using custom font
    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    textbbox = draw.textbbox((0, 0), text, font=font)
    x = (img.shape[1] - textbbox[2]) // 2
    y = (img.shape[0] - textbbox[3]) // 2
    draw.text((x, y), text, font=font, fill=(255))
    img = np.array(img_pil)
    return img


# create black canvas of size 500x500
canvas = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)

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
    cv2.line(canvas, start_point, new_point, (255), 1)

    # update the starting point for the next iteration
    start_point = new_point

canvas = cv2.GaussianBlur(canvas, (3, 3), 1.5)
canvas = write_numbers(canvas, '12345')
_, canvas = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY_INV)


# display the image and wait for a key event
cv2.imshow("Random curve", canvas)
cv2.waitKey(0)

# close the window
cv2.destroyAllWindows()
