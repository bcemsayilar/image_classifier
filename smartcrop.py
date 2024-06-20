import os

import cv2 as cv
import numpy as np


def detect(image, outdir="./cropped", crop=False, square=True):
    # Check if the input is a path or an image object
    if isinstance(image, str):
        img = cv.imread(image)
    else:
        # Convert PIL image to OpenCV format
        img = np.array(image)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    sift = cv.SIFT_create(edgeThreshold=8)
    kp = sift.detect(gray, None)

    all_points = [i.pt for i in kp]
    x_points = [z[0] for z in all_points]
    y_points = [z[1] for z in all_points]
    thresh = 0
    x_min, y_min = int(min(x_points)) - thresh, int(min(y_points) - thresh)
    x_max, y_max = int(max(x_points)) + thresh, int(max(y_points) + thresh)
    min_side = min((x_max - x_min), (y_max - y_min))
    max_side = max((x_max - x_min), (y_max - y_min))
    x_mean, y_mean = int((x_max + x_min) / 2), int((y_max + y_min) / 2)
    squared_x_min, squared_x_max = x_mean - int(min_side / 2), x_mean + int(
        min_side / 2
    )
    squared_y_min, squared_y_max = y_mean - int(min_side / 2), y_mean + int(
        min_side / 2
    )

    if crop and not square:
        cropped_image = img[y_min:y_max, x_min:x_max]

        if isinstance(image, str):
            image_relative = image.split("/")[-1]
            cv.imwrite(os.path.join(outdir, image_relative), cropped_image)

    elif crop and square:
        cropped_image = img[squared_y_min:squared_y_max, squared_x_min:squared_x_max]

        if isinstance(image, str):
            image_relative = image.split("/")[-1]
            cv.imwrite(os.path.join(outdir, image_relative), cropped_image)

    else:
        cv.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv.rectangle(
            img,
            (squared_x_min, squared_y_min),
            (squared_x_max, squared_y_max),
            (0, 0, 255),
            2,
        )

        if isinstance(image, str):
            image_relative = image.split("/")[-1]
            cv.imwrite(os.path.join(outdir, image_relative), img)

    return cropped_image if crop else img
