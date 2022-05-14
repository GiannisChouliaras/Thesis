import numpy as np
import cv2 as cv

from os.path import exists
from skimage.io import imread
from typing import List, Any


class Image:
    def __init__(self, array: np.ndarray):
        """Class representing an image that has the array and the desired cnn features."""
        self.array = array
        self.layer11 = None
        self.layer12 = None
        self.layer28 = None
        self.layer30 = None
        self.fully_connected = None


def resize_window(
        _image, res_width=None, res_height=None, inter=cv.INTER_AREA
) -> np.ndarray:
    """Resizes a window given a @param res_width or @param res_height"""
    (h, w) = _image.shape[:2]
    if res_width is None and res_height is None:
        return _image
    if res_width is None:
        rad = h / float(h)
        dim = (int(w * rad), res_height)
    else:
        rad = res_width / float(w)
        dim = (res_width, int(h * rad))
    return cv.resize(_image, dim, interpolation=inter)


# Create the function that will open and cut the images, they return a list of Image objects
def get_images(path: str) -> tuple[list[Image], Any, list[np.ndarray]]:
    if not exists(path):
        raise FileNotFoundError("Path is not valid. Check if case exists.")

    def crop_image(roi: List, img: np.ndarray) -> np.ndarray:
        return img[
               int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])
               ]

    image = cv.imread(path)
    color_image = imread(path)

    # Centering the image
    cv.namedWindow("select samples")
    image = resize_window(image, res_width=1360)
    color_image = resize_window(color_image, res_width=1360)
    resolution = (1920, 1080)
    width = int(resolution[0] / 2 - image.shape[1] / 2)
    height = int(resolution[1] / 2 - (image.shape[0] / 2) - 30)
    cv.moveWindow("select samples", width, height)
    extract = cv.selectROIs(
        "select samples", img=image, showCrosshair=False, fromCenter=False
    )

    # Check if we've cropped 3 squares or more.
    if len(extract) == 0 or len(extract) < 3:
        raise Exception(
            "The extracted images should be 3. The wounded area first and then the normal skin."
        )

    # Keep the first 3 images of the cropped.
    if len(extract) > 3:
        extract = extract[:3]

    samples: List[Image] = [Image(crop_image(roi=row, img=image)) for row in extract]
    cropped_images = [crop_image(roi=row, img=color_image) for row in extract]
    cv.destroyAllWindows()
    return samples, extract, cropped_images


def draw_squares(case: str, extracted: np.ndarray, score: float) -> None:
    """Draws squares of cropped images and places the score as texture."""
    image = cv.imread(case)
    image = resize_window(image, res_width=1360)
    wound = extracted[0]
    image = cv.rectangle(
        image,
        (wound[0], wound[1]),
        (wound[0] + wound[2], wound[1] + wound[3]),
        (166, 32, 68),
        2,
    )
    for img in extracted[1:]:
        image = cv.rectangle(
            image,
            (img[0], img[1]),
            (img[0] + img[2], img[1] + img[3]),
            (32, 166, 82),
            2,
        )
    cv.putText(
        image,
        f"The score is {score}",
        (20, 100),
        cv.FONT_HERSHEY_SIMPLEX,
        1,
        (209, 80, 0, 255),
        2,
    )
    cv.imshow(winname="Thesis", mat=image)
    cv.waitKey(0)
    cv.destroyAllWindows()
