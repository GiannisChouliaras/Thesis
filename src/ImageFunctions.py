import numpy as np
import cv2 as cv

from os.path import exists
from typing import List


class Image:
    def __init__(self, array: np.ndarray):
        self.array = array
        self.layer30 = None
        self.layer11 = None
        self.fully_connected = None


# Create the function that will open and cut the images, they return a list of Image objects
def get_images(path: str) -> List:
    if not exists(path):
        raise FileNotFoundError("Path is not valid. Check if case exists.")

    def resize_window(_image, res_width=None, res_height=None, inter=cv.INTER_AREA):
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

    def crop_image(roi: List, img: np.ndarray) -> np.ndarray:
        return img[
               int(roi[1]): int(roi[1] + roi[3]), int(roi[0]): int(roi[0] + roi[2])
               ]

    image = cv.imread(path)

    # Centering the image
    cv.namedWindow("select samples")
    image = resize_window(image, res_width=1360)
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
    cv.destroyAllWindows()
    return samples
