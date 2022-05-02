import argparse
from os.path import exists
from typing import List, Callable
from SVM import init_svm

import cv2 as cv
import numpy as np
import torch
import torchvision
from PIL import Image as Img
from sklearn.decomposition import PCA

from func import array_to_tensor, convert_number
from net import Net
from net_crossentropy import SoftNet

parser = argparse.ArgumentParser(description="Evaluating AK treatment")
parser.add_argument(
    "--case",
    metavar="case",
    type=int,
    help="Enter the number of the case",
    required=True,
)
args = parser.parse_args()


# Creating the Image class
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
            int(roi[1]) : int(roi[1] + roi[3]), int(roi[0]) : int(roi[0] + roi[2])
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

    if len(extract) == 0 or len(extract) < 3:
        raise Exception(
            "The extracted images should be 3. The wounded area first and then the normal skin."
        )

    if len(extract) > 3:
        extract = extract[:3]

    samples: List[Image] = [Image(crop_image(roi=row, img=image)) for row in extract]
    cv.destroyAllWindows()
    return samples


def main(case: str) -> None:
    # *******************
    # getting the images
    # *******************

    before_path = f"../data/raw/CASE{case}/paired/before.jpg"
    after_path = f"../data/raw/CASE{case}/paired/after.jpg"

    before: List[Image] = get_images(before_path)
    after: List[Image] = get_images(after_path)

    # ********************************************
    # Downloading and initialize the model (CNN)
    # ********************************************
    model = torchvision.models.vgg16(pretrained=True)
    features = {}

    # helping function for the hooks
    def reg_hook(layer: int) -> Callable:
        def hook(model, input, output):
            features[layer] = output.detach()

        return hook

    # register hooks
    model.features[11].register_forward_hook(reg_hook(11))
    model.features[30].register_forward_hook(reg_hook(30))
    model.classifier[0].register_forward_hook(reg_hook(0))

    # transform callable function for the images
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize(224),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),
        ]
    )

    # *******************************************************
    #   For every image, run the CNN and store the results
    # *******************************************************

    # Run the CNN for the images and update the lists (layers) of the classes
    def normalize(array: np.ndarray) -> np.ndarray:
        return array / np.linalg.norm(array)

    # PCA method for the layer 11
    def pca(pca_image: np.ndarray, dimension: int = 5) -> np.ndarray:
        """A tensor with H x W x C, we reshape it to an array of HW x C (pixels x dimension of data)"""
        N = pca_image.shape[2] * pca_image.shape[3]  # HxW
        C = pca_image.shape[1]  # Dimensions: Kernels
        X = np.reshape(pca_image, [N, C])
        feats = PCA(n_components=dimension).fit_transform(X)
        return np.reshape(
            feats, [pca_image.shape[2], pca_image.shape[3], feats.shape[1]]
        )

    def get_features_from_lst(lst: List[Image]) -> None:
        for image in lst:
            pil_image = Img.fromarray(image.array.astype("uint8"), "RGB")
            x = transform(pil_image).unsqueeze(0).to("cpu")
            _ = model(x)
            image.layer30 = normalize(features[30].cpu().numpy())
            image.layer11 = pca(normalize(features[11].cpu().numpy()))
            image.fully_connected = normalize(features[0].cpu().numpy())

    get_features_from_lst(before)
    get_features_from_lst(after)

    # ********************************************************
    #       Get the cosine similarities for images
    # ********************************************************

    def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
        return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

    def average(lst: List) -> float:
        return sum(lst) / len(lst)

    def calculate_similarities(
        wound: np.ndarray, first: np.ndarray, second: np.ndarray
    ) -> float:
        c1 = cosine_similarity(wound, first)
        c2 = cosine_similarity(wound, second)
        return average([c1, c2])

    def get_final_results(lst: List[Image]) -> List:
        layer30 = calculate_similarities(
            lst[0].layer30.flatten(), lst[1].layer30.flatten(), lst[2].layer30.flatten()
        )
        layer11 = calculate_similarities(
            lst[0].layer11.flatten(), lst[1].layer11.flatten(), lst[2].layer11.flatten()
        )
        fully_c = calculate_similarities(
            lst[0].fully_connected.flatten(),
            lst[1].fully_connected.flatten(),
            lst[2].fully_connected.flatten(),
        )
        return [layer30, layer11, fully_c]

    before_results = array_to_tensor(get_final_results(before))
    after_results = array_to_tensor(get_final_results(after))

    # ********************************************
    #   Ready to use Net to predict the score
    # ********************************************

    # load model
    net = Net(inFeats=3, outFeats=1, fHidden=100, sHidden=50)
    net.load_state_dict(torch.load("../models/net.pt"))
    net.eval()
    svm = init_svm('rbf')

    # predict
    before_prediction = round(net(before_results).item(), 3)
    after_prediction = round(net(after_results).item(), 3)

    svm_before = svm.predict([list(before_results)])[0]
    svm_after = svm.predict([list(after_results)])[0]

    print(
        "Sigmoid: Before: ",
        convert_number(before_prediction, range1=(0, 1), range2=(1, 8)),
        end=" ------- ",
    )

    print("After: ", convert_number(after_prediction, range1=(0, 1), range2=(1, 8)))

    print(f"svm: before: {round(svm_before, 1)} ----- after: {round(svm_after, 1)}")


if __name__ == "__main__":
    main(case=args.case)
