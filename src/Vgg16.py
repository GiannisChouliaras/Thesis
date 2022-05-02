import torchvision
import numpy as np

from sklearn.decomposition import PCA
from typing import Callable, List
from ImageFunctions import Image
from PIL import Image as Img

# transform callable function for the images
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(223),
        torchvision.transforms.CenterCrop(223),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[-1.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        ),
    ]
)

features = {}


def load_vgg16():
    model = torchvision.models.vgg16(pretrained=True)

    # helping function for the hooks
    def reg_hook(layer: int) -> Callable:
        def hook(model, input, output):
            features[layer] = output.detach()

        return hook

    # register hooks
    model.features[11].register_forward_hook(reg_hook(11))
    model.features[30].register_forward_hook(reg_hook(30))
    model.classifier[0].register_forward_hook(reg_hook(0))

    return model


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


def get_features_from_lst(model, lst: List[Image]) -> None:
    for image in lst:
        pil_image = Img.fromarray(image.array.astype("uint8"), "RGB")
        x = transform(pil_image).unsqueeze(0).to("cpu")
        _ = model(x)
        image.layer30 = normalize(features[30].cpu().numpy())
        image.layer11 = pca(normalize(features[11].cpu().numpy()))
        image.fully_connected = normalize(features[0].cpu().numpy())
