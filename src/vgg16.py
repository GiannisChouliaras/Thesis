import torchvision
import numpy as np

from typing import Callable, List
from classes import Image, Layer
from PIL import Image as Img
from sklearn.decomposition import PCA


# transform callable function for the images
transform = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(224),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            mean=[0.4247, 0.4662, 0.6567],
            std=[0.1296, 0.1301, 0.1386],
        ),
    ]
)

features = {}


def load_vgg16() -> torchvision.models.vgg.VGG:
    """Load and @return the vgg 16 model with the hooks on the features"""
    model = torchvision.models.vgg16(pretrained=True)

    def reg_hook(layer: int) -> Callable:
        def hook(model, input, output):
            features[layer] = output.detach()

        return hook

    model.features[Layer.EARLY.value].register_forward_hook(reg_hook(Layer.EARLY.value))
    model.features[Layer.LATE.value].register_forward_hook(reg_hook(Layer.LATE.value))
    model.classifier[Layer.FULLY_CONNECTED.value].register_forward_hook(
        reg_hook(Layer.FULLY_CONNECTED.value)
    )
    return model


def normalize(array: np.ndarray) -> np.ndarray:
    """@return the normalized (divided with the norm) @param array"""
    return array / np.linalg.norm(array)


def principal_component_analysis(image: np.ndarray, dimension: int = 3) -> np.ndarray:
    """A tensor with H x W x C, we reshape it to an array of HW x C (pixels x dimension of data)"""
    N = image.shape[2] * image.shape[3]  # HxW
    C = image.shape[1]  # Dimensions: Kernels
    X = np.reshape(image, [N, C])
    feats = PCA(n_components=dimension).fit_transform(X)
    return np.reshape(feats, [image.shape[2], image.shape[3], feats.shape[1]])


def populate_images_with_features(model, lst: List[Image]) -> None:
    """Using @param model, get the features for every layer."""
    for image in lst:
        pil_image = Img.fromarray(image.array.astype("uint8"), "RGB")
        x = transform(pil_image).unsqueeze(0).to("cpu")
        _ = model(x)
        image.early = principal_component_analysis(
            normalize(features[Layer.EARLY.value].cpu().numpy())
        )
        image.late = normalize(features[Layer.LATE.value].cpu().numpy())
        image.fully_connected = normalize(
            features[Layer.FULLY_CONNECTED.value].cpu().numpy()
        )
