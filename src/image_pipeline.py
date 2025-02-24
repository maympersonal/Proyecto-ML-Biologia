import cv2
import numpy as np
import torch
import torchvision.transforms as transforms


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)


def difference_of_gaussians(
    gray_image: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0
) -> np.ndarray:
    blur1 = cv2.GaussianBlur(gray_image, (0, 0), sigma1)
    blur2 = cv2.GaussianBlur(gray_image, (0, 0), sigma2)
    return cv2.subtract(blur1, blur2)


def segment_image(image):
    _, thresh = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return 1 - thresh


def add_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    if len(image1.shape) == 2 and len(image2.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)
    elif len(image1.shape) == 3 and len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)

    result = cv2.add(image1, image2)
    return result


def estimate_normal_map(rgb_image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)

    normal_map = np.zeros_like(rgb_image, dtype=np.float32)
    normal_map[..., 0] = sobel_x
    normal_map[..., 1] = sobel_y
    normal_map[..., 2] = 1.0

    norm = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map = normal_map / (norm + 1e-5)

    normal_map = (normal_map + 1) / 2 * 255
    return normal_map.astype(np.uint8)


def estimate_depth_map(rgb_image: np.ndarray) -> np.ndarray:
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS")
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform(rgb_image).unsqueeze(0)

    with torch.no_grad():
        depth_map = model(input_tensor)

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))
    depth_map = (
        (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    )
    return depth_map.astype(np.uint8)


def detect_edges(
    gray_image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150
) -> np.ndarray:
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)
    return edges


def find_bounding_boxes(thresh_img: np.ndarray):
    contours, _ = cv2.findContours(
        thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]

    return bounding_boxes


def draw_bounding_boxes(image: np.ndarray, bounding_boxes: list) -> np.ndarray:
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return image


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    img = cv2.imread("1_06_16-19-_jpg.rf.7a24616d5cae5716ba581f2a743d66f6.jpg")

    # img2 = cv2.imread("1_06_16-19-_jpg.rf.7a24616d5cae5716ba581f2a743d66f6.jpg")
    # img = to_grayscale(img)
    # img = difference_of_gaussians(img)
    # img = segment_image(img)
    # img = estimate_normal_map(img)
    # img = estimate_depth_map(img)
    # img = add_images(img, img2)
    # bbox = find_bounding_boxes(img)
    # img = draw_bounding_boxes(img2, bbox)
    # img = detect_edges(img)

    

    if len(img.shape) == 3:
        img = img[..., ::-1]

    plt.imshow(img, "gray")
    plt.axis("off")
    plt.show()
