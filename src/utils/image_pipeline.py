from typing import Literal
import cv2  # type:ignore
import numpy as np
import torch
import torchvision.transforms as transforms


def to_grayscale(image_rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)  # type:ignore


def difference_of_gaussians(
    gray_image: np.ndarray, sigma1: float = 1.0, sigma2: float = 2.0
) -> np.ndarray:
    blur1 = cv2.GaussianBlur(gray_image, (0, 0), sigma1)  # type:ignore
    blur2 = cv2.GaussianBlur(gray_image, (0, 0), sigma2)  # type:ignore
    return cv2.subtract(blur1, blur2)  # type:ignore


def segment_image(image):
    _, thresh = cv2.threshold(image, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # type:ignore
    return 1 - thresh


def add_images(image1: np.ndarray, image2: np.ndarray) -> np.ndarray:
    if len(image1.shape) == 2 and len(image2.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_GRAY2RGB)  # type:ignore
    elif len(image1.shape) == 3 and len(image2.shape) == 2:
        image2 = cv2.cvtColor(image2, cv2.COLOR_GRAY2RGB)  # type:ignore

    result = cv2.add(image1, image2)  # type:ignore
    return result


def estimate_normal_map(rgb_image: np.ndarray) -> np.ndarray:
    gray_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)  # type:ignore
    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=5)  # type:ignore
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=5)  # type:ignore

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
    model.eval()  # type:ignore

    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    input_tensor = transform(rgb_image).unsqueeze(0)  # type:ignore

    with torch.no_grad():
        depth_map = model(input_tensor)  # type:ignore

    depth_map = depth_map.squeeze().cpu().numpy()
    depth_map = cv2.resize(depth_map, (rgb_image.shape[1], rgb_image.shape[0]))  # type:ignore
    depth_map = (
        (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255
    )
    return depth_map.astype(np.uint8)


def detect_edges(
    gray_image: np.ndarray, low_threshold: int = 50, high_threshold: int = 150
) -> np.ndarray:
    edges = cv2.Canny(gray_image, low_threshold, high_threshold)  # type:ignore
    return edges


def calculate_iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    # Calculate intersection coordinates
    x_inter1 = max(x1, x2)
    y_inter1 = max(y1, y2)
    x_inter2 = min(x1 + w1, x2 + w2)
    y_inter2 = min(y1 + h1, y2 + h2)

    # Calculate intersection area
    inter_width = max(0, x_inter2 - x_inter1)
    inter_height = max(0, y_inter2 - y_inter1)
    intersection_area = inter_width * inter_height

    # Calculate areas of the two boxes
    area1 = w1 * h1
    area2 = w2 * h2

    # Calculate IoU
    iou = intersection_area / min(
        area1, area2
    )  # IoA (Intersection over Area of the smaller box)
    return iou


def filter_overlapping_boxes(bounding_boxes, ioa_threshold=0.9):
    filtered_boxes = []
    for i, box1 in enumerate(bounding_boxes):
        keep = True
        for j, box2 in enumerate(bounding_boxes):
            if i == j:
                continue
            ioa = calculate_iou(box1, box2)
            if (
                ioa > ioa_threshold and box1[2] * box1[3] < box2[2] * box2[3]
            ):  # If box1 is smaller and mostly covered by box2
                keep = False
                break
        if keep:
            filtered_boxes.append(box1)
    return filtered_boxes


def find_bounding_boxes(thresh_img: np.ndarray):
    contours, _ = cv2.findContours(  # type:ignore
        thresh_img,
        cv2.RETR_EXTERNAL,  # type:ignore
        cv2.CHAIN_APPROX_SIMPLE,  # type:ignore
    )
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours]  # type:ignore
    filtered_boxes = filter_overlapping_boxes(bounding_boxes)

    return filtered_boxes


def draw_bounding_boxes(image: np.ndarray, bounding_boxes: list) -> np.ndarray:
    for x, y, w, h in bounding_boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)  # type:ignore
    return image


def enhance_image_contrast(image: np.ndarray) -> np.ndarray:
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)  # type: ignore
    l_channel, a_channel, b_channel = cv2.split(lab_image)  # type: ignore
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # type: ignore
    enhanced_l_channel = clahe.apply(l_channel)
    enhanced_lab_image = cv2.merge((enhanced_l_channel, a_channel, b_channel))  # type: ignore
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)  # type: ignore
    return enhanced_image


def apply_sharpness_filter(image: np.ndarray) -> np.ndarray:
    kernel: np.ndarray = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened_image: np.ndarray = cv2.filter2D(image, -1, kernel)  # type: ignore
    return sharpened_image


def increase_image_saturation(
    image: np.ndarray, saturation_factor: float
) -> np.ndarray:
    hsv_image: np.ndarray = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # type: ignore
    h_channel, s_channel, v_channel = cv2.split(hsv_image)  # type: ignore
    s_channel = np.clip(s_channel * saturation_factor, 0, 255).astype(np.uint8)
    enhanced_hsv_image: np.ndarray = cv2.merge((h_channel, s_channel, v_channel))  # type: ignore
    enhanced_image: np.ndarray = cv2.cvtColor(enhanced_hsv_image, cv2.COLOR_HSV2BGR)  # type: ignore
    return enhanced_image


def extract_color_channel(
    image: np.ndarray, color: Literal["red", "green", "blue"]
) -> np.ndarray:
    b_channel, g_channel, r_channel = cv2.split(image)  # type: ignore
    if color == "red":
        return r_channel
    elif color == "green":
        return g_channel
    elif color == "blue":
        return b_channel
    else:
        raise ValueError("Color must be 'red', 'green', or 'blue'")


def enhance_image_resolution(image: np.ndarray, scale_factor: float = 2) -> np.ndarray:
    if scale_factor <= 1:
        raise ValueError("Scale factor must be greater than 1 to enhance resolution.")

    new_dimensions = (
        int(image.shape[1] * scale_factor),
        int(image.shape[0] * scale_factor),
    )
    enhanced_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_LINEAR)  # type:ignore
    return enhanced_image


def resize_and_pad_image(
    image: np.ndarray, target_size: tuple = (224, 224)
) -> np.ndarray:
    original_height, original_width = image.shape[:2]
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = target_size[0]
        new_height = int(new_width / aspect_ratio)
    else:
        new_height = target_size[1]
        new_width = int(new_height * aspect_ratio)

    resized_image = cv2.resize(  # type: ignore
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR,  # type: ignore
    )

    delta_w = target_size[0] - new_width
    delta_h = target_size[1] - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    padded_image = cv2.copyMakeBorder(  # type: ignore
        resized_image,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,  # type: ignore
        value=color,
    )

    return padded_image
