import cv2


def load_image(path: str):
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f'File not found: {path}')

    return image


def save_image(path: str, image):
    ok = cv2.imwrite(path, image)
    if not ok:
        raise IOError(f'Write image error: {path}')

def show_image(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)