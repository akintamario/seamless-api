import cv2
import numpy as np

def make_seamless(image: np.ndarray) -> np.ndarray:
    h, w, _ = image.shape

    # 左右をブレンドしてシームレスにする
    left = image[:, :w//2]
    right = image[:, w//2:]
    blended_lr = cv2.addWeighted(left, 0.5, right, 0.5, 0)

    # 上下もブレンドしてシームレスに
    top = image[:h//2, :]
    bottom = image[h//2:, :]
    blended_tb = cv2.addWeighted(top, 0.5, bottom, 0.5, 0)

    # 真ん中にブレンドした部分を配置
    result = image.copy()
    result[:, :w//2] = blended_lr
    result[:h//2, :] = blended_tb

    return result
