import cv2
import numpy as np
from ID_Reader import extract_selected_mrz_data
def _order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]      # top-left
    rect[2] = pts[np.argmax(s)]      # bottom-right
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]   # top-right
    rect[3] = pts[np.argmax(diff)]   # bottom-left
    return rect

def _four_point_transform(image, pts):
    rect = _order_points(pts)
    (tl, tr, br, bl) = rect
    widthA  = np.linalg.norm(br - bl)
    widthB  = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([[0, 0],
                    [maxWidth - 1, 0],
                    [maxWidth - 1, maxHeight - 1],
                    [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped, rect

def detect_card_by_contours(image, debug=False, min_area_ratio=0.01, aspect_range=(1.2, 2.8)):
    """
    Detect a rectangular ID-like object and return a top-down crop.

    Args:
        image (np.ndarray): BGR image
        debug (bool): if True returns a debug image as 3rd return value
        min_area_ratio (float): minimum contour area relative to image area (default 1%)
        aspect_range (tuple): allowed width/height ratio for candidate (w/h)

    Returns:
        (cropped_image (np.ndarray) or None, quad_points (4x2 float32) or None, debug_image or None)
    """
    h, w = image.shape[:2]
    img_area = h * w
    min_area = max(1000, int(min_area_ratio * img_area))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Edge detection + close gaps
    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    debug_img = image.copy() if debug else None

    # 1) Prefer 4-point approximations
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2).astype("float32")
            warped, rect = _four_point_transform(image, pts)
            ar = warped.shape[1] / (warped.shape[0] + 1e-9)
            if aspect_range[0] <= ar <= aspect_range[1]:
                if debug_img is not None:
                    cv2.drawContours(debug_img, [approx], -1, (0, 255, 0), 3)
                return warped, rect, debug_img

    # 2) Fallback: minAreaRect for rotated rectangles
    for cnt in contours:
        if cv2.contourArea(cnt) < min_area:
            continue
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect).astype("float32")
        warped, rect_pts = _four_point_transform(image, box)
        ar = warped.shape[1] / (warped.shape[0] + 1e-9)
        if aspect_range[0] <= ar <= aspect_range[1]:
            if debug_img is not None:
                cv2.drawContours(debug_img, [np.int0(box)], -1, (255, 0, 0), 3)
            return warped, rect_pts, debug_img

    return None, None, debug_img
if __name__=="__main__":
    img = cv2.imread("id.jpg")
    crop, pts, dbg = detect_card_by_contours(img, debug=True)
    if crop is not None:
        data=extract_selected_mrz_data(crop)
        print(data)