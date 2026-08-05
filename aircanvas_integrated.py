import cv2
import mediapipe as mp
import numpy as np
import math
import time
import os
import random
from datetime import datetime
from collections import deque


# -----------------------------
# Folders
# -----------------------------

os.makedirs("captured_photos", exist_ok=True)
os.makedirs("triangle_random_filtered_photos", exist_ok=True)


# -----------------------------
# Theme Colors - BGR
# -----------------------------

BG_BLUE = (70, 35, 15)
PANEL_BLUE = (120, 75, 35)
DARK_FILL = (34, 22, 12)

ACCENT_BLUE = (255, 150, 50)
ACCENT_BLUE_2 = (255, 190, 100)

WHITE = (245, 245, 245)
LIGHT_TEXT = (225, 225, 235)
MID_TEXT = (190, 210, 235)

YELLOW = (0, 255, 255)
TRASH_RED_BORDER = (65, 65, 230)


# -----------------------------
# Basic Helpers
# -----------------------------

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def boxes_overlap(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    return ax1 < bx2 and ax2 > bx1 and ay1 < by2 and ay2 > by1


def finger_is_up(landmarks, tip_id, pip_id):
    return landmarks[tip_id].y < landmarks[pip_id].y


def point_in_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box[:4]

    return x1 <= px <= x2 and y1 <= py <= y2


def draw_centered_text(img, text, y, font, scale, color, thickness=1):
    text_size = cv2.getTextSize(text, font, scale, thickness)[0]
    x = (img.shape[1] - text_size[0]) // 2

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def wrap_text(text, max_chars):
    words = text.split()
    lines = []
    current = ""

    for word in words:
        test = current + " " + word if current else word

        if len(test) <= max_chars:
            current = test
        else:
            lines.append(current)
            current = word

    if current:
        lines.append(current)

    return lines


def averaged_swipe_delta(point_history):
    if len(point_history) < 6:
        return 0, 0

    points = list(point_history)

    start_points = points[:3]
    end_points = points[-3:]

    start_x = sum(p[0] for p in start_points) / len(start_points)
    start_y = sum(p[1] for p in start_points) / len(start_points)

    end_x = sum(p[0] for p in end_points) / len(end_points)
    end_y = sum(p[1] for p in end_points) / len(end_points)

    return end_x - start_x, end_y - start_y


# -----------------------------
# Hand Gesture Helpers
# -----------------------------

def is_open_palm(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = finger_is_up(lm, 8, 6)
    middle_up = finger_is_up(lm, 12, 10)
    ring_up = finger_is_up(lm, 16, 14)
    pinky_up = finger_is_up(lm, 20, 18)

    return index_up and middle_up and ring_up and pinky_up


def is_peace_sign(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = finger_is_up(lm, 8, 6)
    middle_up = finger_is_up(lm, 12, 10)
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_up and ring_down and pinky_down


def is_index_only(hand_landmarks):
    lm = hand_landmarks.landmark

    index_up = lm[8].y < lm[6].y
    middle_down = lm[12].y > lm[10].y
    ring_down = lm[16].y > lm[14].y
    pinky_down = lm[20].y > lm[18].y

    return index_up and middle_down and ring_down and pinky_down


def is_index_up(hand_landmarks):
    lm = hand_landmarks.landmark
    return lm[8].y < lm[6].y


def normalized_landmark_distance(lm, a, b):
    return math.hypot(lm[a].x - lm[b].x, lm[a].y - lm[b].y)


def finger_is_folded_for_thumbs_up(lm, tip_id, pip_id, mcp_id, wrist_id=0):
    tip_below_pip = lm[tip_id].y > lm[pip_id].y - 0.005
    tip_closer_than_pip = (
        normalized_landmark_distance(lm, wrist_id, tip_id)
        < normalized_landmark_distance(lm, wrist_id, pip_id) * 1.18
    )
    tip_close_to_mcp = (
        normalized_landmark_distance(lm, tip_id, mcp_id)
        < normalized_landmark_distance(lm, wrist_id, mcp_id) * 0.95
    )

    return tip_below_pip or tip_closer_than_pip or tip_close_to_mcp


def is_thumbs_up(hand_landmarks):
    lm = hand_landmarks.landmark

    wrist = lm[0]
    thumb_tip = lm[4]
    thumb_ip = lm[3]
    thumb_mcp = lm[2]

    palm_width = max(normalized_landmark_distance(lm, 5, 17), 0.001)
    palm_height = max(normalized_landmark_distance(lm, 0, 9), 0.001)

    thumb_length = normalized_landmark_distance(lm, 2, 4)
    thumb_above_palm = thumb_tip.y < lm[5].y - palm_height * 0.12
    thumb_above_wrist = thumb_tip.y < wrist.y - palm_height * 0.18
    thumb_mostly_vertical = abs(thumb_tip.y - thumb_mcp.y) > abs(thumb_tip.x - thumb_mcp.x) * 0.75
    thumb_extended = thumb_length > palm_width * 0.62
    thumb_tip_clear = normalized_landmark_distance(lm, 4, 5) > palm_width * 0.45

    index_folded = finger_is_folded_for_thumbs_up(lm, 8, 6, 5)
    middle_folded = finger_is_folded_for_thumbs_up(lm, 12, 10, 9)
    ring_folded = finger_is_folded_for_thumbs_up(lm, 16, 14, 13)
    pinky_folded = finger_is_folded_for_thumbs_up(lm, 20, 18, 17)

    folded_count = sum([index_folded, middle_folded, ring_folded, pinky_folded])

    open_finger_count = sum([
        finger_is_up(lm, 8, 6),
        finger_is_up(lm, 12, 10),
        finger_is_up(lm, 16, 14),
        finger_is_up(lm, 20, 18),
    ])

    return (
        folded_count >= 3
        and open_finger_count <= 1
        and thumb_extended
        and thumb_tip_clear
        and thumb_mostly_vertical
        and (thumb_above_palm or thumb_above_wrist)
    )


def draw_thumbs_up_hold_progress(frame, progress):
    h, w = frame.shape[:2]
    progress = clamp(progress, 0.0, 1.0)

    center = (w // 2, 76)
    radius = 30
    angle = int(360 * progress)

    cv2.circle(frame, center, radius, BG_BLUE, -1)
    cv2.circle(frame, center, radius, WHITE, 1)
    cv2.ellipse(frame, center, (radius, radius), -90, 0, angle, ACCENT_BLUE, 4)

    cv2.putText(
        frame,
        "OK",
        (center[0] - 17, center[1] + 7),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        WHITE,
        2,
        cv2.LINE_AA,
    )


# -----------------------------
# Shape Detection
# -----------------------------

def total_path_length(points):
    total = 0

    for i in range(1, len(points)):
        total += distance(points[i - 1], points[i])

    return total


def remove_near_duplicates(points, min_gap=6):
    if not points:
        return []

    cleaned = [points[0]]

    for p in points[1:]:
        if distance(cleaned[-1], p) >= min_gap:
            cleaned.append(p)

    return cleaned


def get_hull_and_approx(points, eps_factor=0.05):
    contour = np.array(points, dtype=np.int32)
    hull = cv2.convexHull(contour)
    perimeter = cv2.arcLength(hull, True)

    if perimeter == 0:
        return hull, None, perimeter

    approx = cv2.approxPolyDP(hull, eps_factor * perimeter, True)

    return hull, approx, perimeter


def angles_of_polygon(corners):
    angles = []
    n = len(corners)

    for i in range(n):
        p_prev = corners[i - 1][0]
        p_curr = corners[i][0]
        p_next = corners[(i + 1) % n][0]

        v1 = p_prev - p_curr
        v2 = p_next - p_curr

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0 or norm2 == 0:
            continue

        cosang = np.dot(v1, v2) / (norm1 * norm2)
        cosang = np.clip(cosang, -1.0, 1.0)

        angle = np.degrees(np.arccos(cosang))
        angles.append(angle)

    return angles


def is_square_motion(points):
    if len(points) < 32:
        return False

    pts = remove_near_duplicates(points, min_gap=6)

    if len(pts) < 24:
        return False

    pts_np = np.array(pts, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(pts_np)

    if width < 95 or height < 95:
        return False

    aspect_ratio = width / float(height)

    if aspect_ratio < 0.76 or aspect_ratio > 1.24:
        return False

    movement_length = total_path_length(pts)

    if movement_length < 330:
        return False

    start_end_distance = distance(pts[0], pts[-1])

    if start_end_distance > max(80, min(width, height) * 0.42):
        return False

    hull, approx, perimeter = get_hull_and_approx(pts, eps_factor=0.040)

    if approx is None:
        return False

    hull_area = cv2.contourArea(hull)
    bounding_area = width * height

    if hull_area < 7000 or bounding_area == 0:
        return False

    fill_ratio = hull_area / bounding_area

    # A square should fill most of its bounding box. Triangles usually fill less.
    if fill_ratio < 0.58 or fill_ratio > 0.95:
        return False

    corner_count = len(approx)

    if corner_count != 4:
        return False

    angles = angles_of_polygon(approx)

    if len(angles) != 4:
        return False

    if not all(68 <= angle <= 112 for angle in angles):
        return False

    corners = approx.reshape(-1, 2)
    side_lengths = [
        distance(corners[i], corners[(i + 1) % 4])
        for i in range(4)
    ]

    if min(side_lengths) < 55:
        return False

    if max(side_lengths) / min(side_lengths) > 1.75:
        return False

    return True


def is_triangle_motion(points):
    if len(points) < 30:
        return False

    pts = remove_near_duplicates(points, min_gap=6)

    if len(pts) < 22:
        return False

    pts_np = np.array(pts, dtype=np.int32)
    x, y, width, height = cv2.boundingRect(pts_np)

    if width < 90 or height < 80:
        return False

    movement_length = total_path_length(pts)

    if movement_length < 285:
        return False

    start_end_distance = distance(pts[0], pts[-1])

    if start_end_distance > max(90, min(width, height) * 0.50):
        return False

    hull, approx, perimeter = get_hull_and_approx(pts, eps_factor=0.065)

    if approx is None:
        return False

    hull_area = cv2.contourArea(hull)
    bounding_area = width * height

    if hull_area < 4500 or bounding_area == 0:
        return False

    fill_ratio = hull_area / bounding_area

    # Triangles usually occupy about half of the bounding rectangle.
    if fill_ratio < 0.28 or fill_ratio > 0.64:
        return False

    corner_count = len(approx)

    if corner_count != 3:
        return False

    angles = angles_of_polygon(approx)

    if len(angles) != 3:
        return False

    if min(angles) < 28 or max(angles) > 125:
        return False

    corners = approx.reshape(-1, 2)
    side_lengths = [
        distance(corners[i], corners[(i + 1) % 3])
        for i in range(3)
    ]

    if min(side_lengths) < 60:
        return False

    if max(side_lengths) / min(side_lengths) > 2.4:
        return False

    return True


# -----------------------------
# Camera Effects
# -----------------------------

def apply_zoom(frame, zoom_level):
    if zoom_level <= 1.0:
        return frame.copy()

    h, w = frame.shape[:2]

    new_w = int(w / zoom_level)
    new_h = int(h / zoom_level)

    x1 = (w - new_w) // 2
    y1 = (h - new_h) // 2

    cropped = frame[y1:y1 + new_h, x1:x1 + new_w]
    zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)

    return zoomed


def apply_camera_mode(frame, current_mode, selfie_segmentation):
    if current_mode == "PHOTO":
        return frame.copy()

    if current_mode == "PORTRAIT":
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = selfie_segmentation.process(rgb_frame)

        if results.segmentation_mask is None:
            return frame.copy()

        mask = results.segmentation_mask > 0.5
        blurred_bg = cv2.GaussianBlur(frame, (45, 45), 0)
        output = np.where(mask[:, :, None], frame, blurred_bg).astype(np.uint8)

        return output

    if current_mode == "ARTISTIC":
        small = cv2.resize(frame, None, fx=0.5, fy=0.5)
        small = cv2.bilateralFilter(small, 7, 45, 45)

        color = cv2.resize(small, (frame.shape[1], frame.shape[0]))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 5)

        edges = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            9,
            7
        )

        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        artistic = cv2.bitwise_and(color, edges_bgr)

        return artistic

    if current_mode == "B&W":
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if current_mode == "COOL":
        result = frame.copy()
        result[:, :, 0] = cv2.add(result[:, :, 0], 40)
        result[:, :, 1] = cv2.add(result[:, :, 1], 5)
        result[:, :, 2] = cv2.subtract(result[:, :, 2], 10)
        return result

    return frame.copy()


# -----------------------------
# Random Filters
# -----------------------------

filters = [
    "Sketch",
    "Black and White",
    "Warm",
    "Cool",
    "Vintage",
    "Comic Ink",
    "Kaleidoscope"
]


def apply_sketch_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    inverted = cv2.bitwise_not(gray)
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = cv2.bitwise_not(blurred)
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)

    return cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)


def apply_black_white_filter(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def apply_warm_filter(image):
    result = image.copy()

    result[:, :, 2] = cv2.add(result[:, :, 2], 35)
    result[:, :, 1] = cv2.add(result[:, :, 1], 15)

    return result


def apply_cool_filter(image):
    result = image.copy()

    result[:, :, 0] = cv2.add(result[:, :, 0], 40)
    result[:, :, 1] = cv2.add(result[:, :, 1], 5)
    result[:, :, 2] = cv2.subtract(result[:, :, 2], 10)

    return result


def apply_vintage_filter(image):
    result = image.copy().astype(np.float32)

    result[:, :, 0] *= 0.75
    result[:, :, 1] *= 0.95
    result[:, :, 2] *= 1.15

    result = np.clip(result, 0, 255).astype(np.uint8)
    result = cv2.GaussianBlur(result, (3, 3), 0)

    return result


def apply_comic_ink_filter(image):
    color = cv2.bilateralFilter(image, 9, 70, 70)
    color = cv2.convertScaleAbs(color, alpha=1.12, beta=8)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)

    edges = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        9,
        7
    )

    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    comic = cv2.bitwise_and(color, edges_bgr)

    return comic


def apply_kaleidoscope_filter(image):
    h, w = image.shape[:2]

    size = min(h, w)
    cx = w // 2
    cy = h // 2
    half = size // 2

    crop = image[cy - half:cy + half, cx - half:cx + half]
    quad = cv2.resize(crop, (w // 2, h // 2))

    top_left = quad
    top_right = cv2.flip(quad, 1)
    bottom_left = cv2.flip(quad, 0)
    bottom_right = cv2.flip(quad, -1)

    top = np.hstack((top_left, top_right))
    bottom = np.hstack((bottom_left, bottom_right))

    kaleidoscope = np.vstack((top, bottom))
    kaleidoscope = cv2.resize(kaleidoscope, (w, h))

    return kaleidoscope


def apply_random_filter(image):
    filter_name = random.choice(filters)

    if filter_name == "Sketch":
        result = apply_sketch_filter(image)
    elif filter_name == "Black and White":
        result = apply_black_white_filter(image)
    elif filter_name == "Warm":
        result = apply_warm_filter(image)
    elif filter_name == "Cool":
        result = apply_cool_filter(image)
    elif filter_name == "Vintage":
        result = apply_vintage_filter(image)
    elif filter_name == "Comic Ink":
        result = apply_comic_ink_filter(image)
    elif filter_name == "Kaleidoscope":
        result = apply_kaleidoscope_filter(image)
    else:
        result = image.copy()

    return result, filter_name


# -----------------------------
# Sticker
# -----------------------------

class Sticker:
    def __init__(self, x, y, size, sticker_type):
        self.x = x
        self.y = y
        self.size = size
        self.type = sticker_type
        self.is_dragging = False
        self.offset_x = 0
        self.offset_y = 0

    def get_box(self):
        return (
            self.x - self.size,
            self.y - self.size,
            self.x + self.size,
            self.y + self.size,
        )

    def contains(self, point):
        px, py = point
        x1, y1, x2, y2 = self.get_box()

        return x1 <= px <= x2 and y1 <= py <= y2

    def start_drag(self, point):
        px, py = point

        self.is_dragging = True
        self.offset_x = self.x - px
        self.offset_y = self.y - py

    def drag_to(self, point, frame_w, frame_h):
        px, py = point

        self.x = clamp(px + self.offset_x, self.size, frame_w - self.size)
        self.y = clamp(py + self.offset_y, self.size, frame_h - self.size)

    def stop_drag(self):
        self.is_dragging = False

    def is_over_trash(self, trash_rect):
        return boxes_overlap(self.get_box(), trash_rect)

    def draw(self, frame):
        if self.type == "heart":
            self.draw_heart(frame)
        elif self.type == "smile":
            self.draw_smile(frame)
        elif self.type == "star":
            self.draw_star(frame)
        elif self.type == "shark":
            self.draw_shark(frame)

        if self.is_dragging:
            cv2.circle(frame, (self.x, self.y), self.size + 9, YELLOW, 3)

    def draw_heart(self, frame):
        s = self.size
        points = []

        for t in np.linspace(0, 2 * math.pi, 80):
            x = 16 * math.sin(t) ** 3
            y = (
                13 * math.cos(t)
                - 5 * math.cos(2 * t)
                - 2 * math.cos(3 * t)
                - math.cos(4 * t)
            )

            px = int(self.x + x * s / 18)
            py = int(self.y - y * s / 18)

            points.append([px, py])

        points = np.array(points, np.int32)

        cv2.fillPoly(frame, [points], (50, 50, 255))
        cv2.polylines(frame, [points], True, WHITE, 3)

    def draw_smile(self, frame):
        s = self.size

        cv2.circle(frame, (self.x, self.y), s, (0, 220, 255), -1)
        cv2.circle(frame, (self.x, self.y), s, WHITE, 3)

        cv2.circle(frame, (self.x - s // 3, self.y - s // 4), 6, (30, 30, 30), -1)
        cv2.circle(frame, (self.x + s // 3, self.y - s // 4), 6, (30, 30, 30), -1)

        cv2.ellipse(
            frame,
            (self.x, self.y + s // 8),
            (s // 2, s // 3),
            0,
            15,
            165,
            (30, 30, 30),
            4,
        )

    def draw_star(self, frame):
        points = []

        for i in range(10):
            angle = i * math.pi / 5 - math.pi / 2
            radius = self.size if i % 2 == 0 else self.size // 2

            px = int(self.x + radius * math.cos(angle))
            py = int(self.y + radius * math.sin(angle))

            points.append([px, py])

        points = np.array(points, np.int32)

        cv2.fillPoly(frame, [points], (0, 215, 255))
        cv2.polylines(frame, [points], True, WHITE, 3)

    def draw_shark(self, frame):
        s = self.size

        body = np.array(
            [
                [self.x - s, self.y],
                [self.x - s // 3, self.y - s // 2],
                [self.x + s, self.y - s // 4],
                [self.x + s, self.y + s // 4],
                [self.x - s // 3, self.y + s // 2],
            ],
            np.int32,
        )

        tail = np.array(
            [
                [self.x - s, self.y],
                [self.x - s - s // 2, self.y - s // 2],
                [self.x - s - s // 3, self.y],
                [self.x - s - s // 2, self.y + s // 2],
            ],
            np.int32,
        )

        fin = np.array(
            [
                [self.x - s // 5, self.y - s // 3],
                [self.x + s // 8, self.y - s],
                [self.x + s // 4, self.y - s // 4],
            ],
            np.int32,
        )

        cv2.fillPoly(frame, [body], (170, 170, 170))
        cv2.fillPoly(frame, [tail], (130, 130, 130))
        cv2.fillPoly(frame, [fin], (110, 110, 110))

        cv2.polylines(frame, [body], True, WHITE, 2)
        cv2.polylines(frame, [tail], True, WHITE, 2)

        cv2.circle(frame, (self.x + s // 2, self.y - s // 8), 5, (0, 0, 0), -1)

        cv2.line(
            frame,
            (self.x + s // 2, self.y + s // 6),
            (self.x + s - 5, self.y + s // 8),
            (0, 0, 0),
            2,
        )


# -----------------------------
# Photo Piece
# -----------------------------

class PhotoPiece:
    def __init__(self, image, x, y):
        self.image = image
        self.x = x
        self.y = y
        self.h, self.w = image.shape[:2]
        self.is_dragging = False
        self.offset_x = 0
        self.offset_y = 0

    def get_box(self):
        return (
            self.x,
            self.y,
            self.x + self.w,
            self.y + self.h,
        )

    def contains(self, point):
        px, py = point
        x1, y1, x2, y2 = self.get_box()

        return x1 <= px <= x2 and y1 <= py <= y2

    def start_drag(self, point):
        px, py = point

        self.is_dragging = True
        self.offset_x = self.x - px
        self.offset_y = self.y - py

    def drag_to(self, point, frame_w, frame_h):
        px, py = point

        self.x = clamp(px + self.offset_x, 0, frame_w - self.w)
        self.y = clamp(py + self.offset_y, 0, frame_h - self.h)

    def stop_drag(self):
        self.is_dragging = False

    def is_over_trash(self, trash_rect):
        return boxes_overlap(self.get_box(), trash_rect)

    def draw(self, frame):
        fh, fw = frame.shape[:2]

        x1 = clamp(self.x, 0, fw - 1)
        y1 = clamp(self.y, 0, fh - 1)
        x2 = clamp(self.x + self.w, 0, fw)
        y2 = clamp(self.y + self.h, 0, fh)

        piece_w = x2 - x1
        piece_h = y2 - y1

        if piece_w <= 0 or piece_h <= 0:
            return

        frame[y1:y2, x1:x2] = self.image[0:piece_h, 0:piece_w]

        if self.is_dragging:
            cv2.rectangle(frame, (x1, y1), (x2, y2), YELLOW, 2)


# -----------------------------
# UI Drawing
# -----------------------------

def draw_clear_button(frame, progress=0.0):
    h, w = frame.shape[:2]

    button_w = 74
    button_h = 32
    margin = 14

    x1 = w - button_w - margin
    y1 = margin
    x2 = w - margin
    y2 = y1 + button_h

    progress = clamp(progress, 0.0, 1.0)

    cv2.rectangle(frame, (x1, y1), (x2, y2), BG_BLUE, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_BLUE_2, 1)

    fill_w = int((button_w - 6) * progress)

    if fill_w > 0:
        cv2.rectangle(
            frame,
            (x1 + 3, y2 - 6),
            (x1 + 3 + fill_w, y2 - 3),
            ACCENT_BLUE,
            -1
        )

    cv2.putText(
        frame,
        "CLEAR",
        (x1 + 11, y1 + 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        WHITE,
        1,
        cv2.LINE_AA,
    )

    return (x1, y1, x2, y2)


def draw_zoom_buttons(frame, minus_progress=0.0, plus_progress=0.0):
    h, w = frame.shape[:2]

    button_size = 46
    gap = 10
    margin = 16

    y1 = h - button_size - margin
    y2 = h - margin

    minus_x1 = margin
    minus_x2 = minus_x1 + button_size

    plus_x1 = minus_x2 + gap
    plus_x2 = plus_x1 + button_size

    minus_box = (minus_x1, y1, minus_x2, y2)
    plus_box = (plus_x1, y1, plus_x2, y2)

    def draw_button(box, label, progress):
        x1, y1, x2, y2 = box
        progress = clamp(progress, 0.0, 1.0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), BG_BLUE, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_BLUE_2, 1)

        fill_h = int((button_size - 6) * progress)

        if fill_h > 0:
            cv2.rectangle(
                frame,
                (x1 + 3, y2 - 3 - fill_h),
                (x2 - 3, y2 - 3),
                ACCENT_BLUE,
                -1
            )

        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
        text_x = x1 + (button_size - text_size[0]) // 2
        text_y = y1 + (button_size + text_size[1]) // 2

        cv2.putText(
            frame,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            WHITE,
            2,
            cv2.LINE_AA,
        )

    draw_button(minus_box, "-", minus_progress)
    draw_button(plus_box, "+", plus_progress)

    return minus_box, plus_box


def draw_trash_icon(frame, active=False):
    h, w = frame.shape[:2]

    size = 58
    right_margin = 44
    bottom_margin = 18

    x1 = w - size - right_margin
    y1 = h - size - bottom_margin
    x2 = x1 + size
    y2 = y1 + size

    bg_color = BG_BLUE
    border_color = TRASH_RED_BORDER

    if active:
        bg_color = (85, 70, 80)
        border_color = YELLOW

    cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), border_color, 2)

    body_x1 = x1 + 17
    body_y1 = y1 + 23
    body_x2 = x2 - 17
    body_y2 = y2 - 11

    cv2.rectangle(frame, (body_x1, body_y1), (body_x2, body_y2), border_color, 2)
    cv2.rectangle(frame, (body_x1 - 5, body_y1 - 8), (body_x2 + 5, body_y1 - 5), border_color, -1)
    cv2.rectangle(frame, (x1 + 24, y1 + 13), (x2 - 24, y1 + 16), border_color, -1)

    cv2.line(frame, (body_x1 + 6, body_y1 + 5), (body_x1 + 6, body_y2 - 4), border_color, 1)
    cv2.line(frame, ((body_x1 + body_x2) // 2, body_y1 + 5), ((body_x1 + body_x2) // 2, body_y2 - 4), border_color, 1)
    cv2.line(frame, (body_x2 - 6, body_y1 + 5), (body_x2 - 6, body_y2 - 4), border_color, 1)

    return (x1, y1, x2, y2)


def draw_countdown(frame, seconds_left):
    overlay = frame.copy()
    h, w = frame.shape[:2]

    center = (w // 2, h // 2)
    radius = 78

    cv2.circle(overlay, center, radius, (10, 10, 15), -1)
    cv2.addWeighted(overlay, 0.48, frame, 0.52, 0, frame)

    cv2.circle(frame, center, radius, WHITE, 3)

    text = str(seconds_left)
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 2.8
    thickness = 5

    text_size = cv2.getTextSize(text, font, scale, thickness)[0]

    text_x = center[0] - text_size[0] // 2
    text_y = center[1] + text_size[1] // 2

    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        scale,
        WHITE,
        thickness,
        cv2.LINE_AA,
    )


def draw_peace_hold_progress(frame, progress):
    h, w = frame.shape[:2]

    progress = clamp(progress, 0.0, 1.0)

    center = (w // 2, 76)
    radius = 28

    cv2.circle(frame, center, radius, BG_BLUE, -1)
    cv2.circle(frame, center, radius, WHITE, 1)

    angle = int(360 * progress)

    cv2.ellipse(
        frame,
        center,
        (radius, radius),
        -90,
        0,
        angle,
        ACCENT_BLUE,
        4,
    )

    cv2.line(frame, (center[0] - 7, center[1] - 5), (center[0] - 7, center[1] + 8), WHITE, 2)
    cv2.line(frame, (center[0] + 7, center[1] - 5), (center[0] + 7, center[1] + 8), WHITE, 2)


def draw_exit_hold_progress(frame, box, progress):
    x1, y1, x2, y2 = box[:4]

    progress = clamp(progress, 0.0, 1.0)

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    radius = 25

    angle = int(360 * progress)

    cv2.ellipse(
        frame,
        (cx, cy),
        (radius, radius),
        -90,
        0,
        angle,
        ACCENT_BLUE,
        4,
    )


def draw_sticker_menu(frame, exit_hold_progress=0.0):
    h, w = frame.shape[:2]

    box_size = 58
    gap = 12
    panel_h = 78

    menu_items = ["heart", "smile", "star", "shark", "none"]

    total_w = len(menu_items) * box_size + (len(menu_items) - 1) * gap
    start_x = (w - total_w) // 2
    menu_y = h - panel_h

    boxes = []

    cv2.rectangle(
        frame,
        (start_x - 12, menu_y - 6),
        (start_x + total_w + 12, menu_y + box_size + 8),
        BG_BLUE,
        -1,
    )

    cv2.rectangle(
        frame,
        (start_x - 12, menu_y - 6),
        (start_x + total_w + 12, menu_y + box_size + 8),
        ACCENT_BLUE_2,
        1,
    )

    for i, item in enumerate(menu_items):
        x1 = start_x + i * (box_size + gap)
        y1 = menu_y
        x2 = x1 + box_size
        y2 = y1 + box_size

        cv2.rectangle(frame, (x1, y1), (x2, y2), PANEL_BLUE, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), WHITE, 1)

        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2

        if item == "none":
            cv2.line(frame, (cx - 13, cy - 13), (cx + 13, cy + 13), WHITE, 3)
            cv2.line(frame, (cx + 13, cy - 13), (cx - 13, cy + 13), WHITE, 3)

            if exit_hold_progress > 0:
                draw_exit_hold_progress(frame, (x1, y1, x2, y2, item), exit_hold_progress)
        else:
            preview_size = 18

            if item == "heart":
                preview_size = 20

            preview = Sticker(cx, cy, preview_size, item)
            preview.draw(frame)

        boxes.append((x1, y1, x2, y2, item))

    return boxes


def draw_mode_label(frame, mode, zoom_level):
    x1, y1, x2, y2 = 12, 12, 140, 42

    cv2.rectangle(frame, (x1, y1), (x2, y2), BG_BLUE, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), ACCENT_BLUE_2, 1)

    label = mode

    if zoom_level > 1.01:
        label = f"{mode} {zoom_level:.1f}x"

    cv2.putText(
        frame,
        label,
        (x1 + 9, y1 + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        WHITE,
        1,
        cv2.LINE_AA,
    )


def draw_shape_ready_icon(frame, progress):
    center = (34, 74)
    radius = 16

    progress = clamp(progress, 0.0, 1.0)
    angle = int(360 * progress)

    cv2.circle(frame, center, radius, BG_BLUE, -1)
    cv2.circle(frame, center, radius, WHITE, 1)

    cv2.ellipse(
        frame,
        center,
        (radius, radius),
        -90,
        0,
        angle,
        ACCENT_BLUE,
        3,
    )


# -----------------------------
# Guide Window
# -----------------------------

def draw_guide_window():
    guide_w = 820
    guide_h = 760

    guide = np.zeros((guide_h, guide_w, 3), dtype=np.uint8)
    guide[:] = BG_BLUE

    cv2.rectangle(guide, (24, 24), (guide_w - 24, guide_h - 24), DARK_FILL, -1)
    cv2.rectangle(guide, (24, 24), (guide_w - 24, guide_h - 24), ACCENT_BLUE_2, 2)

    draw_centered_text(
        guide,
        "Welcome to AirCanvas",
        78,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.12,
        WHITE,
        2
    )

    draw_centered_text(
        guide,
        "Gesture-Controlled Creative Capture System",
        112,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        MID_TEXT,
        1
    )

    sections = [
        ("Swipe", "Move left or right with your index finger to switch camera modes."),
        ("Zoom", "Pinch and hold the plus or minus button for 0.5 seconds to zoom in or out."),
        ("Thumbs Up", "Hold a clear thumbs up briefly to take a picture."),
        ("Open Palm + Triangle", "Show an open palm, then draw a triangle for a random filter capture."),
        ("Open Palm + Square", "Show an open palm, then draw a square to split the scene into 16 pieces."),
        ("Peace Sign", "Hold a peace sign to open the sticker menu."),
        ("Pinch", "Pinch to select, drag, and place stickers or photo pieces."),
        ("Trash", "Drag an item onto the trash icon and release to delete it."),
        ("Clear", "Pinch and hold CLEAR for 3 seconds to reset the screen."),
        ("Quit", "Press Q in the camera window to close the app."),
    ]

    y = 150
    bullet_x = 64
    text_x = 92

    title_font_scale = 0.58
    body_font_scale = 0.46
    line_gap = 19
    section_gap = 8

    for title, description in sections:
        cv2.circle(guide, (bullet_x, y - 6), 6, ACCENT_BLUE, -1)

        cv2.putText(
            guide,
            title,
            (text_x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            title_font_scale,
            WHITE,
            2,
            cv2.LINE_AA,
        )

        wrapped = wrap_text(description, 76)
        line_y = y + 22

        for line in wrapped:
            cv2.putText(
                guide,
                line,
                (text_x, line_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                body_font_scale,
                LIGHT_TEXT,
                1,
                cv2.LINE_AA,
            )

            line_y += line_gap

        y = line_y + section_gap

    return guide


# -----------------------------
# Saving / Split
# -----------------------------

def save_photo(image):
    filename = f"captured_photos/thumbs_up_capture_{int(time.time())}.jpg"
    cv2.imwrite(filename, image)

    print(f"Saved: {filename}")


def save_triangle_photo(image, filter_name):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_filter_name = filter_name.lower().replace(" ", "_")

    filename = os.path.join(
        "triangle_random_filtered_photos",
        f"triangle_{safe_filter_name}_{timestamp}.jpg"
    )

    cv2.imwrite(filename, image)

    print(f"Saved: {filename}")


def split_scene_into_pieces(scene):
    h, w = scene.shape[:2]

    source = scene.copy()

    crop_w = int(w * 0.76)
    crop_h = int(h * 0.76)

    crop_x = (w - crop_w) // 2
    crop_y = (h - crop_h) // 2

    crop = source[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]

    rows = 4
    cols = 4

    piece_w = crop_w // cols
    piece_h = crop_h // rows

    start_x = (w - crop_w) // 2
    start_y = (h - crop_h) // 2

    pieces = []

    for row in range(rows):
        for col in range(cols):
            x_start = col * piece_w
            y_start = row * piece_h

            piece_img = crop[
                y_start:y_start + piece_h,
                x_start:x_start + piece_w,
            ].copy()

            cv2.rectangle(
                piece_img,
                (0, 0),
                (piece_img.shape[1] - 1, piece_img.shape[0] - 1),
                (230, 230, 230),
                1,
            )

            x = start_x + col * piece_w
            y = start_y + row * piece_h

            pieces.append(PhotoPiece(piece_img, x, y))

    return pieces


RESULT_WINDOWS = [
    "AirCanvas Photo Capture",
    "AirCanvas Filter Result",
]


def close_result_windows(except_window=None):
    for window_name in RESULT_WINDOWS:
        if window_name == except_window:
            continue

        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass


def show_result_window(window_name, image):
    close_result_windows(except_window=window_name)
    cv2.imshow(window_name, image)


def move_object_to_front(obj, stickers, photo_pieces):
    if obj in stickers:
        stickers.remove(obj)
        stickers.append(obj)
    elif obj in photo_pieces:
        photo_pieces.remove(obj)
        photo_pieces.append(obj)


# -----------------------------
# Main
# -----------------------------

def main():
    modes = ["PHOTO", "PORTRAIT", "ARTISTIC", "B&W", "COOL"]
    mode_index = 0

    point_history = deque(maxlen=12)
    last_swipe_time = 0

    swipe_cooldown = 0.75
    min_swipe_distance = 0.11
    max_vertical_drift = 0.22

    zoom_level = 1.0
    min_zoom = 1.0
    max_zoom = 2.0
    zoom_step = 0.2

    zoom_minus_hold_start = None
    zoom_plus_hold_start = None
    ZOOM_BUTTON_HOLD_SECONDS = 0.5

    photo_gesture_hold_start = None
    shape_path = []

    stickers = []
    photo_pieces = []
    active_object = None

    menu_until = 0
    exit_hold_start = None

    cursor_smooth_x = None
    cursor_smooth_y = None

    pinch_hold_start = None
    last_pinch_point = None

    clear_hold_start = None

    peace_hold_start = None
    peace_menu_opened_until_release = False
    last_peace_menu_time = 0

    open_palm_hold_start = None
    shape_ready_until = 0

    photo_countdown_end = 0
    triangle_countdown_end = 0
    split_countdown_end = 0

    last_photo_time = 0
    last_shape_action_time = 0

    PINCH_START_DISTANCE = 36
    PINCH_RELEASE_DISTANCE = 68
    PINCH_CONFIRM_SECONDS = 0.22
    PINCH_STABILITY_PIXELS = 28
    SMOOTHING = 0.35

    MENU_DURATION = 4.0
    CLEAR_HOLD_SECONDS = 3.0
    PEACE_HOLD_SECONDS = 1.0
    PEACE_MENU_COOLDOWN = 1.5
    EXIT_HOLD_SECONDS = 1.0

    OPEN_PALM_HOLD_SECONDS = 0.6
    SHAPE_READY_SECONDS = 5.0
    COUNTDOWN_SECONDS = 3.0
    THUMBS_UP_HOLD_SECONDS = 0.65

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_selfie_segmentation = mp.solutions.selfie_segmentation

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    guide = draw_guide_window()
    cv2.imshow("AirCanvas Guide", guide)

    selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        model_complexity=1,
        min_detection_confidence=0.72,
        min_tracking_confidence=0.72,
    ) as hands:

        while True:
            ret, raw_frame = cap.read()

            if not ret:
                break

            raw_frame = cv2.flip(raw_frame, 1)
            raw_frame = apply_zoom(raw_frame, zoom_level)

            h, w, _ = raw_frame.shape
            current_time = time.time()

            rgb = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            countdown_active = (
                photo_countdown_end > 0
                or triangle_countdown_end > 0
                or split_countdown_end > 0
            )

            base_scene = apply_camera_mode(raw_frame, modes[mode_index], selfie_segmentation)
            scene = base_scene.copy()

            for piece in photo_pieces:
                piece.draw(scene)

            for sticker in stickers:
                sticker.draw(scene)

            scene_without_ui = scene.copy()
            frame = scene.copy()

            clear_progress = 0.0

            if clear_hold_start is not None:
                clear_progress = (current_time - clear_hold_start) / CLEAR_HOLD_SECONDS

            clear_rect = draw_clear_button(frame, clear_progress)

            preview_trash_rect = (
                w - 58 - 44,
                h - 58 - 18,
                w - 44,
                h - 18,
            )

            trash_active = False

            if active_object is not None and active_object.is_over_trash(preview_trash_rect):
                trash_active = True

            trash_rect = draw_trash_icon(frame, trash_active)

            draw_mode_label(frame, modes[mode_index], zoom_level)

            zoom_minus_progress = 0.0
            zoom_plus_progress = 0.0

            if zoom_minus_hold_start is not None:
                zoom_minus_progress = (current_time - zoom_minus_hold_start) / ZOOM_BUTTON_HOLD_SECONDS

            if zoom_plus_hold_start is not None:
                zoom_plus_progress = (current_time - zoom_plus_hold_start) / ZOOM_BUTTON_HOLD_SECONDS

            zoom_minus_rect, zoom_plus_rect = draw_zoom_buttons(
                frame,
                zoom_minus_progress,
                zoom_plus_progress
            )

            exit_hold_progress = 0.0

            if exit_hold_start is not None:
                exit_hold_progress = (current_time - exit_hold_start) / EXIT_HOLD_SECONDS

            menu_boxes = []

            if current_time < menu_until:
                menu_boxes = draw_sticker_menu(frame, exit_hold_progress)

            if current_time < shape_ready_until and not countdown_active:
                progress = (shape_ready_until - current_time) / SHAPE_READY_SECONDS
                draw_shape_ready_icon(frame, progress)

            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                lm = hand_landmarks.landmark

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style(),
                )

                index_point = (int(lm[8].x * w), int(lm[8].y * h))
                thumb_point = (int(lm[4].x * w), int(lm[4].y * h))

                raw_pinch_point = (
                    (index_point[0] + thumb_point[0]) // 2,
                    (index_point[1] + thumb_point[1]) // 2,
                )

                if cursor_smooth_x is None:
                    cursor_smooth_x, cursor_smooth_y = raw_pinch_point
                else:
                    cursor_smooth_x = int(
                        (1 - SMOOTHING) * cursor_smooth_x
                        + SMOOTHING * raw_pinch_point[0]
                    )

                    cursor_smooth_y = int(
                        (1 - SMOOTHING) * cursor_smooth_y
                        + SMOOTHING * raw_pinch_point[1]
                    )

                pinch_point = (cursor_smooth_x, cursor_smooth_y)
                pinch_distance = distance(index_point, thumb_point)

                open_palm = is_open_palm(hand_landmarks)
                peace_sign = is_peace_sign(hand_landmarks)
                index_only = is_index_only(hand_landmarks)
                index_up = is_index_up(hand_landmarks)

                if active_object is None:
                    is_pinching = pinch_distance < PINCH_START_DISTANCE
                else:
                    is_pinching = pinch_distance < PINCH_RELEASE_DISTANCE

                cv2.circle(frame, index_point, 7, (0, 255, 0), -1)
                cv2.circle(frame, thumb_point, 7, (255, 0, 255), -1)
                cv2.circle(frame, pinch_point, 8, YELLOW, -1)
                cv2.line(frame, index_point, thumb_point, WHITE, 2)

                if not countdown_active:
                    shape_ready = current_time < shape_ready_until

                    if (
                        not shape_ready
                        and not open_palm
                        and open_palm_hold_start is None
                        and index_up
                        and not is_pinching
                        and active_object is None
                        and current_time >= menu_until
                        and len(shape_path) == 0
                    ):
                        point_history.append((lm[8].x, lm[8].y))

                        if len(point_history) >= 6:
                            dx, dy = averaged_swipe_delta(point_history)

                            horizontal_swipe = (
                                abs(dx) > min_swipe_distance
                                and abs(dy) < max_vertical_drift
                                and abs(dx) > abs(dy) * 1.15
                            )

                            if horizontal_swipe and current_time - last_swipe_time >= swipe_cooldown:
                                if dx > 0:
                                    mode_index = (mode_index + 1) % len(modes)
                                else:
                                    mode_index = (mode_index - 1) % len(modes)

                                last_swipe_time = current_time
                                point_history.clear()
                                photo_gesture_hold_start = None
                                shape_path.clear()
                    else:
                        point_history.clear()

                    if open_palm and not is_pinching and active_object is None:
                        if open_palm_hold_start is None:
                            open_palm_hold_start = current_time

                        if current_time - open_palm_hold_start >= OPEN_PALM_HOLD_SECONDS:
                            shape_ready_until = current_time + SHAPE_READY_SECONDS
                            shape_path.clear()
                            photo_gesture_hold_start = None
                            point_history.clear()
                    else:
                        open_palm_hold_start = None

                    shape_ready = current_time < shape_ready_until

                    thumbs_up = is_thumbs_up(hand_landmarks)

                    if (
                        thumbs_up
                        and not shape_ready
                        and not is_pinching
                        and active_object is None
                        and current_time >= menu_until
                        and current_time - last_photo_time > 4
                    ):
                        if photo_gesture_hold_start is None:
                            photo_gesture_hold_start = current_time

                        thumb_progress = (current_time - photo_gesture_hold_start) / THUMBS_UP_HOLD_SECONDS
                        draw_thumbs_up_hold_progress(frame, thumb_progress)

                        if thumb_progress >= 1.0:
                            photo_countdown_end = current_time + COUNTDOWN_SECONDS
                            photo_gesture_hold_start = None
                            shape_path.clear()
                            point_history.clear()
                    else:
                        photo_gesture_hold_start = None

                    if (
                        shape_ready
                        and index_only
                        and not is_pinching
                        and active_object is None
                        and current_time >= menu_until
                    ):
                        if not shape_path or distance(shape_path[-1], index_point) >= 5:
                            shape_path.append(index_point)

                        if len(shape_path) > 120:
                            shape_path = shape_path[-120:]

                        if current_time - last_shape_action_time > 3:
                            if is_square_motion(shape_path):
                                split_countdown_end = current_time + COUNTDOWN_SECONDS
                                shape_path.clear()
                                photo_gesture_hold_start = None
                                point_history.clear()
                                shape_ready_until = 0
                                menu_until = 0
                                last_shape_action_time = current_time

                            elif is_triangle_motion(shape_path):
                                triangle_countdown_end = current_time + COUNTDOWN_SECONDS
                                shape_path.clear()
                                photo_gesture_hold_start = None
                                point_history.clear()
                                shape_ready_until = 0
                                menu_until = 0
                                last_shape_action_time = current_time
                    else:
                        if not index_only and not open_palm and active_object is None and not is_pinching:
                            shape_path.clear()

                    if (
                        peace_sign
                        and not is_pinching
                        and active_object is None
                        and current_time >= menu_until
                        and not peace_menu_opened_until_release
                        and current_time - last_peace_menu_time > PEACE_MENU_COOLDOWN
                    ):
                        if peace_hold_start is None:
                            peace_hold_start = current_time

                        peace_progress = (current_time - peace_hold_start) / PEACE_HOLD_SECONDS
                        draw_peace_hold_progress(frame, peace_progress)

                        if peace_progress >= 1.0:
                            menu_until = current_time + MENU_DURATION
                            shape_path.clear()
                            photo_gesture_hold_start = None
                            point_history.clear()
                            peace_hold_start = None
                            peace_menu_opened_until_release = True
                            last_peace_menu_time = current_time
                    elif not peace_sign:
                        peace_hold_start = None
                        peace_menu_opened_until_release = False

                    if is_pinching and point_in_box(pinch_point, zoom_minus_rect):
                        if zoom_minus_hold_start is None:
                            zoom_minus_hold_start = current_time

                        zoom_plus_hold_start = None

                        if current_time - zoom_minus_hold_start >= ZOOM_BUTTON_HOLD_SECONDS:
                            zoom_level = max(min_zoom, zoom_level - zoom_step)
                            zoom_minus_hold_start = None

                    elif is_pinching and point_in_box(pinch_point, zoom_plus_rect):
                        if zoom_plus_hold_start is None:
                            zoom_plus_hold_start = current_time

                        zoom_minus_hold_start = None

                        if current_time - zoom_plus_hold_start >= ZOOM_BUTTON_HOLD_SECONDS:
                            zoom_level = min(max_zoom, zoom_level + zoom_step)
                            zoom_plus_hold_start = None

                    else:
                        zoom_minus_hold_start = None
                        zoom_plus_hold_start = None

                    if is_pinching and point_in_box(pinch_point, clear_rect):
                        if clear_hold_start is None:
                            clear_hold_start = current_time

                        if current_time - clear_hold_start >= CLEAR_HOLD_SECONDS:
                            stickers.clear()
                            photo_pieces.clear()
                            active_object = None

                            shape_path.clear()
                            photo_gesture_hold_start = None
                            point_history.clear()

                            menu_until = 0
                            shape_ready_until = 0

                            clear_hold_start = None
                            pinch_hold_start = None
                            last_pinch_point = None
                            zoom_minus_hold_start = None
                            zoom_plus_hold_start = None
                            exit_hold_start = None
                            peace_hold_start = None
                            open_palm_hold_start = None
                            peace_menu_opened_until_release = False

                            split_countdown_end = 0
                            photo_countdown_end = 0
                            triangle_countdown_end = 0

                            mode_index = 0
                            zoom_level = 1.0

                            try:
                                cv2.destroyWindow("AirCanvas Filter Result")
                            except cv2.error:
                                pass

                            try:
                                cv2.destroyWindow("AirCanvas Photo Capture")
                            except cv2.error:
                                pass
                    else:
                        clear_hold_start = None

                    if is_pinching:
                        if (
                            point_in_box(pinch_point, clear_rect)
                            or point_in_box(pinch_point, zoom_minus_rect)
                            or point_in_box(pinch_point, zoom_plus_rect)
                        ):
                            pinch_hold_start = None
                            last_pinch_point = pinch_point
                        else:
                            shape_path.clear()
                            photo_gesture_hold_start = None
                            point_history.clear()

                            stable_pinch = (
                                last_pinch_point is None
                                or distance(pinch_point, last_pinch_point) <= PINCH_STABILITY_PIXELS
                            )

                            # Do not start dragging from an open palm. This avoids false grabs
                            # when the hand is simply moving across the camera.
                            allowed_to_start_grab = not open_palm

                            if active_object is None and allowed_to_start_grab and stable_pinch:
                                if pinch_hold_start is None:
                                    pinch_hold_start = current_time

                                pinch_confirm_progress = (current_time - pinch_hold_start) / PINCH_CONFIRM_SECONDS
                                pinch_confirm_progress = clamp(pinch_confirm_progress, 0.0, 1.0)

                                cv2.circle(frame, pinch_point, 13, ACCENT_BLUE_2, 1)
                                cv2.ellipse(
                                    frame,
                                    pinch_point,
                                    (17, 17),
                                    -90,
                                    0,
                                    int(360 * pinch_confirm_progress),
                                    ACCENT_BLUE,
                                    3,
                                )

                                if current_time - pinch_hold_start >= PINCH_CONFIRM_SECONDS:
                                    selected_from_menu = False

                                    if current_time < menu_until:
                                        for box in menu_boxes:
                                            if point_in_box(pinch_point, box):
                                                sticker_type = box[4]

                                                if sticker_type == "none":
                                                    if exit_hold_start is None:
                                                        exit_hold_start = current_time

                                                    if current_time - exit_hold_start >= EXIT_HOLD_SECONDS:
                                                        menu_until = 0
                                                        exit_hold_start = None

                                                    selected_from_menu = True
                                                    break

                                                exit_hold_start = None

                                                new_sticker = Sticker(
                                                    pinch_point[0],
                                                    pinch_point[1],
                                                    45,
                                                    sticker_type,
                                                )

                                                stickers.append(new_sticker)
                                                active_object = new_sticker
                                                active_object.start_drag(pinch_point)
                                                selected_from_menu = True
                                                pinch_hold_start = None

                                                menu_until = 0
                                                break

                                        if not selected_from_menu:
                                            exit_hold_start = None
                                    else:
                                        exit_hold_start = None

                                    if not selected_from_menu:
                                        for sticker in reversed(stickers):
                                            if sticker.contains(pinch_point):
                                                active_object = sticker
                                                move_object_to_front(active_object, stickers, photo_pieces)
                                                active_object.start_drag(pinch_point)
                                                pinch_hold_start = None
                                                break

                                    if active_object is None and not selected_from_menu:
                                        for piece in reversed(photo_pieces):
                                            if piece.contains(pinch_point):
                                                active_object = piece
                                                move_object_to_front(active_object, stickers, photo_pieces)
                                                active_object.start_drag(pinch_point)
                                                pinch_hold_start = None
                                                break

                            elif active_object is None:
                                pinch_hold_start = None

                            if active_object is not None:
                                active_object.drag_to(pinch_point, w, h)

                            last_pinch_point = pinch_point

                    else:
                        exit_hold_start = None
                        pinch_hold_start = None
                        last_pinch_point = None

                        if active_object is not None:
                            object_touches_trash = active_object.is_over_trash(trash_rect)
                            cursor_over_trash = point_in_box(pinch_point, trash_rect)

                            should_delete = object_touches_trash or cursor_over_trash

                            active_object.stop_drag()

                            if should_delete:
                                if active_object in stickers:
                                    stickers.remove(active_object)
                                elif active_object in photo_pieces:
                                    photo_pieces.remove(active_object)
                            else:
                                move_object_to_front(active_object, stickers, photo_pieces)

                            active_object = None

            else:
                cursor_smooth_x = None
                cursor_smooth_y = None
                pinch_hold_start = None
                last_pinch_point = None
                point_history.clear()
                photo_gesture_hold_start = None
                shape_path.clear()
                clear_hold_start = None
                zoom_minus_hold_start = None
                zoom_plus_hold_start = None
                peace_hold_start = None
                exit_hold_start = None
                open_palm_hold_start = None
                peace_menu_opened_until_release = False

                if active_object is not None:
                    active_object.stop_drag()
                    active_object = None

            if photo_countdown_end > 0:
                remaining = photo_countdown_end - current_time

                if remaining > 0:
                    draw_countdown(frame, math.ceil(remaining))
                else:
                    captured_photo = scene_without_ui.copy()
                    save_photo(captured_photo)
                    show_result_window("AirCanvas Photo Capture", captured_photo)

                    photo_countdown_end = 0
                    last_photo_time = current_time

            if triangle_countdown_end > 0:
                remaining = triangle_countdown_end - current_time

                if remaining > 0:
                    draw_countdown(frame, math.ceil(remaining))
                else:
                    filtered_result, filter_name = apply_random_filter(scene_without_ui)
                    save_triangle_photo(filtered_result, filter_name)
                    show_result_window("AirCanvas Filter Result", filtered_result)
                    triangle_countdown_end = 0

            if split_countdown_end > 0:
                remaining = split_countdown_end - current_time

                if remaining > 0:
                    draw_countdown(frame, math.ceil(remaining))
                else:
                    photo_pieces = split_scene_into_pieces(scene_without_ui)
                    active_object = None
                    shape_path.clear()
                    photo_gesture_hold_start = None
                    split_countdown_end = 0

            cv2.imshow("AirCanvas Integrated System", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()
    selfie_segmentation.close()


if __name__ == "__main__":
    main()