import cv2
import numpy as np

# Constants
ALPHA = 0.5
FONT = cv2.FONT_HERSHEY_PLAIN
TEXT_SCALE = 1.0
TEXT_THICKNESS = 1
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def show_fps(img, fps):
    """Draw fps number at top-left corner of the image."""
    font = cv2.FONT_HERSHEY_PLAIN
    line = cv2.LINE_AA
    fps_text = f"FPS: {fps:.2f}"
    cv2.putText(img, fps_text, (11, 20), font, 1.0, (32, 32, 32), 4, line)
    cv2.putText(img, fps_text, (10, 20), font, 1.0, (240, 240, 240), 1, line)
    return img


def gen_colors(num_colors):
    """Generate different colors.

    # Arguments
      num_colors: total number of colors/classes.

    # Output
      bgrs: a list of (B, G, R) tuples which correspond to each of
            the colors/classes.
    """
    import colorsys
    import random

    hsvs = [[float(x) / num_colors, 1.0, 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = [list(colorsys.hsv_to_rgb(*x)) for x in hsvs]
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255)) for rgb in rgbs]
    return bgrs


def draw_boxed_text(img, text, topleft, color):
    """Draw a transluent boxed text in white, overlayed on top of a
    colored patch surrounded by a black border. FONT, TEXT_SCALE,
    TEXT_THICKNESS and ALPHA values are constants (fixed) as defined
    on top.

    # Arguments
      img: the input image as a numpy array.
      text: the text to be drawn.
      topleft: XY coordinate of the topleft corner of the boxed text.
      color: color of the patch, i.e. background of the text.

    # Output
      img: note the original image is modified inplace.
    """
    assert img.dtype == np.uint8
    img_h, img_w, _ = img.shape
    if topleft[0] >= img_w or topleft[1] >= img_h:
        return img
    margin = 3
    size = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICKNESS)
    w = size[0][0] + margin * 2
    h = size[0][1] + margin * 2
    # the patch is used to draw boxed text
    patch = np.zeros((h, w, 3), dtype=np.uint8)
    patch[...] = color
    cv2.putText(
        patch,
        text,
        (margin + 1, h - margin - 2),
        FONT,
        TEXT_SCALE,
        WHITE,
        thickness=TEXT_THICKNESS,
        lineType=cv2.LINE_8,
    )
    cv2.rectangle(patch, (0, 0), (w - 1, h - 1), BLACK, thickness=1)
    w = min(w, img_w - topleft[0])  # clip overlay at image boundary
    h = min(h, img_h - topleft[1])
    # Overlay the boxed text onto region of interest (roi) in img
    roi = img[topleft[1] : topleft[1] + h, topleft[0] : topleft[0] + w, :]
    cv2.addWeighted(patch[0:h, 0:w, :], ALPHA, roi, 1 - ALPHA, 0, roi)
    return img


class BBoxVisualization:
    """BBoxVisualization class implements nice drawing of boudning boxes.

    # Arguments
      cls_dict: a dictionary used to translate class id to its name.
    """

    def __init__(self, cls_dict):
        self.cls_dict = cls_dict
        self.colors = gen_colors(len(cls_dict))

    def draw_bboxes(self, img, boxes, confs, clss):
        """Draw detected bounding boxes on the original image."""
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            color = self.colors[cl]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min + 2, 0), max(y_min + 2, 0))
            cls_name = self.cls_dict.get(cl, f"CLS{cl}")
            txt = f"{cls_name} {cf:.2f}"
            img = draw_boxed_text(img, txt, txt_loc, color)
        return img


class Displayer:
    def __init__(self):
        self.window_name = "Test"
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, 1)
        self.break_keys = [ord("q"), ord("Q"), 27]

    def key_trigger_event(self) -> bool:
        key = cv2.waitKey(1)

        # Esc
        if key in self.break_keys:
            return False
        # F12
        elif key in [201, ord("t"), ord("T")]:
            stats = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN)
            new_state = (
                cv2.WINDOW_NORMAL
                if stats == cv2.WINDOW_FULLSCREEN
                else cv2.WINDOW_FULLSCREEN
            )
            cv2.setWindowProperty(self.window_name, cv2.WND_PROP_FULLSCREEN, new_state)

        # Close Windows
        if cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1:
            return False

        return True

    def show(self, frame):
        cv2.imshow(self.window_name, frame)
        return self.key_trigger_event()

    def release(self):
        cv2.destroyAllWindows()
