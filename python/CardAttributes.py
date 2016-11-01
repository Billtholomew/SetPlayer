import numpy as np

import image_processing as ip


class AbstractAttribute:

    def __init__(self, name=None, default_data=None):
        self.name = name
        self.data = default_data
        self.contours = None
        self.classification = None

    def classify(self, classification):
        self.classification = classification

    def find_contours(self, card_image):
        card_image_mask = (255 - ip.im_mask(card_image))
        contours, _ = ip.cv2.findContours(card_image_mask, ip.cv2.RETR_EXTERNAL, ip.cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(lambda contour: (card_image_mask.size / 20) < ip.cv2.contourArea(contour), contours)
        self.contours = contours

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.classification == other.classification
        else:
            return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        if self.classification is None:
            raise ValueError('Card not classified')
        return self.classification


class Infill(AbstractAttribute):

    def __init__(self, card_image=None):
        AbstractAttribute.__init__(self, name='infill', default_data=None)
        if card_image is not None:
            self.parse_image(card_image)

    def parse_image(self, card_image):
        self.find_contours(card_image)
        card_image_mask = np.zeros((card_image.shape[0], card_image.shape[1]), dtype=np.uint8)
        ip.cv2.drawContours(card_image_mask, self.contours, -1, 255, -1)
        edges = ip.auto_canny(ip.bgr2gray(card_image))
        #ip.cv2.imshow("TEST", edges)
        #ip.cv2.waitKey(0)
        card_image_mask = ip.cv2.erode(card_image_mask, np.ones((15, 15), np.uint8), iterations=1)
        color, std_dev = ip.cv2.meanStdDev(ip.bgr2gray(card_image), mask=card_image_mask.astype("uint8"))
        color = color[0][0]
        std_dev = std_dev[0][0]
        self.data = [color, std_dev]


class Shape(AbstractAttribute):

    def __init__(self, card_image=None):
        AbstractAttribute.__init__(self, name='shape', default_data=None)
        if card_image is not None:
            self.parse_image(card_image)

    def parse_image(self, card_image):
        self.find_contours(card_image)
        ap_ratio = []
        hw_ratio = []
        points = []
        for contour in self.contours:
            area = ip.cv2.contourArea(contour)
            perimeter = ip.cv2.arcLength(contour, True)
            _, _, w, h = ip.cv2.boundingRect(contour)
            ap_ratio.append(float(area) / float(perimeter))
            hw_ratio.append(float(h) / float(w))
            epsilon = 0.01*ip.cv2.arcLength(contour, True)
            simple_contour = ip.cv2.approxPolyDP(contour, epsilon, True)
            points.append(len(simple_contour))
        self.data = [ip.np.mean(ap_ratio) ** 0.5 * 10, ip.np.mean(points) ** 0.5 * 10]


class Count(AbstractAttribute):

    def __init__(self, card_image=None):
        AbstractAttribute.__init__(self, name='count', default_data=None)
        if card_image is not None:
            self.parse_image(card_image)

    def parse_image(self, card_image):
        self.find_contours(card_image)
        self.data = (len(self.contours), len(self.contours) ** 2)


class Color(AbstractAttribute):

    def __init__(self, card_image=None):
        AbstractAttribute.__init__(self, name='color', default_data={})
        if card_image is not None:
            self.parse_image(card_image)

    def parse_image(self, card_image):
        self.find_contours(card_image)
        card_image_mask = np.zeros((card_image.shape[0], card_image.shape[1]), dtype=np.uint8)
        ip.cv2.drawContours(card_image_mask, self.contours, -1, 255, -1)
        color = ip.cv2.mean(card_image, mask=card_image_mask)

        color = ip.cv2.cvtColor(np.uint8([[color]]), ip.cv2.COLOR_BGR2HSV)
        color = color[0][0]
        theta = color[0] * 2  # H will be in [0, 180] convert to [0, 360]
        theta *= (np.pi / 180)  # convert to radians
        self.data = [np.cos(theta) / 2, np.sin(theta) / 2]  # reduce by 2 to make colors more likely to cluster
