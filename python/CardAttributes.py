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
        card_area = card_image.shape[0] * card_image.shape[1]
        contours = [contour for contour in contours
                    if (card_area / 20) < ip.cv2.contourArea(contour)]
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
        # self.find_contours(card_image)
        # card_image_mask = np.zeros((card_image.shape[0], card_image.shape[1], 1))
        # ip.cv2.drawContours(card_image_mask, self.contours, -1, (255, 0, 0), -1)
        # color = ip.cv2.mean(card_image, (card_image_mask / 255.0).astype("uint8"))
        # color = ip.cv2.cvtColor(np.uint8([[color]]), ip.cv2.COLOR_BGR2HSV)
        # color = color[0][0]
        # color = [color[0].astype('float') / 180.0 * 360.0, 255, 255]
        #
        # color2 = color.copy()
        # color[0] = color[0].astype('float') / 180.0 * 360.0
        # color[1] = 255.0
        # color[2] = 255.0
        # color = ip.cv2.cvtColor(np.uint8([[color]]), ip.cv2.COLOR_HSV2RGB)
        # color = ip.cv2.cvtColor(color, ip.cv2.COLOR_RGB2BGR)[0][0]
        # color = color.astype('float')
        # swap = color[0]
        # color[0] = color[1]
        # color[1] = swap
        #
        # card_image = ip.cv2.GaussianBlur(card_image, (15, 15), 0)
        #
        # card_mask = (1 - ip.cv2.dilate(card_image_mask, np.ones((10, 10), np.uint8), iterations=1) / 255.0).astype("uint8")
        # card_color = ip.cv2.mean(card_image, card_mask)
        # infill_mask = (ip.cv2.erode(card_image_mask, np.ones((20, 20), np.uint8), iterations=1) / 255.0).astype("uint8")
        #
        # infill_color = ip.cv2.mean(card_image, infill_mask)
        # infill_color2 = ip.cv2.cvtColor(np.uint8([[color2]]), ip.cv2.COLOR_HSV2BGR)[0][0]
        #
        # dA = np.absolute(np.subtract(np.int16([infill_color[:3]]), np.int16([card_color[:3]])))
        # dB = np.absolute(np.subtract(np.int16([infill_color2[:3]]), np.int16([card_color[:3]])))
        # #self.data = np.max(np.divide(dA.astype("float"), dB.astype("float"))
        self.data = -1


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
        self.data = map(np.round, [ip.np.mean(ap_ratio) ** 0.5 * 10, ip.np.mean(points) ** 0.5 * 10])


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
        card_image_mask = np.zeros((card_image.shape[0], card_image.shape[1], 1))
        ip.cv2.drawContours(card_image_mask, self.contours, -1, (255, 0, 0), -1)
        color = ip.cv2.mean(card_image, (card_image_mask / 255.0).astype("uint8"))
        color = ip.cv2.cvtColor(np.uint8([[color]]), ip.cv2.COLOR_BGR2HSV)
        color = color[0][0]
        theta = color[0] * 2  # H will be in [0, 180] convert to [0, 360]
        theta *= (np.pi / 180)  # convert to radians
        self.data = [np.cos(theta) / 2, np.sin(theta) / 2]  # reduce by 2 to make colors more likely to cluster


