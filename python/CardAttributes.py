import numpy as np

import image_processing as ip


class AbstractAttribute:

    def __init__(self, name=None, default_data=None):
        self.name = name
        self.data = default_data
        self.class_data = None
        self.contours = None
        self.classification = None

    def classify(self, classification, centroid):
        self.classification = classification
        self.class_data = centroid

    def find_contours(self, card_image):
        shape_mask = (255 - ip.im_mask(card_image, sigma=1.5))
        contours, _ = ip.cv2.findContours(shape_mask, ip.cv2.RETR_EXTERNAL, ip.cv2.CHAIN_APPROX_SIMPLE)
        contours = filter(lambda contour: (shape_mask.size / 20) < ip.cv2.contourArea(contour), contours)
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

        shape_mask = np.zeros((card_image.shape[0], card_image.shape[1]))
        ip.cv2.drawContours(shape_mask, self.contours, -1, 1, -1)
        shape_mask = ip.cv2.erode(shape_mask, np.ones((16, 16), np.uint8), iterations=1)
        card_image_hsv = ip.cv2.cvtColor(card_image, ip.cv2.COLOR_BGR2HSV)

        shape_hsv = card_image_hsv[shape_mask == 1]
        shape_saturation_mean = np.mean(shape_hsv[:, 1])
        shape_saturation_median = np.median(shape_hsv[:, 1])

        self.data = [shape_saturation_mean ** 0.5, np.log2(shape_saturation_median)]


class Shape(AbstractAttribute):

    def __init__(self, card_image=None):
        AbstractAttribute.__init__(self, name='shape', default_data=None)
        if card_image is not None:
            self.parse_image(card_image)

    def parse_image(self, card_image):
        self.find_contours(card_image)

        shapes_radii = np.mean(
            map(lambda contour: map(lambda (t, r): r, ip.contour_xy2polar(contour, n_points=180)), self.contours),
            axis=0)

        # smooth
        kernel_size = 7
        kernel = np.ones(kernel_size) / float(kernel_size)
        shapes_radii = np.hstack((shapes_radii[-kernel_size / 2 + 1:], shapes_radii, shapes_radii[:kernel_size / 2]))
        shapes_radii = np.convolve(shapes_radii, kernel, mode='valid')
        self.data = shapes_radii.tolist()


class Count(AbstractAttribute):

    def __init__(self, card_image=None):
        AbstractAttribute.__init__(self, name='count', default_data=None)
        if card_image is not None:
            self.parse_image(card_image)

    def parse_image(self, card_image):
        self.find_contours(card_image)
        self.data = [len(self.contours), len(self.contours) ** 2]


class Color(AbstractAttribute):

    def __init__(self, card_image=None):
        AbstractAttribute.__init__(self, name='color', default_data={})
        if card_image is not None:
            self.parse_image(card_image)

    def parse_image(self, card_image):
        self.find_contours(card_image)

        shape_mask = np.zeros(card_image.shape)
        ip.cv2.drawContours(shape_mask, self.contours, -1, (1, 1, 1), -1)
        card_image_hsv = ip.cv2.cvtColor(card_image, ip.cv2.COLOR_BGR2HSV)
        shape_hsv = card_image_hsv * shape_mask
        shape_mask = shape_hsv[:, :, 1] > 64
        shape_hue = shape_hsv[shape_mask, 0].astype(np.int32) * 2  # rescale to be [0, 360)
        shape_hue_radians = shape_hue * (np.pi / 180)
        xs = np.cos(shape_hue_radians)
        x = np.mean(xs)
        ys = np.sin(shape_hue_radians)
        y = np.mean(ys)
        self.data = [x, y]
