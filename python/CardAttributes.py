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
        efficiency = []
        complexity = []

        #ip.cv2.imshow("CARD", card_image)
        #ip.cv2.waitKey(0)

        shapes_radii = []

        for contour in self.contours:

            nim = np.zeros(card_image.shape)
            cy = int(nim.shape[0] / 2)
            cx = int(nim.shape[1] / 2)
            contour_polar = ip.contour_xy2polar(contour)

            shapes_radii.append(map(lambda (t, r): r, contour_polar))

            contour_xy = ip.contour_polar2xy(contour_polar, scale=100, center=(cy, cx))
            #ip.cv2.drawContours(nim, [contour_xy], -1, (255, 0, 0), 3)
            #ip.cv2.imshow("TEST", nim)
            #ip.cv2.waitKey(0)

            epsilon = 0.01 * ip.cv2.arcLength(contour, True)
            simple_contour = ip.cv2.approxPolyDP(contour, epsilon, True)
            efficiency.append(float(len(contour)) / float(len(simple_contour)))

            hull_i = ip.cv2.convexHull(contour, returnPoints=False)
            depths = map(lambda (p1, p2, p3, d): d / 256.0, ip.cv2.convexityDefects(contour, hull_i).reshape((-1, 4)))
            defects = filter(lambda d: d > epsilon, depths)
            complexity.append(2 ** len(defects))

        shapes_radii = np.mean(shapes_radii, axis=0)
        # smooth
        kernel_size = 7
        kernel = np.ones(kernel_size) / float(kernel_size)
        shapes_radii = np.hstack((shapes_radii[-kernel_size / 2 + 1:], shapes_radii, shapes_radii[:kernel_size / 2]))
        shapes_radii = np.convolve(shapes_radii, kernel, mode='valid')
        self.data = shapes_radii.tolist()

        #self.data = [np.mean(efficiency), np.min(complexity)]


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
