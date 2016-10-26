import numpy as np
from scipy.spatial import Delaunay


class Transformer:

    def __init__(self, target_image_dimensions, target_vertices=None):
        self.target_image_dimensions = target_image_dimensions
        # if target vertices are not set, assume we want to fill the entire targetImage
        if not target_vertices:
            self.target_vertices = np.array([[0, 0],
                                             [0, target_image_dimensions[1]],
                                             [target_image_dimensions[0], target_image_dimensions[1]],
                                             [target_image_dimensions[0], 0]])
        else:
            self.target_vertices = target_vertices

        # get barycentric coordinates and corresponding points in target (new) image
        nys = np.arange(self.target_image_dimensions[0])
        nxs = np.arange(self.target_image_dimensions[1])
        image_pixels = np.transpose([np.repeat(nys, len(nxs)), np.tile(nxs, len(nys))])

        triangles = Delaunay(self.target_vertices)
        memberships = triangles.find_simplex(image_pixels)  # returns the triangle that each pixel is a member of
        Ts = triangles.transform[memberships, :2]  # transformation matrices
        prs = image_pixels - triangles.transform[memberships, 2]  # intermediate transformation

        barycentric_coordinates = np.array([Ts[i].dot(pr) for i, pr in enumerate(prs)])
        barycentric_coordinates = np.hstack((barycentric_coordinates,
                                            1 - np.sum(barycentric_coordinates, axis=1, keepdims=True)))

        target_vertices_indices = triangles.simplices[memberships]

        self.targetRow = image_pixels[:, 0]
        self.targetCol = image_pixels[:, 1]
        self.barycentric = barycentric_coordinates
        self.indices = target_vertices_indices

    def transform(self, oim, source_vertices):
        nim = np.zeros(self.target_image_dimensions, np.uint8)
        rpts = source_vertices[self.indices, :]
        original_row = np.multiply(rpts[:, :, 0], self.barycentric).sum(axis=1, keepdims=True).astype("int32")
        original_col = np.multiply(rpts[:, :, 1], self.barycentric).sum(axis=1, keepdims=True).astype("int32")
        nim[self.targetRow, self.targetCol, :] = oim[original_row, original_col, :].reshape((-1, 3))
        return nim


# given a roughly trapezoidal contour, turn it into a simple trapezoid with only 4 vertices
def simple_trapezoid(contour):
    contour = np.reshape(contour, (len(contour), 2))
    rectangle = np.zeros((4, 2), dtype="int32")
    s = contour.sum(axis=1)
    rectangle[0] = contour[np.argmin(s)]
    rectangle[2] = contour[np.argmax(s)]
    diff = np.diff(contour, axis=1)
    rectangle[1] = contour[np.argmin(diff)]
    rectangle[3] = contour[np.argmax(diff)]
    return np.fliplr(rectangle)
