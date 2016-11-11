import numpy as np
from scipy.spatial import Delaunay


class Transformer:

    # create either a source or a target triangulation
    # this object can be re-used for all processes with a common source or target shape
    def __init__(self, is_target, image_dimensions, vertices=None):

        self.image_dimensions = image_dimensions
        # if vertices are not set, assume we want to use the entire image
        if vertices is None:
            rows, columns = image_dimensions[:2]
            self.vertices = np.array([[0, 0],
                                      [0, columns - 1],
                                      [rows - 1, 0],
                                      [rows - 1, columns - 1]])
        else:
            self.vertices = vertices.reshape((-1, 2))

        r0, c0 = np.round(np.mean(self.vertices, axis=0)).astype(np.int32)
        self.vertices = np.array(sorted(self.vertices, key=lambda (r, c): np.arctan2(r0 - r, c0 - c)))
        # get barycentric coordinates and corresponding points in target (new) image
        nys = np.arange(self.image_dimensions[0])
        nxs = np.arange(self.image_dimensions[1])
        image_pixels = np.transpose([np.repeat(nys, len(nxs)), np.tile(nxs, len(nys))])

        image_pixels = np.array(image_pixels)

        triangles = Delaunay(self.vertices)

        # code below is abbout 0.01 s
        memberships = triangles.find_simplex(image_pixels)  # returns the triangle that each pixel is a member of
        Ts = triangles.transform[memberships, :2]  # transformation matrices
        prs = image_pixels - triangles.transform[memberships, 2]  # intermediate transformation

        # code below is almost 1 s
        bl = Ts[:, 0, 0] * prs[:, 0] + Ts[:, 0, 1] * prs[:, 1]
        br = Ts[:, 1, 0] * prs[:, 0] + Ts[:, 1, 1] * prs[:, 1]
        barycentric_coordinates = np.hstack((bl.reshape((-1, 1)), br.reshape((-1, 1))))
        barycentric_coordinates = np.hstack((barycentric_coordinates,
                                            1 - np.sum(barycentric_coordinates, axis=1, keepdims=True)))

        target_vertices_indices = triangles.simplices[memberships]

        self.is_target = is_target
        self.rows = image_pixels[:, 0]
        self.cols = image_pixels[:, 1]
        self.barycentric = barycentric_coordinates
        self.indices = target_vertices_indices

    def transform(self, source_image, vertices, target_image=None):
        if target_image is None:
            target_image = np.zeros(self.image_dimensions, np.uint8)

        vertices = vertices.reshape((-1, 2))
        vertices = np.fliplr(vertices)
        r0, c0 = np.round(np.mean(vertices, axis=0)).astype(np.int32)
        vertices = np.array(sorted(vertices, key=lambda (r, c): np.arctan2(r0 - r, c0 - c)))

        rpts = vertices[self.indices, :]
        # for when object was built with vertices as a common target
        if self.is_target:
            source_rows = np.multiply(rpts[:, :, 0], self.barycentric).sum(axis=1, keepdims=False).astype("int32")
            source_cols = np.multiply(rpts[:, :, 1], self.barycentric).sum(axis=1, keepdims=False).astype("int32")
            target_rows = self.rows
            target_cols = self.cols
        # for when object was built with vertices as a common source
        else:
            source_rows = self.rows
            source_cols = self.cols
            target_rows = np.multiply(rpts[:, :, 0], self.barycentric).sum(axis=1, keepdims=False).astype("int32")
            target_cols = np.multiply(rpts[:, :, 1], self.barycentric).sum(axis=1, keepdims=False).astype("int32")

        target_image[target_rows, target_cols, :] = source_image[source_rows, source_cols, :]

        return target_image
