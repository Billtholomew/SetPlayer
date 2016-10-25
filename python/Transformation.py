import numpy as np
from scipy.spatial import Delaunay


class Transformer:

    def __init__(self, targetImageDimensions, targetVertices=None):
        self.targetImageDimensions = targetImageDimensions
        # if target vertices are not set, assume we want to fill the entire targetImage
        if not targetVertices:
            self.targetVertices = np.array([[0, 0],
                                            [0, targetImageDimensions[1]],
                                            [targetImageDimensions[0], targetImageDimensions[1]],
                                            [targetImageDimensions[0], 0]])
        else:
            self.targetVertices = targetVertices

        # get barycentric coordinates and corresponding points in target (new) image
        nys = np.arange(self.targetImageDimensions[0])
        nxs = np.arange(self.targetImageDimensions[1])
        imagePixels = np.transpose([np.repeat(nys, len(nxs)), np.tile(nxs, len(nys))])

        triangles = Delaunay(self.targetVertices)
        memberships = triangles.find_simplex(imagePixels)  # returns the triangle that each pixel is a member of
        Ts = triangles.transform[memberships, :2]  # transformation matrices
        prs = imagePixels - triangles.transform[memberships, 2]  # intermediate transformation

        barycentricCoordinates = np.array([Ts[i].dot(pr) for i, pr in enumerate(prs)])
        barycentricCoordinates = np.hstack((barycentricCoordinates,
                                            1 - np.sum(barycentricCoordinates, axis=1, keepdims=True)))

        targetVerticesIndices = triangles.simplices[memberships]

        self.targetRow = imagePixels[:, 0]
        self.targetCol = imagePixels[:, 1]
        self.barys = barycentricCoordinates
        self.idxs = targetVerticesIndices

    def transform(self, oim, sourceVertices):
        nim = np.zeros(self.targetImageDimensions, np.uint8)
        rpts = sourceVertices[self.idxs, :]
        originalRow = np.multiply(rpts[:, :, 0], self.barys).sum(axis=1, keepdims=True).astype("int32")
        originalCol = np.multiply(rpts[:, :, 1], self.barys).sum(axis=1, keepdims=True).astype("int32")
        nim[self.targetRow, self.targetCol, :] = oim[originalRow, originalCol, :].reshape((-1, 3))

        return nim

    # given a roughly trapezoidal contour, turn it into a simple trapezoid with only 4 vertices
    def simple_trapezoid(self,contour):
        contour = np.reshape(contour, (len(contour), 2))
        rectangle = np.zeros((4, 2), dtype="int32")
        s = contour.sum(axis=1)
        rectangle[0] = contour[np.argmin(s)]
        rectangle[2] = contour[np.argmax(s)]
        diff = np.diff(contour, axis=1)
        rectangle[1] = contour[np.argmin(diff)]
        rectangle[3] = contour[np.argmax(diff)]
        return np.fliplr(rectangle)
