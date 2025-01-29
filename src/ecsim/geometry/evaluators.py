"""
Classes for evaluating coefficient functions in regions of interest.
"""
from typing import Sequence

import numpy as np
from ngsolve import Mesh, CoefficientFunction


class PointEvaluator:
    """
    Evaluates a coefficient function at a set of given points.
    """

    def __init__(self, mesh: Mesh, points: np.ndarray):
        """
        Initializes the PointEvaluator with the mesh and points to evaluate.
        :param mesh: The NGSolve mesh object.
        :param points: A numpy array of shape (n, 3) representing the n points to evaluate.
        """
        if points.shape[1] != 3:
            raise ValueError("Points must have shape (n, 3).")

        self._raw_points = points
        self._eval_points = [mesh(x, y, z) for x, y, z in points]

    def evaluate(self, coefficient_function: CoefficientFunction):
        """
        Evaluates the given coefficient function at the points defined by the evaluator.
        :param coefficient_function: The NGSolve coefficient function to evaluate.
        :return: A numpy array of evaluated values.
        """
        return np.array([coefficient_function(point) for point in self._eval_points])

    @property
    def raw_points(self):
        """
        Returns the raw points (x, y, z coordinates) used for evaluation.
        :return: A numpy array of shape (n, 3) representing the points to evaluate.
        """
        return self._raw_points


class LineEvaluator(PointEvaluator):
    """
    Evaluates a coefficient function along a straight line segment.
    """

    def __init__(self, mesh: Mesh, start: Sequence, end: Sequence, n: int):
        """
        Initializes the LineEvaluator with the mesh, start, and end points of
        the line, and the number of points to evaluate.
        :param mesh: The NGSolve mesh object.
        :param start: A numpy array of shape (3,) representing the starting point of the line.
        :param end: A numpy array of shape (3,) representing the ending point of the line.
        :param n: The number of points to evaluate along the line segment.
        """
        # Generate the coordinates for the line segment
        x_coords = np.linspace(start[0], end[0], n)
        y_coords = np.linspace(start[1], end[1], n)
        z_coords = np.linspace(start[2], end[2], n)

        # Use the parent class to initialize the evaluator
        raw_points = np.column_stack((x_coords, y_coords, z_coords))
        super().__init__(mesh, raw_points)
