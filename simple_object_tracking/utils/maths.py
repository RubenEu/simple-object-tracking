import numpy as np

from simple_object_detection.typing import Point2D


def euclidean_norm(p: Point2D, q: Point2D) -> float:
    """Calcula la distancia euclídea entre 2 puntos.

    :param p: punto 1.
    :param q: punto 2.
    :return: distancia euclídea.
    """
    return np.linalg.norm(np.array(p) - np.array(q))
