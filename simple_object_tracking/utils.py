from typing import Any

import numpy as np
import cv2

from simple_object_detection.typing import Point2D
from simple_object_detection.utils.video import StreamSequence

from simple_object_tracking.datastructures import SequenceObjects


def calculate_euclidean_distance(point_1: Point2D, point_2: Point2D) -> float:
    """Calcula la distancia euclídea entre 2 puntos.

    :param point_1: punto p.
    :param point_2: punto q.
    :return: distancia euclídea.
    """
    p = np.array(point_1)
    q = np.array(point_2)
    return np.linalg.norm(p - q)


def sequence_with_traces():
    """TODO: Usar una clase de VideoWriter para ir escribiendo frame a frame, en lugar de cargarlo
     todo en memoria.
     """
    raise DeprecationWarning()
