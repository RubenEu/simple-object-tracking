from typing import List, Tuple
import numpy as np

from simple_object_detection.typing import Point2D


def point_distance_to_line(point: Point2D, line: Tuple[Point2D, Point2D]) -> float:
    """Calcula la distancia entre un punto y una recta definida por sus extremos.

    Extraído de:
    https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points

    :param point: punto 2D (x, y).
    :param line: recta definida por dos puntos [(x0, y0), (x1, y1)].
    :return: mínima distancia entre el punto y la recta.
    """
    p1, p2 = np.array(line[0]), np.array(line[1])
    p3 = np.array(point)
    return np.abs(np.cross(p2-p1, p1-p3)) / np.linalg.norm(p2-p1)


def find_closest_position_to_line(positions: List[Point2D], line: Tuple[Point2D, Point2D]) -> int:
    """Busca el índice de las posiciones en la que la posición 2D es la más cercana a la recta
    definida por dos puntos.

    :param positions: lista de posiciones.
    :param line: recta definida por dos puntos.
    :return: índice de la posición más cercana a la recta.
    """
    distances = [point_distance_to_line(p, line) for p in positions]
    return distances.index(min(distances))
