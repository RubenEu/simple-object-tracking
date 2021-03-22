import cv2
import numpy as np


def show_object_in_sequence():
    # TODO: Dado objeto, seguirlo durante una secuencia (?
    pass


def calculate_euclidean_distance(point_1, point_2):
    """Calcula la distancia euclídea entre 2 puntos.
    :param point_1: 
    :param point_2: 
    :return: distancia euclídea.
    """
    p = np.array(point_1)
    q = np.array(point_2)
    return np.linalg.norm(p - q)


