import numpy as np
import cv2
from typing import List

from simple_object_detection.typing import Point2D
from simple_object_detection.object import Object

from simple_object_tracking.typing import Sequence, Timestamps
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


def sequence_with_traces(sequence: Sequence, timestamps: Timestamps,
                         objects_stored: SequenceObjects):
    """Genera una secuencia de vídeo con los trazados del seguimiento de los objetos.

    Además, se mantendrá una caja delimitadora con cada uno de los objetos detectados en ese frame,
    y con el texto de qué clase de objeto es y su puntuación.

    Abajo a la derecha se podrá observar información del vídeo: frame actual, milisegundo, cantidad
    de objetos en la escena, cantidad de objetos desregistrados.

    :param sequence: secuencia de video
    :param timestamps: lista de marcas de tiempo indexada por frame.
    :param objects_stored: almacenamiento e información de los objetos de la secuencia.
    :return: secuencia de vídeo con la información plasmada en él.
    """
    # Copiar la secuencia para no editar la misma que se pasa por parámetro.
    sequence = sequence.copy()
    # Generar colores aleatorios.
    colors = np.random.uniform(0, 255, size=(len(objects_stored), 3))
    # Iterar sobre los frames de la secuencia.
    for frame_id, frame in enumerate(sequence):
        # Pintar la información de la trayectoria de cada objeto hasta el frame actual.
        for object_uid in range(len(objects_stored)):
            object_history = objects_stored.object_uid(object_uid)
            # Iterar sobre las detecciones del objeto hasta el frame actual.
            object_history_index, object_frame = 1, 0
            while object_history_index < len(object_history) and object_frame <= frame_id:
                object_frame_prev, object_detection_prev = object_history[object_history_index-1]
                object_frame, object_detection = object_history[object_history_index]
                # Solo dibujar trazado si es anterior al frame actual.
                if object_frame <= frame_id:
                    cv2.line(frame, object_detection_prev.center, object_detection.center,
                             colors[object_uid], 2)
                # Solo dibujar la bounding box si es el frame actual.
                if object_frame == frame_id:
                    top_left_corner = object_detection.bounding_box[0]
                    bottom_right_corner = object_detection.bounding_box[2]
                    cv2.rectangle(frame, top_left_corner, bottom_right_corner, colors[object_uid],
                                  3)
                object_history_index += 1
    return sequence





