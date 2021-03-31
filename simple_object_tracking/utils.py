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
    sequence = [frame.copy() for frame in sequence]
    # Generar colores aleatorios.
    colors = np.random.uniform(0, 255, size=(len(objects_stored), 3))
    # Iterar sobre los frames de la secuencia.
    for frame_id, frame in enumerate(sequence):
        # 1. Trazado.
        for object_uid in range(len(objects_stored)):
            object_history = objects_stored.object_uid(object_uid)
            # Iterar sobre las detecciones del objeto hasta el frame actual.
            object_history_index, object_frame = 1, 0
            # Obtener la detección previa y siguiente para realizar el trazado.
            while object_history_index < len(object_history) and object_frame <= frame_id:
                object_frame_prev, object_detection_prev = object_history[object_history_index-1]
                object_frame, object_detection = object_history[object_history_index]
                # Solo dibujar trazado si es anterior al frame actual.
                if object_frame <= frame_id:
                    cv2.line(frame, object_detection_prev.center, object_detection.center,
                             colors[object_uid], 2)
                object_history_index += 1
        # 2. Bounding box
        font = cv2.FONT_HERSHEY_SIMPLEX
        for object_uid, object_detection in objects_stored.objects_frame(frame_id):
            top_left_corner = object_detection.bounding_box[0]
            bottom_right_corner = object_detection.bounding_box[2]
            cv2.rectangle(frame, top_left_corner, bottom_right_corner, colors[object_uid],
                          2)
            # Object UID text.
            text = f'UID: {object_uid}'
            top_left_corner_x, top_left_corner_y = top_left_corner
            position = (top_left_corner_x, top_left_corner_y - 7)
            cv2.putText(frame, text, position, font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            # Object label text.
            score = int(object_detection.score * 100)
            text = f'{object_detection.label} {score}'
            position = (top_left_corner_x, top_left_corner_y - 20)
            cv2.putText(frame, text, position, font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
            # Object position text.
            text = f'{object_detection.center}'
            position = (top_left_corner_x, top_left_corner_y - 33)
            cv2.putText(frame, text, position, font, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
        # 3. Caja de información
        width, height = sequence[0].shape[1], sequence[0].shape[0]
        box_width, box_height = int(0.75 * width), int(0.19 * height)
        # Pintar la línea superior.
        p1, p2 = (width - box_width, height - box_height), (width, height - box_height)
        cv2.line(frame, p1, p2, (0, 255, 255), 3)
        # Pintar la linea izquierda.
        p3 = (width - box_width, height)
        cv2.line(frame, p1, p3, (0, 255, 255), 3)
        # Añadir texto.
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Consola de informacion de salida!', (p1[0] + 5, p1[1] + 23), font, 0.7, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame, f'Frame: {frame_id}', (p1[0] + 5, p1[1] + 50), font, 0.55, (255, 255, 255),
                    1, cv2.LINE_AA)


    return sequence





