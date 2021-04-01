import numpy as np
import cv2

from simple_object_detection.typing import Point2D

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


def sequence_with_traces(sequence: Sequence,
                         timestamps: Timestamps,
                         objects_stored: SequenceObjects,
                         frames_missing_to_remove_trace=30):
    """Genera una secuencia de vídeo con los trazados del seguimiento de los objetos.

    Además, se mantendrá una caja delimitadora con cada uno de los objetos detectados en ese frame,
    y con el texto de qué clase de objeto es y su puntuación.

    Abajo a la derecha se podrá observar información del vídeo: frame actual, milisegundo, cantidad
    de objetos en la escena, cantidad de objetos desregistrados.

    :param sequence: secuencia de video
    :param timestamps: lista de marcas de tiempo indexada por frame.
    :param objects_stored: almacenamiento e información de los objetos de la secuencia.
    :param frames_missing_to_remove_trace: Cantidad de frames que tienen pasar para eliminar el trazado del objeto.
    :return: secuencia de vídeo con la información plasmada en él.
    """
    # Copiar la secuencia para no editar la misma que se pasa por parámetro.
    sequence = [frame.copy() for frame in sequence]
    # Generar colores aleatorios.
    colors = np.random.uniform(0, 255, size=(len(objects_stored), 3))
    # Iterar sobre los frames de la secuencia.
    for frame_id, frame in enumerate(sequence):
        # 1. Caja de información
        width, height = sequence[0].shape[1], sequence[0].shape[0]
        box_width, box_height = min(width, 500), min(height, 130)
        # Pintar la línea superior.
        p1, p2 = (width - box_width, height - box_height), (width, height - box_height)
        cv2.line(frame, p1, p2, (0, 255, 255), 3)
        # Pintar la linea izquierda.
        p3 = (width - box_width, height)
        cv2.line(frame, p1, p3, (0, 255, 255), 3)
        # Añadir texto.
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Output information console!', (p1[0] + 5, p1[1] + 23), font, 0.7, (255, 255, 255),
                    2, cv2.LINE_AA)
        cv2.putText(frame, f'Frame: {frame_id}.', (p1[0] + 5, p1[1] + 52), font, 0.65, (255, 255, 255),
                    1, cv2.LINE_AA)
        text = f'Timestamp: {timestamps[frame_id]/1000}s ({timestamps[frame_id]}ms)'
        cv2.putText(frame, text, (p1[0] + 5, p1[1] + 80), font, 0.65, (255, 255, 255),
                    1, cv2.LINE_AA)
        # 2. Trazado.
        for object_uid in range(len(objects_stored)):
            object_history = objects_stored.object_uid(object_uid)
            # No dibujar trazado si el objeto lleva desaparecido más de 'frames_missing_to_remove_trace' frames.
            frames_elapsed = frame_id - object_history[-1][0]
            if frames_elapsed > frames_missing_to_remove_trace:
                continue
            # Iterar sobre las detecciones del objeto hasta el frame actual.
            object_history_index, object_frame = 1, 0
            # Obtener la detección previa y siguiente para realizar el trazado.
            while object_history_index < len(object_history) and object_frame <= frame_id:
                object_frame_prev, object_detection_prev = object_history[object_history_index-1]
                object_frame, object_detection = object_history[object_history_index]
                # Solo dibujar trazado si es anterior al frame actual.
                if object_frame <= frame_id:
                    cv2.line(frame, object_detection_prev.center, object_detection.center,
                             colors[object_uid], 2, cv2.LINE_AA)
                object_history_index += 1
        # 3. Bounding box objectos
        font = cv2.FONT_HERSHEY_SIMPLEX
        for object_uid, object_detection in objects_stored.objects_frame(frame_id):
            p1 = top_left_corner = object_detection.bounding_box[0]
            p2 = top_right_corner = object_detection.bounding_box[1]
            p3 = bottom_right_corner = object_detection.bounding_box[2]
            p4 = bottom_left_corner = object_detection.bounding_box[3]
            # cv2.rectangle(frame, top_left_corner, bottom_right_corner, colors[object_uid], 2)
            cv2.line(frame, p1, p2, colors[object_uid], 2, cv2.LINE_AA)
            cv2.line(frame, p2, p3, colors[object_uid], 2, cv2.LINE_AA)
            cv2.line(frame, p3, p4, colors[object_uid], 2, cv2.LINE_AA)
            cv2.line(frame, p4, p1, colors[object_uid], 2, cv2.LINE_AA)
            # Object UID text.
            text = f'UID: {object_uid}'
            top_left_corner_x, top_left_corner_y = top_left_corner
            position = (top_left_corner_x, top_left_corner_y - 7)
            cv2.putText(frame, text, position, font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            # Object label text.
            score = int(object_detection.score * 100)
            text = f'{object_detection.label} {score}%'
            position = (top_left_corner_x, top_left_corner_y - 20)
            cv2.putText(frame, text, position, font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            # Object position text.
            text = f'{object_detection.center}'
            position = (top_left_corner_x, top_left_corner_y - 33)
            cv2.putText(frame, text, position, font, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return sequence





