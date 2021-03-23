import cv2
import numpy as np


def calculate_euclidean_distance(point_1, point_2):
    """Calcula la distancia euclídea entre 2 puntos.

    :param point_1: 
    :param point_2: 
    :return: distancia euclídea.
    """
    p = np.array(point_1)
    q = np.array(point_2)
    return np.linalg.norm(p - q)


def positions_in_object_tracking(object_history, sequence):
    """Cálculo de la lista de posiciones para cada frame en la que un objeto fue visto.

    :param object_history: lista de tuplas del número del frame en que visto y el objeto.
    :param sequence: secuencia de video.
    :return: lista indexada por los frames del video. En cada posición se encuentra la posición
    (x,y) donde se encontraba el objeto en ese instante. Si el objeto no fue visto en ese frame,
    el valor en esa posición es None.
    """
    positions = list()
    frame_seen, obj = zip(*object_history)
    for frame_id in range(len(sequence)):
        if frame_id in frame_seen:
            index = frame_seen.index(frame_id)
            positions.append(obj[index].get_centroid())
        else:
            positions.append(None)
    return positions


def sequence_with_objects_trace(sequence, tracker, max_frames_elapsed=40):
    """Dada una secuencia, dibuja todas las trazas de todos los objetos que se han podido seguir.

    :param sequence: secuencia de vídeo.
    :param tracker: objeto del modelo de seguimiento ya ejecutado.
    :param max_frames_elapsed: número máximo de frames que deben pasar para eliminar el trazado
    que ha realizado un objeto.
    :return: secuencia de vídeo con las trazas de los objetos pintadas.
    """
    class Track:
        """Mantiene la información del seguimiento de un objeto."""
        def __init__(self,
                     positions,
                     last_frame_seen,
                     color_trace=(255, 0, 0),
                     color_box=(255, 0, 0)):
            # Posiciones del objeto indexadas por el nº del frame.
            self.positions = positions
            # Último frame en el que fue visto el objeto.
            self.last_frame_seen = last_frame_seen
            # Color con el que se pintará la info. del objeto.
            self.color_trace = color_trace
            self.color_box = color_box
    # Copia del video para pintarlo.
    seq_with_traces = sequence.copy()
    # Extraer todos los objetos (incluso los eliminados en algún momento).
    objects_uid = tracker.registered_objects.objects_uid
    # Crear lista con los tracks de cada objeto.
    objects_track = list()
    for obj_uid in objects_uid:
        obj_history = tracker.registered_objects.history[obj_uid]
        positions = positions_in_object_tracking(obj_history, sequence)
        last_frame_seen = [index for index, pos in enumerate(positions) if pos is not None][-1]
        colors = np.random.uniform(0, 255, size=(len(objects_uid), 3))
        track = Track(positions,
                      last_frame_seen,
                      color_trace=colors[obj_uid],
                      color_box=colors[obj_uid])
        objects_track.append(track)
    # Pintar en cada frame el seguimiento hasta ese frame.
    for frame_id, frame in enumerate(sequence):
        # Pintar el seguimiento para cada objeto hasta el frame_id.
        for obj_uid in objects_uid:
            # Comprobar si el objeto se detectó en este frame, si no se detectó, pasar, ya que
            # no se hará nada con él.
            obj_history = tracker.registered_objects.history[obj_uid]
            # Obtener el objeto en el frame actual (si es posible, ya que no siempre se detecta).
            objs_actual_frame = [obj for fr_id, obj in obj_history if frame_id == fr_id]
            obj_actual_frame = None
            if len(objs_actual_frame) == 1:
                obj_actual_frame = objs_actual_frame[0]
            elif len(objs_actual_frame) > 1:
                raise Exception('This shouldnt happen.')
            obj_track = objects_track[obj_uid]
            frames_elapsed = frame_id - obj_track.last_frame_seen
            # Si han pasado más de N frames desde que se siguió ese objeto, pasar al siguiente.
            if frames_elapsed > max_frames_elapsed:
                continue
            # Posiciones de ese objeto.
            positions = obj_track.positions
            # Posiciones hasta el frame actual.
            positions_until_frame = positions[:frame_id+1]
            # Eliminar los frames en los que no se obtuvo su posición
            # TODO. SOT-12. Interpolar posiciones.
            positions_only_detected = list(filter(None, positions_until_frame))
            # Obtener todos los pares de puntos para dibujar.
            pair_points = list()
            for i in range(1, len(positions_only_detected)):
                pair = (positions_only_detected[i-1], positions_only_detected[i])
                pair_points.append(pair)
            # Lets draw!
            # Pintar bounding box.
            if obj_actual_frame:
                (left, right, top, bottom) = obj_actual_frame.bounding_box
                seq_with_traces[frame_id] = cv2.rectangle(frame, (left, top), (right, bottom),
                                                          obj_track.color_box, 2)
            for pair_point in pair_points:
                p1, p2 = pair_point
                # Pintar línea de seguimiento.
                seq_with_traces[frame_id] = cv2.line(frame, p1, p2, obj_track.color_trace, 2)
                # seq_with_traces[frame_id] = cv2.circle(frame, point_1, 0, (0, 255, 0), 3)
    return seq_with_traces

