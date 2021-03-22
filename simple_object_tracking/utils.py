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


def positions_in_object_tracking(objects_history, sequence):
    """Devuelve la lista de posiciones para cada frame en la que se vio el objeto.

    Si el objeto no fue visto en una posición, para ese índice (frame) el valor es None.
    """
    positions = list()
    frame_seen, obj = zip(*objects_history)
    for frame_id in range(len(sequence)):
        if frame_id in frame_seen:
            index = frame_seen.index(frame_id)
            positions.append(obj[index].get_centroid())
        else:
            positions.append(None)
    return positions


def sequence_with_objects_trace(sequence, tracker):
    """Dada una secuencia, dibuja todas las trazas de todos los objetos que se han podido seguir.
    """
    class Track:
        """Mantiene la información del seguimiento de un objeto."""
        def __init__(self, positions, last_frame_seen, color_trace=(255, 0, 0)):
            # Posiciones del objeto indexadas por el nº del frame.
            self.positions = positions
            # Último frame en el que fue visto el objeto.
            self.last_frame_seen = last_frame_seen
            # Color con el que se pintará este objeto.
            self.color_trace = color_trace
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
        track = Track(positions, last_frame_seen, color_trace=colors[obj_uid])
        objects_track.append(track)
    # Pintar en cada frame el seguimiento hasta ese frame.
    for frame_id, frame in enumerate(sequence):
        # Pintar el seguimiento para cada objeto hasta el frame_id.
        for obj_uid in objects_uid:
            obj_track = objects_track[obj_uid]
            frames_elapsed = frame_id - obj_track.last_frame_seen
            # Si han pasado más de N frames desde que se siguió ese objeto, pasar al siguiente.
            if frames_elapsed > 40:
                continue
            obj_history = tracker.registered_objects.history[obj_uid]
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
            for pair_point in pair_points:
                p1, p2 = pair_point
                seq_with_traces[frame_id] = cv2.line(frame, p1, p2, obj_track.color_trace, 5)
                # seq_with_traces[frame_id] = cv2.circle(frame, point_1, 0, (0, 255, 0), 3)
    return seq_with_traces

