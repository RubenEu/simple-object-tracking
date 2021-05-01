from typing import List, Tuple

from simple_object_detection.object import Object

from simple_object_tracking.tracker import ObjectTracker
from simple_object_tracking.utils import euclidean_norm

# TODO: Hay que cambiar muchas cosas debido a los cambios en datastructures.py


class CentroidTracker(ObjectTracker):
    """Modelo de seguimiento de objetos basados en su centro.

    Algoritmo de seguimiento:
    1. Registra todos los objetos detectados en el primer frame.
    2. Para cada frame en la secuencia:
        2.0. Obtener los objetos del frame actual.
        2.1. Buscar los emparejamientos posibles, resolver los conflictos y emparejar, actualizando
             la lista que mantiene el seguimiento de los objetos.
        2.2. Registrar aquellos objetos detectados en el frame actual que no han sido emparejados.
        2.3. Desregistrar los objetos que llevan sin detectarse un número de frames, determinado por
        el parámetro frames_to_unregister_object.

    Principales problemas:
    - Si se detecta un objeto de manera múltiple, se genera el trazado bien, pero supone que so
    distintos objetos los que está viendo, no uno mismo.
    """

    def __init__(self, max_distance_allowed: int = 120, *args, **kwargs):
        """

        :param max_distance_allowed: máxima distancia para considerar que dos objetos en dos frames
        distintos pueden considerarse el mismo.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.max_distance_allowed = max_distance_allowed

    def _matching_step(self,
                       objs_registered: List[Tuple[int, int, Object]],
                       objs_registered_uids_matched: List[int],
                       objs_actual: List[Object],
                       objs_actual_ids_matched: List[int],
                       frame_actual: int) -> None:
        # Si no hay ningún objeto registrado, saltar fase de emparejeamiento para el frame actual.
        if len(objs_registered) == 0:
            return
        # Si no se ha detectado ningún objeto en el frame actual, tampoco se puede realizar
        # emparejamiento.
        if len(objs_actual) == 0:
            return
        # Para cada uno de los objetos actuales, buscar con cuál de los objetos registrados
        # emparejar.
        for obj_actual_id, obj_actual_detection in enumerate(objs_actual):
            distances_to_objs_actual = list()
            # Calcular las distancias de los objetos registrados al objeto actual detectado.
            for obj_registered_id, obj_registered in enumerate(objs_registered):
                (obj_registered_uid, obj_registered_last_frame_seen,
                 obj_registered_detected) = obj_registered
                distance = euclidean_norm(obj_registered_detected.center,
                                          obj_actual_detection.center)
                distances_to_objs_actual.append(distance)
            # Busca la distancia mínima.
            min_distance = min(distances_to_objs_actual)
            min_distance_index = distances_to_objs_actual.index(min_distance)
            # Comprueba que no se ha realizado el emparejamiento del objeto registrado con alguno de
            # los actuales.
            not_matched = min_distance_index not in objs_registered_uids_matched
            # TODO: Si no se puede realizar el emparejamiento con el primero que está a menos
            #  distancia, se podría probar con los siguientes...
            if not_matched and min_distance <= self.max_distance_allowed:
                # Hacer el emparejamiento.
                self.objects.update_object(
                    objs_actual[obj_actual_id],
                    objs_registered[min_distance_index][0],
                    frame_actual
                )
                objs_registered_uids_matched.append(min_distance_index)
                objs_actual_ids_matched.append(obj_actual_id)

    def _algorithm(self):
        # Paso 1. Registrar objetos iniciales.
        for object_detected in self.objects_in_frame(0):
            self.objects.register_object(object_detected, 0)
        # Paso 2. Emparejar, registrar, y desregistrar objetos en el resto de frames.
        for frame_actual in range(1, len(self.sequence)):
            # 1. Emparejar objetos.
            # Objetos registrados hasta el momento.
            objs_registered = self.objects.objects()
            objs_registered_uids_matched = list()
            # Objetos detectados en el frame actual, candidatos para emparejar con alguno de los
            # registrados.
            objs_actual = self.objects_in_frame(frame_actual)
            objs_actual_ids_matched = list()
            self._matching_step(objs_registered, objs_registered_uids_matched,
                                objs_actual, objs_actual_ids_matched,
                                frame_actual)
            # 2. Registrar objetos no emparejados.
            for obj_actual_id, object_candidate_detection in enumerate(objs_actual):
                if obj_actual_id not in objs_actual_ids_matched:
                    self.objects.register_object(object_candidate_detection, frame_actual)
            # 3. Desregistrar objetos desaparecidos.
            self.objects.unregister_missing_objects(
                frame_actual,
                self.frames_to_unregister_missing_objects
            )
