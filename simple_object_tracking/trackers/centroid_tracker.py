import numpy as np

from simple_object_tracking.tracker import ObjectTracker
from simple_object_tracking.utils import calculate_euclidean_distance
from simple_object_tracking.typing import DistanceToleranceFunction, Width, Height, FPS


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

    def __init__(self,
                 distance_tolerance_f: DistanceToleranceFunction = lambda w, h, f: 0.15 * max(w, h),
                 *args,
                 **kwargs):
        """Crea una instancia de este modelo de seguimiento.

        :param distance_tolerance_f: función que recibe por parámetro el ancho, alto y fps de la
        secuencia y devuelve la distancia máxima a la que puede estar un objeto para ser considerado
        como un emparejamiento posible.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.distance_tolerance_function = distance_tolerance_f
        self.max_distance_allowed = distance_tolerance_f(self.frame_width, self.frame_height,
                                                         self.fps)

    def _algorithm(self):
        # Paso 1. Registrar objetos iniciales.
        for object_detected in self.objects_in_frame(0):
            self.objects.register_object(object_detected, 0)
        # Paso 2. Emparejar, registrar, y desregistrar objetos en el resto de frames.
        for frame_actual in range(1, len(self.sequence)):
            objects_actual = self.objects_in_frame(frame_actual)
            # 1. Emparejar objetos.
            # Objetos registrados hasta el momento.
            objects_registered = self.objects.objects()
            object_registered_uids_matched = list()
            # Objetos detectados en el frame actual, candidatos para emparejar con alguno de los registrados.
            objects_candidates = self.objects_in_frame(frame_actual)
            object_candidates_ids_matched = list()
            # Si no hubo ningún candidato, pasar a la siguiente iteración.
            if len(objects_candidates) == 0:
                continue
            # Para cada uno de los objetos registrados, buscar cuál puede ser su candidato.
            # El mejor candidato vendrá determinado por el que se encuentre a menor distancia.
            for object_registered_id, object_registered in enumerate(objects_registered):
                object_registered_uid, object_registered_last_frame_seen, object_registered_detected = object_registered
                best_actual_candidate_id = None
                best_actual_candidate_distance = None
                distances_to_object_registered = list()
                for candidate_id, candidate_obj in enumerate(objects_candidates):
                    # TODO: Si ya ha hecho matching con el que menos distancia tiene, ir a por el segundo con menos
                    #  distancia.
                    # TODO: Importante que no se emparejen más de un candidato a uno mismo registrado!
                    # Distancias de cada candidato al objeto registrado actual.
                    distance = calculate_euclidean_distance(object_registered_detected.center, candidate_obj.center)
                    distances_to_object_registered.append(distance)
                minimum_distance_candidate = min(distances_to_object_registered)
                minimum_distance_candidate_index = distances_to_object_registered.index(minimum_distance_candidate)
                not_matched = minimum_distance_candidate_index not in object_candidates_ids_matched and (
                    object_registered_uid not in object_registered_uids_matched
                )
                if not_matched and minimum_distance_candidate <= self.max_distance_allowed:
                    # Hacer el emparejamiento.
                    self.objects.update_object(
                        objects_candidates[minimum_distance_candidate_index],
                        object_registered_uid,
                        frame_actual
                    )
                    object_registered_uids_matched.append(object_registered_uid)
                    object_candidates_ids_matched.append(minimum_distance_candidate_index)
                    # Eliminar de la lista de candidatos para registrar luego aquellos que no han sido emparejados.
                    # TODO: habría que marcarlos al acabar la iteración ,pq estamos editando la lista que se está
                    #  ejecutando.
                    # objects_candidates.pop(minimum_distance_candidate_index)
                    # # Si se acabaron los candidatos, pasar al siguiente frame.
                    # if len(objects_candidates) == 0:
                    #     continue



                #     # TODO: Poner todas las distancias y elegir la mejor, del tirón. Sin tanto lío wtf.
                #     distance = calculate_euclidean_distance(object_registered_detected.center, candidate_obj.center)
                #     # Se encuentra a una distancia que podría ser un posible emparejamiento.
                #     if distance <= self.max_distance_allowed:
                #         # Existe un candidato y el actual tiene menor distancia.
                #         p = best_actual_candidate_id is not None and distance < best_actual_candidate_distance
                #         # No existe mejor candidato aún.
                #         q = best_actual_candidate_id is None
                #         if p or q:
                #             best_actual_candidate_id = candidate_id
                #             best_actual_candidate_distance = distance
                # # Si hay candidato disponible, hacer el emparejamiento.
                # if best_actual_candidate_id is not None:
                #     # Hacer el emparejamiento.
                #     self.objects.update_object(
                #         objects_candidates[best_actual_candidate_id],
                #         object_registered_uid,
                #         frame_actual
                #     )
                #     # Eliminar de la lista de candidatos para registrar luego aquellos que no han sido emparejados.
                #     objects_candidates.pop(best_actual_candidate_id)
                #     # Eliminar de la lista de registrados en este frame para evitar que se le empareje más de uno de
                #     # los candidatos.
                #     # TODO: esto debería ir bien... Q_Q
                #     # objects_registered.pop(object_registered_id)
            # 2. Registrar objetos no emparejados.
            for candidate_not_matched in objects_candidates:
                self.objects.register_object(candidate_not_matched, frame_actual)
            # 3. Desregistrar objetos desaparecidos.
            self.objects.unregister_missing_objects(
                frame_actual,
                self.frames_to_unregister_missing_objects
            )
