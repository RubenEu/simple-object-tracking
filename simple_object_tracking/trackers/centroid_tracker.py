from typing import List, Tuple, NamedTuple, Optional

from tqdm import tqdm

from simple_object_detection.object import Object
from simple_object_detection.typing import Image
from simple_object_tracking.datastructures import TrackedObjectDetection

from simple_object_tracking.tracker import ObjectTracker
from simple_object_tracking.utils import euclidean_norm


class PointTracker(ObjectTracker):
    """Modelo de seguimiento de objetos representando el objeto por un punto (su centro):
    """

    class MatchedObject(NamedTuple):
        """Representación del emparejamiento de un objeto, su detección previa y la actual."""
        previous: Object
        actual: Object

    def __init__(self,
                 max_distance_allowed: int = 120,
                 register_mask_region: Image = None,
                 *args,
                 **kwargs):
        """

        :param max_distance_allowed: máxima distancia para considerar que dos objetos en dos frames
        distintos pueden considerarse el mismo.
        :param register_mask_region: máscara que indica la región donde únicamente se podrán
        registrar objetos nuevos, es decir, si aparece un objeto en otra zona, no podrá ser
        registrado, solo propuesto para emparejamiento.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.max_distance_allowed = max_distance_allowed
        self.register_mask_region = register_mask_region

    def _matching_step(self,
                       frame_actual: int,
                       objects_actual: List[Object]) -> Optional[List[MatchedObject]]:
        """Paso que realiza el emparejamiento de los objetos.

        Proceso:
            #. Se buscan todos los objetos que constan como registrados (es decir, se han seguido
               hasta hace poco).
            #. Comprobar que hay algún objeto registrado o se ha detectado algún objeto en el frame
               actual, si no, no hay emparejamiento que realizar.
            #. Ordenar los objetos actuales y registrados por puntuación, así se harán el
               emparejamiento con ellos primeramente.
            #. Para cada objeto del frame actual: calcular su distancia y emparejarlo con el que
               más cerca esté de los registrados, si la distancia es menor que un factor dado.
            #. Finalmente se devuelven todos los emparejamientos realizados.

        :param frame_actual: identificador del frame actual.
        :param objects_actual: lista de objetos del frame actual.
        :return: lista de emparejamientos si es que se ha realizado alguno, si no, None.
        """
        matched_objects = []
        registered_tracked_objects = self.objects.registered_tracked_objects()
        # Comprobar si hay algún objeto con el que poder emparejar.
        if len(registered_tracked_objects) == 0:
            return None
        # Comprobar si hay algún objeto en el frame actual con el que poder emparejar.
        if len(objects_actual) == 0:
            return None
        # Ordenar los objetos del frame actual por puntuación, así se hará el emparejamiento de
        # ellos primero.
        objects_actual.sort(reverse=True, key=lambda obj: obj.score)
        # Ordenar los objetos registrados por puntuación también.
        registered_tracked_objects.sort(reverse=True, key=lambda tobj: tobj[-1].object.score)
        # Emparejar cada uno de los objetos del frame actual con los registrados.
        for object_actual in objects_actual:
            # Calcular las distancias del objeto actual a los registrados.
            distances = [euclidean_norm(object_actual.center, tracked_object[-1].object.center)
                         for tracked_object in registered_tracked_objects]
            # Obtener el objeto registro al que menor distancia existe.
            index = distances.index(min(distances))
            match = self.MatchedObject(previous=registered_tracked_objects[index],
                                       actual=object_actual)
            # Lista de objetos previous ya emparejados.
            matched_registered_objects = [match.previous for match in matched_objects]
            # Comprobación si el objeto registrado con el que se va a emparejar ya ha sido
            # emparejado.
            object_previous_matched = match.previous in matched_registered_objects
            if not object_previous_matched and distances[index] <= self.max_distance_allowed:
                previous_object_id = registered_tracked_objects[index].id
                # Actualizar el objeto seguido con el nuevo emparejamiento.
                self.objects.update_object(object_actual, previous_object_id, frame_actual)
                # Guardar la información de los emparejamientos realizados.
                matched_objects.append(match)
        return matched_objects

    def _register_step(self,
                       frame_actual: int,
                       objects_actual: List[Object],
                       matches: Optional[List[MatchedObject]]) -> None:
        """Paso de registro de objetos.

        Registra los objetos que no han sido emparejados. Además, si se ha indicicado una región
        donde registrar objetos únicamente, el registro se limitará a ella.

        :param frame_actual: identificador del frame actual.
        :param objects_actual: lista de objetos del frame actual.
        :param matches: lista de emparejamientos realizados o None si no hubo ninguno.
        :return: None.
        """
        # Si se han producido matches, quitar de los objetos propuestos para ser registrados.
        if matches is None:
            remaining_objects = objects_actual
        else:
            remaining_objects = list(set(objects_actual) - set([match.actual for match in matches]))
        # Filtrar los objetos que están dentro de la máscara de registro (si la hay).
        if self.register_mask_region is not None:
            remaining_objects = [obj for obj in remaining_objects
                                 if self.register_mask_region[obj.center[1], obj.center[0]].all()]
        # Registrar los objetos restantes.
        for object_ in remaining_objects:
            self.objects.register_object(object_, frame_actual)

    def _unregister_step(self, frame_actual: int) -> None:
        """Paso de desregistro de los objetos.

        Aquellos objetos que llevan sin verse un número de frames indicado en el modelo de
        seguimiento, se cambiará su estado a *desregistrado* para así no tenerlos en cuenta en la
        fase del emparejamiento.

        :param frame_actual: identificador del frame actual.
        :return: None.
        """
        max_frames_missing = self.frames_to_unregister_missing_objects
        self.objects.unregister_missing_objects(frame_actual, max_frames_missing)

    def _algorithm(self) -> None:
        """Algoritmo de seguimiento:

        #. Para cada frame de la secuencia:
            #. Emparejar los objetos del frame actual con alguno de los objetos que constan como
               registrados.
            #. Registrar los objetos que no han podido ser emparejados.
            #. Desregistrar los objetos que llevan sin detectarse un número de frames determinado.

        :return: None.
        """
        t = tqdm(total=len(self.sequence), desc='PointTracker')
        for frame_actual in range(0, len(self.sequence)):
            objects_actual = self.frame_objects(frame_actual)
            # 1. Emparejar.
            matches = self._matching_step(frame_actual, objects_actual)
            # 2. Registrar de los objetos no emparejados.
            self._register_step(frame_actual, objects_actual, matches)
            # 3. Desregistrar de los objetos desaparecidos.
            self._unregister_step(frame_actual)
            # Actualizar tqdm.
            t.update()


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
                       objs_registered: List[TrackedObjectDetection],
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
        # Mejora: ordenar los objetos detectados en el frame actual por puntuación, así ante un
        # empate futuro, se emparejará con el que más puntuación haya obtenido.
        objs_actual.sort(reverse=True, key=lambda obj: obj.score)
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
        for object_detected in self.frame_objects(0):
            self.objects.register_object(object_detected, 0)
        # Paso 2. Emparejar, registrar, y desregistrar objetos en el resto de frames.
        t = tqdm(total=len(self.sequence), desc='CentroidTracker')
        for frame_actual in range(1, len(self.sequence)):
            # 1. Emparejar objetos.
            # Objetos registrados hasta el momento.
            objs_registered = self.objects.registered_objects()
            objs_registered_uids_matched = list()
            # Objetos detectados en el frame actual, candidatos para emparejar con alguno de los
            # registrados.
            objs_actual = self.frame_objects(frame_actual)
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
            t.update()
