from typing import List, Optional, NamedTuple

from simple_object_detection.object import Object

from simple_object_tracking.exceptions import SimpleObjectTrackingException


class TrackedObjectDetection(NamedTuple):
    """Detección de un objeto en su seguimiento."""
    id: int
    frame: int
    object: Object


class TrackedObject:
    """Estructura que almacena la información de un objeto seguido a lo largo de una secuencia.
    """

    def __init__(self, obj_id: int, status: bool, ini_frame: int, ini_obj: Object):
        """

        :param obj_id: identificador del objeto seguido.
        :param status: estado del objeto en el seguimiento.
        :param ini_frame: frame inicial.
        :param ini_obj: detección inicial.
        """
        self.id = obj_id
        self.status = status
        self.frames = [ini_frame]
        self.detections = [ini_obj]

    def __getitem__(self, item: int) -> TrackedObjectDetection:
        """Devuelve el frame y la detección del objeto registrada.

        :param item: posición del registro del objeto.
        :return: tupla del frame y detección del objeto.
        """
        if item >= len(self):
            raise IndexError(f'El índice {item} no está registrado para el objeto {self.id}')
        return TrackedObjectDetection(self.id, self.frames[item], self.detections[item])

    def __len__(self) -> int:
        """Cantidad de veces que ha sido detectado el objeto.

        :return: número de seguimientos del objeto.
        """
        return len(self.detections)

    def __str__(self) -> str:
        return f'TrackedObject(id={self.id}, status={self.status}, detections: {len(self)})'

    def append(self, frame: int, obj: Object) -> None:
        """Añade un registro al seguimiento del objeto.

        :param frame: frame en el que fue visto.
        :param obj: objeto detectado.
        :return: None
        """
        self.frames.append(frame)
        self.detections.append(obj)

    def find_in_frame(self, frame: int) -> Optional[TrackedObjectDetection]:
        """Busca la detección del objeto en el frame indicado.

        :param frame: número del frame.
        :return: el objeto si fue encontrado o None.
        """
        try:
            index = self.frames.index(frame)
        except ValueError:
            return None
        return TrackedObjectDetection(self.id, self.frames[index], self.detections[index])

    def interpolate_positions(self):
        """TODO: Realiza la interpolación de las posiciones entre la primera y la última.
            Planear cómo se haría para funcionar con los métodos externos que usan el objeto.
        """


class TrackedObjects:
    """Estructura de datos que almacena y gestiona los objetos detectados en una secuencia de
    vídeo.
    """

    def __init__(self):
        # Gestion interna de la estructura.
        self._next_uid = 0
        self._tracked_objects: List[TrackedObject] = []

    def __len__(self):
        """Cantida de objetos almacenados (tanto registrados como ya desregistrados).

        :return: número de objetos almacenados.
        """
        return len(self._tracked_objects)

    def __getitem__(self, item: int) -> TrackedObject:
        """Busca un objeto por su uid.

        :param item: índice del objeto seguido.
        :return: instancia del objeto seguido.
        """
        if item >= len(self._tracked_objects):
            raise IndexError(f'El objeto {item} no se encuentra registrado.')
        return self._tracked_objects[item]

    def __setitem__(self, key: int, value: TrackedObject) -> None:
        """Modifica el objeto almacenado.

        No se permiten añadir nuevos objetos fuera de los índices ya establecidos. Para añadir
        nuevos objetos usar el método ``register_object``.

        :param key: índice del objeto.
        :param value: nueva instancia del objeto seguido.
        :return: None.
        """
        if key >= len(self._tracked_objects):
            raise IndexError(f'El objeto {key} no se encuentra registrado.')
        self._tracked_objects[key] = value

    def __str__(self) -> str:
        return f'TrackedObjects({len(self)} objects registered).'

    def register_object(self, obj: Object, frame: int) -> bool:
        """Registra un objeto.

        :param obj: objeto detectado.
        :param frame: frame en el que se detectó.
        :return: si se pudo registrar con éxito.
        """
        stored_object = TrackedObject(
            obj_id=self._next_uid_and_increment(),
            status=True,
            ini_frame=frame,
            ini_obj=obj
        )
        self._tracked_objects.append(stored_object)
        return True

    def update_object(self, obj: Object, obj_id: int, frame: int) -> bool:
        """Actualiza un objeto dada su identificador único (uid), el nuevo objeto detectado, y el
        frame en el que fue visto.

        :param obj: objeto detectado.
        :param obj_id: identificador único del objeto.
        :param frame: identificador del frame en el que fue visto.
        :return: true si el objeto fue actualizado correctamente.
        """
        # Comprobar que el objeto está ya insertado.
        if len(self._tracked_objects) < obj_id:
            raise SimpleObjectTrackingException(f'The object uid {obj_id} is not registered.')
        tracked_object = self._tracked_objects[obj_id]
        tracked_object.append(frame, obj)
        return True

    def unregister_missing_objects(self, frame, max_frames_missing: int) -> None:
        """Elimina los objetos que han desaparecido durante una cantidad de frames mayor que la
        indicada por parámetro.

        :param frame: identificador del frame actual.
        :param max_frames_missing: cantidad de frames para borrar los objetos.
        :return: lista de tuplas (uid, objeto) desaparecidos eliminados.
        """
        # Analizar la situación de cada objeto.
        for tracked_object in self._tracked_objects:
            frames_elapsed = frame - tracked_object[-1].frame
            # Si ha desaparecido una cantidad de frames mayor que la indicada, cambiar el estado de
            # registro a False. Esto indicará que el objeto está desregistrado.
            if frames_elapsed > max_frames_missing:
                tracked_object.status = False

    def frame_objects(self, frame: int) -> List[TrackedObjectDetection]:
        """Crea una lista de los objetos que hay en un frame.

        :param frame: número del frame del que se quiere obtener los objetos registrados.
        :return: lista de pares de identificador del objeto y objeto.
        """
        objects_in_frame: List[TrackedObjectDetection] = list()
        for object_tracked in self._tracked_objects:
            detection = object_tracked.find_in_frame(frame)
            if detection is not None:
                objects_in_frame.append(detection)
        return objects_in_frame

    def tracked_objects(self) -> List[TrackedObject]:
        """Devuelve la lista de objetos seguidos.

        :return: lista de objetos seguidos.
        """
        return self._tracked_objects

    def registered_objects(self) -> List[TrackedObjectDetection]:
        """Devuelve la lista de los objetos registrados con el último frame en el que fue visto.

        Únicamente se devuelven los objetos cuyo estado esté mercado como registrado.

        :return: lista de (último frame visto, objeto).
        """
        registered_objects = []
        for obj in self._tracked_objects:
            if obj.status:
                registered_objects.append(obj[-1])
        return registered_objects

    def purge_objects(self, ids: List[int]) -> None:
        """Elimina los objetos indicados y restablece los ids para que mantengan un orden
        secuencial.

        :param ids: identificadores de los objetos seguidos a eliminar.
        :return: None.
        """
        for id_ in ids:
            # Eliminar el objeto de la estructura
            self._tracked_objects.pop(id_)
            # Reajustar los índices.
            for tracked_object in self._tracked_objects[id_:]:
                tracked_object.id -= 1

    def remove_objects_with_less_detections_than(self, minimum_detections: int) -> List[int]:
        """Elimina los objetos que tienen un número de detecciones menor al indicado.

        :param minimum_detections: número mínimo de detecciones que deben tener los objetos.
        :return: identificadores de los objetos eliminados.
        """
        to_remove = [tracked_object.id for tracked_object in self._tracked_objects
                     if len(tracked_object) < minimum_detections]
        self.purge_objects(to_remove)
        return to_remove

    def _next_uid(self) -> int:
        """Devuelve el uid siguiente para asignar.

        :return: identificador único siguiente.
        """
        return self._next_uid

    def _next_uid_and_increment(self) -> int:
        """Devuelve el uid siguiente para asignar e incrementa el contador para el siguiente.

        :return: identificador único siguiente.
        """
        actual_uid = self._next_uid
        self._next_uid += 1
        return actual_uid
