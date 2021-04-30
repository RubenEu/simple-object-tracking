from typing import List, Tuple

from simple_object_detection.object import Object
from simple_object_detection.typing import Point2D
from simple_object_detection.utils import Sequence

from simple_object_tracking.exceptions import SimpleObjectTrackingException
from simple_object_tracking.typing import (ObjectTracking,
                                           ObjectWithUID,
                                           ObjectWithUIDFrame,
                                           ObjectHistory)


class SequenceObjects:
    """Estructura de datos que almacena y gestiona los objetos detectados en una secuencia de
    vídeo.

    TODO: Mejorar la clase. Plantear si listas enlazadas o arraylists
      Eliminar tanta tupla... usar mejor clases o namedtuples.
    """
    def __init__(self, sequence: Sequence):
        """

        :param sequence: secuencia de vídeo con la información de él.
        """
        self.sequence_length = len(sequence)
        self.sequence_fps = sequence.fps
        # Gestion interna de la estructura.
        self._next_uid = 0
        # Lista con la información de los objetos almacenados.
        # Se almacena una lista de objetos distintos (únicos) con su historial de detecciones con el
        # frame y el objeto asociado.
        self._stored_objects: List[ObjectTracking] = list()

    def register_object(self, object_detected: Object, frame_id: int) -> bool:
        """
        Registra un objeto.

        Se asegura además, que el objeto va a ser insertado en la posición uid-ésima. Es decir, el
        uid del objeto corresponderá con la posición en la lista de almacenamiento interna
        '_registered_objects'.

        :param object_detected: objeto detectado.
        :param frame_id: frame en el que se detectó.
        :return: si se pudo registrar con éxito.
        """
        stored_object = (self.next_uid_and_increment(), True, [(frame_id, object_detected)])
        self._stored_objects.append(stored_object)
        return True

    def update_object(self, object_detected: Object, object_uid: int, frame_id: int) -> bool:
        """Actualiza un objeto dada su identificador único (uid), el nuevo objeto detectado, y el
        frame en el que fue visto.

        :param object_detected: objeto detectado.
        :param object_uid: identificador único del objeto.
        :param frame_id: identificador del frame en el que fue visto.
        :return: true si el objeto fue actualizado correctamente.
        """
        # Comprobar que el objeto está ya insertado.
        if len(self._stored_objects) < object_uid:
            raise SimpleObjectTrackingException(f'The object uid {object_uid} is not in the data '
                                                'structure')
        uid, registered_status, object_history = self._stored_objects[object_uid]
        object_history.append((frame_id, object_detected))
        return True

    def unregister_missing_objects(self, frame_id, frames_missing: int) -> None:
        """Elimina los objetos que han desaparecido durante una cantidad de frames mayor que la
        indicada por parámetro.

        :param frame_id: identificador del frame actual.
        :param frames_missing: cantidad de frames para borrar los objetos.
        :return: lista de tuplas (uid, objeto) desaparecidos eliminados.
        """
        # Analizar la situación de cada objeto.
        for stored_object in self._stored_objects:
            uid, status, history = stored_object
            last_frame_seen, last_object_detection = history[-1]
            frames_elapsed = frame_id - last_frame_seen
            # Si ha desaparecido una cantidad de frames mayor que la indicada, cambiar el estado de
            # registro a False. Esto indicará que el objeto está desregistrado.
            if frames_elapsed > frames_missing:
                self._stored_objects[uid] = (uid, False, history)

    def objects_frame(self, frame_id: int) -> List[ObjectWithUID]:
        """Crea una lista de los objetos que hay en un frame.

        :param frame_id: identificador del frame del que se quiere obtener los objetos registrados.
        :return: lista de pares de identificador del objeto y objeto.
        """
        objects_in_frame: List[ObjectWithUID] = list()
        for object_stored in self._stored_objects:
            uid, state, history = object_stored
            # Buscar cuál de los objetos en el historial está en ese frame.
            object_history_index = 0
            object_history_found = False
            # Buscar cuál de los objetos guardadas en el historial pertenece al frame indicado.
            while not object_history_found and object_history_index < len(history):
                object_detected_frame_id, object_detected = history[object_history_index]
                if frame_id == object_detected_frame_id:
                    objects_in_frame.append((uid, object_detected))
                    object_history_found = True
                # Pasar al siguiente objeto del historial.
                object_history_index += 1
        return objects_in_frame

    def object_uid(self, uid: int) -> ObjectHistory:
        """
        Busca un objeto por su uid.

        :param uid: identificador único del objeto.
        :return: la información del objeto [(frame, objeto)] o error si no existe.
        """
        uid, state, history = self._stored_objects[uid]
        return history

    def object_frame_positions(self, object_uid: int) -> List[Tuple[int, Point2D]]:
        """
        Obtiene las posiciones en las que estuvo el objeto en cada frame en el que fue registrado.

        :param object_uid: identificador del objeto.
        :return: lista de pares (frame, posición).
        """
        object_history = self.object_uid(object_uid)
        return [(frame, obj.center) for frame, obj in object_history]

    def objects(self) -> List[ObjectWithUIDFrame]:
        """
        Devuelve la lista de los objetos registrados con el último frame en el que fue visto.

        Únicamente se devuelven los objetos cuyo estado esté mercado como registrado.

        :return: lista de (último frame visto, objeto).
        """
        objects = list()
        for uid, status, history in self._stored_objects:
            # Añadir solo si está como registrado:
            if status:
                last_frame_seen, last_object_detection = history[-1]
                objects.append((uid, last_frame_seen, last_object_detection))
        return objects

    def next_uid(self) -> int:
        """Devuelve el uid siguiente para asignar.

        :return: identificador único siguiente.
        """
        return self._next_uid

    def next_uid_and_increment(self) -> int:
        """Devuelve el uid siguiente para asignar e incrementa el contador para el siguiente.

        :return: identificador único siguiente.
        """
        actual_uid = self._next_uid
        self._next_uid += 1
        return actual_uid

    def __len__(self):
        """Cantida de objetos almacenados (tanto registrados como ya desregistrados).

        :return: número de objetos almacenados.
        """
        return len(self._stored_objects)
