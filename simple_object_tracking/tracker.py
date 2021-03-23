from abc import ABC, abstractmethod
from simple_object_detection.utils import *


class ObjectTracker(ABC):
    """Clase abstracta para la implementación de modelos de seguimiento.

    Para implementar un modelo se necesitará sobreescribir únicamente el método _algorithm(self).
    """

    def __init__(self,
                 sequence_with_information,
                 object_detector=None,
                 object_detections=None,
                 objects_min_score=0,
                 objects_classes=None,
                 objects_avoid_duplicated=False,
                 frames_to_unregister_object=10,
                 *args,
                 **kwargs):
        """

        :param sequence_with_information: información de la secuencia y lista de frames
        con el formato de tupla (width, height, fps, frames).
        :param object_detector: detector de objetos inicializado.
        :param object_detections: lista con las detecciones de objetos realizadas en los frames.
        :param objects_min_score: puntuación mínima que deben tener los objetos detectados.
        :param objects_classes: clases a las que deben pertener los objetos detectados.
        :param objects_avoid_duplicated: evitar que un mismo objeto se detecte múltiples veces.
        :param frames_to_unregister_object: número de frames sin detectar un objeto tras los cuáles
        se elimina del seguimiento.
        :param args:
        :param kwargs:
        """
        # Lista de frames (secuencia) e información sobre la secuencia (ancho, alto, fps).
        self.frame_width, self.frame_height, self.fps, self.sequence = sequence_with_information
        # Comprobar que se ha pasado un detector de objetos o la lista con las detecciones.
        assert_msg = 'You must provide an object detector or a file with detections.'
        assert object_detector or object_detections, assert_msg
        # Detector de objetos.
        self.object_detector = object_detector
        # Lista de detecciones de objetos con los frames por índice.
        self.object_detections = object_detections
        # Asegurar que hay detecciones de tantos frames como secuencias.
        assert_msg = 'The sequence and objects detections length must be the same.'
        assert object_detections and len(object_detections) == len(self.sequence), assert_msg
        # Opciones de filtrado para la obteción de los objetos en un frame.
        self.objects_min_score = objects_min_score
        self.objects_classes = objects_classes
        self.objects_avoid_duplicated = objects_avoid_duplicated
        # Estructura de datos de los objetos registrados.
        self.registered_objects = self.ObjectsRegistered(frames_to_unregister_object)

    def objects_in_frame(self, frame_id):
        """Método para la obtención de los objetos en un frame.

        Si las detecciones se encuentran precalculadas o precargadas, se obtienen directamente del
        atributo de instancia self.object_detections, en caso de que este esté vacío, se calculan
        en la propia llamada a este método (esto puede ser bastante lento y costoso).

        :param frame_id: índice del frame del que se quieren extraer los objetos.
        :return: ndarray de objetos detectados.
        """
        objects = None
        # Si se han cargado ya las detecciones en todos los frames, utilizarlas.
        if self.object_detections:
            objects = self.object_detections[frame_id]
        else:
            frame = self.sequence[frame_id]
            objects = self.object_detector.get_objects(frame)
        # Realizar los filtrados.
        if self.objects_min_score:
            objects = filter_objects_by_min_score(objects, self.objects_min_score)
        if self.objects_classes:
            objects = filter_objects_by_classes(objects, self.objects_classes)
        if self.objects_avoid_duplicated:
            objects = filter_objects_avoiding_duplicated(objects)
        return objects

    @abstractmethod
    def _algorithm(self):
        raise NotImplemented()

    def run(self):
        """Ejecuta el algoritmo de seguimiento y calcula el registro de seguimiento de los objetos.
        """
        self._algorithm()

    class ObjectsRegistered:
        """Estructura para el almacenamiento y manejo de los objetos registrados
        durante el seguimiento de objetos en una secuencia.
        """

        def __init__(self, frames_to_unregister_object):
            # Contador para la asignación de índices únicos.
            self.next_uid = 0
            # Objetos diferentes registrados.
            self.objects = list()
            # Identificador único del objeto i-ésimo.
            self.objects_uid = list()
            # Último frame en el que el objeto i-ésimo fue visto.
            self.last_frame = list()
            # Histórico del frame y los objetos detectados que se han actualizado para ese objeto.
            self.history = list()
            # Identificadores únicos de los objetos que han sido desregistrados.
            self.unregistered_objects = list()
            # Número de frames tras los que se eliminará un objeto que lleva sin ser visto.
            self.frames_to_unregister_object = frames_to_unregister_object

        def objects_with_uid(self, unregistered=False):
            """Devuelve todos los objetos registrados, excepto los desregistrados.

            :param unregistered: indica si se quieren devolver los objetos desregistrado también.
            Por defecto se encuentra en False.
            :return: lista de tuplas de identificador único y objeto.
            """
            # Realizar las tuplas (uid, obj).
            objects_with_uid = zip(self.objects_uid, self.objects)
            # Filtrar las tuplas de objetos que se mantienen registrados.
            if not unregistered:
                # Función para filtrar solo los objetos que se mantienen registrados.
                def object_with_uid_not_unregistered(obj_with_uid):
                    uid, obj = obj_with_uid
                    return uid not in self.unregistered_objects
                objects_with_uid = list(filter(object_with_uid_not_unregistered, objects_with_uid))
            return objects_with_uid

        def register_object(self, obj, frame_id):
            """Registra un objeto en la estructura.

            :param obj: objeto detectado.
            :param frame_id: frame en el que se detectó.
            """
            # Almacenar el objeto en la estructura.
            self.objects.append(obj)
            self.objects_uid.append(self.next_uid)
            self.last_frame.append(frame_id)
            self.history.append([(frame_id, obj)])
            # Incrementar el identificador único siguiente.
            self.next_uid += 1

        def update_object(self, obj, object_uid, frame_id):
            """Actualiza el objeto con identificador único object_uid.

            :param obj: objeto detectado.
            :param object_uid: identificador del objeto a actualizar.
            :param frame_id: índice del frame en el que se ha visto por última vez.
            """
            # Comprobar que no se está intentando actualizar un objeto desregistrado.
            assert_msg = 'The object {} was unregistered and can\'t be updated.'.format(object_uid)
            assert object_uid not in self.unregistered_objects, assert_msg
            # Actualizar el seguimiento del objeto.
            self.objects[object_uid] = obj
            self.last_frame[object_uid] = frame_id
            self.history[object_uid].append((frame_id, obj))

        def unregister_dissapeared_objects(self, frame_id):
            """Elimina los objetos registrados que llevan desaparecidos
            self.frames_to_unregister_object cantidad de frames.

            :param frame_id: frame en el que se encuentra actualmente.
            """
            # Recorrer todos los objetos registrados.
            for obj_uid, last_frame in zip(self.objects_uid, self.last_frame):
                # Ignorar los objetos desregistrados.
                if obj_uid not in self.unregistered_objects:
                    # Comprobar si han pasado los frames indicados desde la última vez que se vio.
                    frames_elapsed = frame_id - last_frame
                    if frames_elapsed >= self.frames_to_unregister_object:
                        self.unregistered_objects.append(obj_uid)

        def __str__(self):
            or_str = ''
            for obj_uid, obj in self.objects_with_uid():
                obj_str = 'UniqueId({}). {}'.format(obj_uid, obj)
                or_str = '\n'.join([or_str, obj_str])
            return or_str
