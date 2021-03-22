import numpy as np
from abc import ABC, abstractmethod
from simple_object_detection.utils import filter_objects


class ObjectTracker:
    """



    EN DESARROLLO
    ----------------------------------
    Hay que mantener un histórico del objeto a lo largo del tiempo.
    Saber cuántos frames hay por cada segundo.

    A tener en cuenta:
    - Comprobar que el detector de objetos se ha introducido ya inicializado,
      puesto que algunos pueden tener parámetros especiales o lo qu sea.
    - Filtrar por confianza!!!
    - Se puede detectar un objeto de forma múltiple con diferentes puntuaciones,
      habría que elegir el que más alta puntuación tenga.
    - No tienen por qué emparejarse todos los objetos.

    Posibles parámetros:
    - Lista de modelos de detección, para realizar la detección con más de
      una red.

    - Intervalo de detección. Realizar la detección+matching cada N frames.

    - Puntuación mínima que deben tener los objetos.

    - Clases de los objetos que se quieren hacer seguimiento.

    """

    def __init__(self,
                 sequence,
                 object_detector=None,
                 object_detections=None,
                 objects_min_score=0.10,
                 objects_classes=None,
                 objects_avoid_duplicated=False,
                 frames_to_unregister_object=5,
                 *args,
                 **kwargs):
        """

        :param sequence: lista de imágenes (frames).
        :param object_detector: detector de objetos inicializado.
        :param object_detections: lista con las detecciones de objetos realizadas en los frames.
        :param objects_min_score: puntuación mínima que deben tener los objetos detectados.
        :param objects_classes: clases a las que deben pertener los objetos detectados.
        :param objects_avoid_duplicated: evitar que un mismo objeto se detecte múltiples veces.
            # TODO: Esta opción requiere implementar SOD-8.
        :param frames_to_unregister_object: número de frames sin detectar un objeto tras los cuáles
        se elimina del seguimiento.
        :param args:
        :param kwargs:
        """
        # Lista de frames (secuencia).
        self.sequence = sequence
        # Comprobar que se ha pasado un detector de objetos o la lista con las detecciones.
        assert_msg = 'You must provide an object detector or a file with detections.'
        assert object_detector or object_detections, assert_msg
        # Detector de objetos.
        self.object_detector = object_detector
        # Lista de detecciones de objetos con los frames por índice.
        self.object_detections = object_detections
        # Asegurar que hay detecciones de tantos frames como secuencias.
        assert_msg = 'The sequence and objects detections length must be the same.'
        assert object_detections and len(object_detections) == len(sequence), assert_msg
        self.objects_min_score = objects_min_score  # TODO Implementar SOT-2
        self.objects_classes = objects_classes  # TODO Implementar SOT-2
        self.objects_avoid_duplicated = objects_avoid_duplicated  # TODO Implementar SOT-2
        # Información del vídeo introducido.
        self.frames_per_second = None  # TODO
        self.height, self.width, _ = self.sequence[0].shape  # TODO
        # Estructura de datos de los objetos registrados.
        self.registered_objects = self.ObjectsRegistered(frames_to_unregister_object)

    def objects_in_frame(self, frame_id):
        objects = None
        # Si se han cargado ya las detecciones en todos los frames, utilizarlas.
        if self.object_detections:
            objects = self.object_detections[frame_id]
        else:
            frame = self.sequence[frame_id]
            objects = self.object_detector.get_objects(frame)
        # TODO: SOT-2 here!!!
        # TODO: temporalmente se filtran los objetos con alta puntuación.
        # Realizar la mejora en el simple-object-detection para buscar un objeto en una
        # ventana, y así obtener el mejor objeto de esa ventana!!
        objects = filter_objects(objects, min_score=0.1)
        return objects

    @abstractmethod
    def _algorithm(self):
        return None

    def run(self):
        self._algorithm()

    class ObjectsRegistered:
        """Estructura para el almacenamiento y manejo de los objetos registrados
        durante el seguimiento de objetos en una secuencia.

        DESARROLLO
        ---------------------
        - Tener en cuenta si el objeto ha desaparecido, eliminar?
        - Mantener un histórico de los objetos detectados.
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
            # Identificadores únicos de los objetos que han sido desregistrados.
            self.unregistered_objects = list()
            # Número de frames tras los que se eliminará un objeto que lleva sin ser visto.
            self.frames_to_unregister_object = frames_to_unregister_object

        def objects_with_uid(self, unregistered=False):
            """Devuelve todos los objetos registrados, excepto los desregistrados.

            :param unregistered: TODO
            :return: lista de tuplas de identificador único y objeto.
            """
            # Función para filtrar solo los objetos que se mantienen registrados.
            def object_with_uid_not_unregistered(obj_with_uid):
                uid, obj = obj_with_uid
                return uid not in self.unregistered_objects
            # Realizar las tuplas (uid, obj).
            objects_with_uid = zip(self.objects_uid, self.objects)
            # Filtrar las tuplas de objetos que se mantienen registrados.
            return list(filter(object_with_uid_not_unregistered, objects_with_uid))

        def register_object(self, obj, frame_id):
            # Almacenar el objeto en la estructura.
            self.objects.append(obj)
            self.objects_uid.append(self.next_uid)
            self.last_frame.append(frame_id)
            # Incrementar el identificador único siguiente.
            self.next_uid += 1

        def update_object(self, obj, object_uid, frame_id):
            """Actualiza el objeto con identificador único object_uid.

            :param obj: objeto detectado.
            :param object_uid: identificador del objeto a actualizar.
            :param frame_id: índice del frame en el que se ha visto por última vez.
            :return:
            """
            # Comprobar que no se está intentando actualizar un objeto desregistrado.
            assert_msg = 'The object {} was unregistered and can\'t be updated.'.format(object_uid)
            assert object_uid not in self.unregistered_objects, assert_msg
            # Actualizar el seguimiento del objeto.
            self.objects[object_uid] = obj
            self.last_frame[object_uid] = frame_id

        def unregister_dissapeared_objects(self, frame_id):
            """Elimina los objetos registrados que llevan desaparecidos
            self.frames_to_unregister_object cantidad de frames.

            :param frame_id: frame en el que se encuentra actualmente.
            :return:
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
