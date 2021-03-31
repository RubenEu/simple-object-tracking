from abc import ABC, abstractmethod
from typing import List

from simple_object_detection.detection_model import DetectionModel

from simple_object_tracking.datastructures import SequenceObjects
from simple_object_tracking.exceptions import SimpleObjectTrackingException
from simple_object_tracking.typing import (SequenceInformation,
                                           SequenceObjectsDetections,
                                           ObjectsFilterFunction,
                                           FrameID,
                                           Object)


class ObjectTracker(ABC):
    """Clase abstracta para la implementación de modelos de seguimiento.

    Para implementar un modelo se necesitará sobreescribir únicamente el método _algorithm(self).
    """

    def __init__(self,
                 sequence_with_information: SequenceInformation,
                 object_detector: DetectionModel = None,
                 object_detections: SequenceObjectsDetections = None,
                 objects_filters: List[ObjectsFilterFunction] = None,
                 frames_to_unregister_missing_objects: int = 10,
                 *args,
                 **kwargs):
        """

        :param sequence_with_information: información sobre la secuencia y lista de frames.
        :param object_detector: modelo de detección de objetos.
        :param object_detections: detecciones de objetos cargada.
        :param objects_filters: lista de filtros para aplicar sobre los objetos detectados.
        :param frames_to_unregister_missing_objects: cantidad de frames para eliminar un objeto
        registrado.
        :param args:
        :param kwargs:
        """
        (self.frame_width, self.frame_height,
         self.fps, self.sequence, self.timestamps) = sequence_with_information
        self.object_detector = object_detector
        self.object_detections = object_detections
        self.objects_filters = objects_filters
        self.frames_to_unregister_missing_objects = frames_to_unregister_missing_objects
        # Estructura de datos de los objetos almacenados.
        self.objects = SequenceObjects(len(self.sequence), self.fps, self.timestamps)
        # Comprobaciones
        if object_detector is None and object_detections is None:
            raise SimpleObjectTrackingException('You must provide and object detector or list with '
                                                'detections per frame preloaded.')

    def objects_in_frame(self, frame_id: FrameID) -> List[Object]:
        """Método para la obtención de los objetos en un frame.

        Si las detecciones se encuentran cargadas, se leen directamente de memoria, en caso
        contrario se realiza la detección de los objetos en ese frame en el mismo instante.

        Después se aplica la lista de funciones filtro de objetos.

        Es recomendable usar el método de instancia: preload_objects() para calcular todas las
        detecciones a lo largo de la secuencia.

        :param frame_id: índice del frame del que se quieren extraer los objetos.
        :return: lista de los objetos detectados en ese frame.
        """
        objects_in_frame = self.object_detections[frame_id]
        # Comprobar que se ha establecido una lista de filtros.
        if isinstance(self.objects_filters, list):
            for filter_function in self.objects_filters:
                objects_in_frame = filter_function(objects_in_frame)
        return objects_in_frame

    def preload_objects(self) -> None:
        """Realiza la detección de todos los objetos a lo largo de la secuencia y los almacena en
        el atributo de instancia self.object_detections.
        """
        ...  # TODO

    def run(self) -> None:
        """Ejecuta el algoritmo de seguimiento y calcula el registro de seguimiento de los objetos.
        """
        self._algorithm()

    @abstractmethod
    def _algorithm(self) -> None:
        """Método a implementar que realiza el algoritmo del modelo de seguimiento.

        Este método es llamado por run(). No se espera que devuelva nada.
        """
        ...
