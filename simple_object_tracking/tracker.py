from abc import ABC, abstractmethod
from typing import List, Callable

from simple_object_detection.utils import StreamSequence
from simple_object_detection.object import Object

from simple_object_tracking.datastructures import TrackedObjects


class ObjectTracker(ABC):
    """Clase abstracta para la implementación de modelos de seguimiento.

    Para implementar un modelo se necesitará sobreescribir únicamente el método _algorithm(self).
    """

    def __init__(self,
                 sequence: StreamSequence,
                 objects_detections: List[List[Object]],
                 objects_filters: List[Callable[[List[Object]], List[Object]]] = None,
                 frames_to_unregister_missing_objects: int = 10,
                 *args,
                 **kwargs):
        """

        :param sequence: secuencia de vídeo e información de ella.
        :param objects_detections: detecciones de objetos cargada.
        :param objects_filters: lista de filtros para aplicar sobre los objetos detectados.
        :param frames_to_unregister_missing_objects: cantidad de frames para eliminar un objeto
        registrado.
        :param args:
        :param kwargs:
        """
        self.sequence = sequence
        self.objects_detections = objects_detections
        self.objects_filters = objects_filters or []
        self.frames_to_unregister_missing_objects = frames_to_unregister_missing_objects
        # Estructura de datos de los objetos almacenados.
        self.objects = TrackedObjects()

    def __str__(self):
        return f'Tracker({str(self.objects)}'

    def frame_objects(self, fid: int) -> List[Object]:
        """Método para obtener los objetos detectados en un frame.

        Además, se aplican los filtros pasados a la instancia con el parámetro ``objects_filters``.

        :param fid: índice del frame del que se quieren extraer los objetos.
        :return: lista de los objetos detectados en ese frame.
        """
        # Obtener los objetos del detector si no hay detecciones precargadas.
        objects_in_frame = self.objects_detections[fid]
        # Aplicar los filtros a los objetos.
        for filter_function in self.objects_filters:
            objects_in_frame = filter_function(objects_in_frame)
        return objects_in_frame

    def run(self) -> None:
        """Ejecuta el algoritmo de seguimiento y calcula el registro de seguimiento de los objetos.
        """
        self._algorithm()

    @abstractmethod
    def _algorithm(self) -> None:
        """Método a implementar que realiza el algoritmo del modelo de seguimiento.

        Este método es llamado por run(). No se espera que devuelva nada.
        """
