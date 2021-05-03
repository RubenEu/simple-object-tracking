import cv2
import numpy as np
from typing import Callable, Dict, Any
from enum import Enum

from simple_object_detection.typing import Image
from simple_object_detection.utils import StreamSequence
from simple_object_detection.utils.video import StreamSequenceWriter
from simple_object_tracking.datastructures import TrackedObjectDetection, TrackedObject, \
    TrackedObjects


class TrackingVideoProperty(Enum):
    DRAW_BOUNDING_BOXES = 1
    DRAW_TRACES = 2


class TrackingVideo:
    """Clase para el dibujado de información sobre un vídeo con seguimiento de objetos.
    """
    def __init__(self, tracked_objects: TrackedObjects, input_sequence: StreamSequence):
        """

        :param tracked_objects: estructura de datos con el seguimiento de los objetos.
        :param input_sequence: secuencia de vídeo.
        """
        self.tracked_objects = tracked_objects
        self.input_sequence = input_sequence
        self._properties: Dict[TrackingVideoProperty, Any] = {}
        self._objects_colors = np.random.uniform(0, 255, size=(len(tracked_objects), 3))
        self._functions = []

    def __getitem__(self, item: int) -> Image:
        """Obtiene el frame item-ésimo con los dibujados aplicados.

        Primeramente aplica las funciones y luego los dibujados internos.

        :param item: índice del frame.
        :return: frame con los dibujados aplicados.
        """
        frame = self.input_sequence[item].copy()
        # Aplicar funciones añadidas.
        for function in self._functions:
            frame = function(frame)
        # Aplicar dibujados internos.
        frame = self._apply_internal_drawings(item, frame)
        return frame

    def _apply_internal_drawings(self, fid: int, frame: Image) -> Image:
        """Aplica las funciones internas de dibujado.

        Son aplicadas a través de las propiedades.

        :param fid: número del frame.
        :param frame: frame.
        :return: frame con los dibujados.
        """
        # DRAW_BOUNDING_BOXES
        if self.get_property(TrackingVideoProperty.DRAW_BOUNDING_BOXES):
            frame = self._draw_objects_bounding_boxes(fid, frame)
        # DRAW TRACES
        if self.get_property(TrackingVideoProperty.DRAW_TRACES):
            frame = self._draw_objects_traces(fid, frame)
        return frame

    @staticmethod
    def _draw_object_trace(fid: int, frame: Image, tracked_obj: TrackedObject) -> Image:
        """Dibuja los trazados de un objeto hasta el frame en el que se encuentra.

        :param fid: número del frame.
        :param frame: frame.
        :param tracked_obj: estructura del seguimiento del objeto.
        :return: imagen con el seguimiento del objeto.
        """
        positions_centroid = [t_obj.object.center for t_obj in tracked_obj if t_obj.frame <= fid]
        # Dibujar cada una de las posiciones
        prev_position = positions_centroid[0]
        for position in positions_centroid:
            cv2.line(frame, position, prev_position, (255, 43, 155), 2, cv2.LINE_AA)
            cv2.circle(frame, position, 0, (107, 37, 74), 5, cv2.LINE_AA)
            prev_position = position
        return frame

    def _draw_objects_traces(self, fid: int, frame: Image) -> Image:
        """Dibujar todos los trazados de los objetos que aparecen en el frame hasta el frame en el
        que se encuentran.

        :param fid: número del frame.
        :param frame: frame.
        :return: frame con los trazados de los objetos aplicados.
        """
        # Obtener los objetos seguidos que aparecen en el frame fid.
        tracked_objects_ids = [tracked_object.id
                               for tracked_object in self.tracked_objects.frame_objects(fid)]
        # Dibujar los trazados de cada objeto hasta el frame fid-ésimo.
        for tracked_obj in self.tracked_objects:
            # Dibujar únicamente si se encuentra en el frame actual.
            if tracked_obj.id in tracked_objects_ids:
                frame = self._draw_object_trace(fid, frame, tracked_obj)
        return frame

    def _draw_object_bounding_box(self, frame: Image, tracked_obj: TrackedObjectDetection) -> Image:
        # Obtener la bounding box.
        bounding_box = tracked_obj.object.bounding_box
        # Dibujar sobre el frame.
        color = self._objects_colors[tracked_obj.id]
        cv2.line(frame, bounding_box.top_left, bounding_box.top_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.top_right, bounding_box.bottom_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_right, bounding_box.bottom_left, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_left, bounding_box.top_left, color, 2, cv2.LINE_AA)
        return frame

    def _draw_objects_bounding_boxes(self, fid: int, frame: Image) -> Image:
        # Objetos detectados en el frame fid.
        tracked_objects_in_frame = self.tracked_objects.frame_objects(fid)
        # Dibujar las bounding boxes de cada objeto en el frame fid.
        for tracked_obj in tracked_objects_in_frame:
            frame = self._draw_object_bounding_box(frame, tracked_obj)
        return frame

    def add_function(self, function: Callable[[Image], Image]) -> None:
        """Añade una función que recibe como argumento una imagen y devuelve una imagen.

        Esto puede ser útil para aplicar funciones de homografía, por ejemplo.

        :param function: función que recibe una imagen y devuelve una imagen.
        :return: None.
        """
        self._functions.append(function)

    def get_property(self, property_: TrackingVideoProperty) -> Any:
        """Devuelve el valor de una propiedad. Si no se encuentra, devuelve None.

        :param property_: propiedad.
        :return: valor de la propiedad o None si no existe.
        """
        return self._properties.get(property_)

    def set_property(self, property_: TrackingVideoProperty, value: Any) -> None:
        """Establece una propiedad a la creación del vídeo de seguimiento.

        :param property_: propiedad para añadir.
        :param value: valor de la propiedad.
        :return: None.
        """
        self._properties[property_] = value

    def remove_property(self, property_: TrackingVideoProperty):
        """Elimina una propiedad a la creación del vídeo de seguimiento.

        :param property_: propiedad para eliminar.
        :return: None.
        """
        self._properties.pop(property_)

    def properties(self) -> Dict[TrackingVideoProperty, Any]:
        """Devuelve la lista de propiedades.

        :return: diccionario con las propiedades.
        """
        return self._properties

    def add_object_information(self):
        ...

    def add_frame_information(self):
        ...

    def generate_video(self, file_output: str) -> None:
        """Genera la secuencia de vídeo y la guarda en un archivo.

        :param file_output: archivo de salida.
        :return: None.
        """
        output_stream = StreamSequenceWriter(file_output, self.input_sequence.properties())
        for frame in self:
            output_stream.write(frame)
