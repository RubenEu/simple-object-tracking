import cv2
import numpy as np
from typing import Set, Callable
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
    def __init__(self, tracked_objects: TrackedObjects, sequence: StreamSequence):
        """

        :param tracked_objects: estructura de datos con el seguimiento de los objetos.
        :param sequence: secuencia de vídeo.
        """
        self.tracked_objects = tracked_objects
        self.sequence = sequence
        self._properties: Set[TrackingVideoProperty] = set()
        self._objects_colors = np.random.uniform(0, 255, size=(len(tracked_objects), 3))
        self._functions = []

    def __getitem__(self, item: int) -> Image:
        """Obtiene el frame item-ésimo con los dibujados aplicados.

        Primeramente aplica las funciones y luego los dibujados internos.

        :param item: índice del frame.
        :return: frame con los dibujados aplicados.
        """
        frame = self.sequence[item].copy()
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
        if TrackingVideoProperty.DRAW_BOUNDING_BOXES in self._properties:
            frame = self._draw_objects_bounding_boxes(fid, frame)
        # DRAW TRACES
        if TrackingVideoProperty.DRAW_TRACES in self._properties:
            frame = self._draw_objects_traces(fid, frame)
        return frame

    def _draw_object_trace(self,
                           fid: int,
                           frame: Image,
                           tracked_obj: TrackedObject) -> Image:
        """Dibuja los trazados de un objeto hasta el frame en el que se encuentra.

        :param fid: número del frame.
        :param frame: frame.
        :param tracked_obj: estructura del seguimiento del objeto.
        :return: imagen con el seguimiento del objeto.
        """
        positions_centroid = [t_obj.object.center for t_obj in tracked_obj if t_obj.frame <= fid]
        # Si ese objeto no aparece en el frame, finalizar.
        if len(positions_centroid) == 0:
            return frame
        # Eliminación de trazados pasados N frames.

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
        # Dibujar los trazados de cada objeto hasta el frame fid-ésimo.
        for tracked_obj in self.tracked_objects:
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
        :return:
        """
        self._functions.append(function)

    def add_property(self, property_: TrackingVideoProperty) -> None:
        """Añade una propiedad a la creación del vídeo de seguimiento.

        :param property_: propiedad para añadir.
        :return: None.
        """
        self._properties.add(property_)

    def remove_property(self, property_: TrackingVideoProperty):
        """Elimina una propiedad a la creación del vídeo de seguimiento.

        :param property_: propiedad para eliminar.
        :return: None.
        """
        self._properties.remove(property_)

    def properties(self):
        """Devuelve la lista de propiedades.

        :return: lista de propiedades.
        """
        return list(self._properties)

    def add_object_information(self):
        ...

    def add_frame_information(self):
        ...

    def generate_video(self, sequence: StreamSequence, file_output: str) -> None:
        """Genera la secuencia de vídeo con las características especificadas.

        :param sequence: vídeo sobre el que se genera el seguimiento de los objetos.
        :param file_output: archivo de salida.
        :return: None.
        """
        output_stream = StreamSequenceWriter(file_output, sequence.properties())
        for fid, frame in enumerate(sequence):
            frame = self.__getitem__(fid)
            output_stream.write(frame)
