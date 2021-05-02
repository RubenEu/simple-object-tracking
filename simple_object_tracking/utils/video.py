import cv2
import numpy as np
from typing import Set
from enum import Enum

from simple_object_detection.typing import Image
from simple_object_detection.utils import StreamSequence
from simple_object_detection.utils.video import StreamSequenceWriter
from simple_object_tracking.datastructures import TrackedObjectDetection, TrackedObject

from simple_object_tracking.tracker import ObjectTracker


class TrackingVideoProperty(Enum):
    DRAW_BOUNDING_BOXES = 1


class TrackingVideo:
    """Clase para el dibujado de información sobre un vídeo con seguimiento de objetos.

    TODO:
      - Añadir a Youtrack y las funcionalidades referidas a esta clase.
    """
    def __init__(self, tracker: ObjectTracker):
        self.tracker = tracker
        self._properties: Set[TrackingVideoProperty] = set()
        self._objects_colors = np.random.uniform(0, 255, size=(len(tracker.objects), 3))

    def _draw_object_bounding_box(self, frame: Image, tracked_obj: TrackedObjectDetection) -> Image:
        # Copiar el frame para no editar el original.
        frame = frame.copy()
        # Obtener la bounding box.
        bounding_box = tracked_obj.object.bounding_box
        # Dibujar sobre el frame.
        color = self._objects_colors[tracked_obj.id]
        cv2.line(frame, bounding_box.top_left, bounding_box.top_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.top_right, bounding_box.bottom_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_right, bounding_box.bottom_left, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_left, bounding_box.top_left, color, 2, cv2.LINE_AA)
        return frame

    def _draw_bounding_boxes(self, fid: int, frame: Image) -> Image:
        # Objetos detectados en el frame fid.
        objects_in_frame = self.tracker.objects.frame_objects(fid)
        # Dibujar las bounding boxes de cada objeto en el frame fid.
        for tracked_obj in objects_in_frame:
            frame = self._draw_object_bounding_box(frame, tracked_obj)
        return frame

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
            # TODO: Aplicar cosos
            if TrackingVideoProperty.DRAW_BOUNDING_BOXES in self._properties:
                frame = self._draw_bounding_boxes(fid, frame)
            # ...
            # ...
            output_stream.write(frame)
