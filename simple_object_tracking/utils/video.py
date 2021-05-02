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
        ¿Cómo plantearla? ¿Clase para dibujar frame a frame? ¿ esta dibuja, guarda y todo?
    """
    def __init__(self, tracker: ObjectTracker):
        self.tracker = tracker
        self._properties: Set[TrackingVideoProperty] = set()
        self._objects_colors = np.random.uniform(0, 255, size=(len(tracker.objects), 3))

    def _draw_object_bounding_box(self,
                                  fid: int,
                                  frame: Image,
                                  tracked_obj: TrackedObject) -> Image:
        # Buscar el objeto en la lista de seguimientos en el frame indicado.
        obj_in_frame = tracked_obj.find_in_frame(fid)
        # Si el objeto no está en ese frame, devolver el frame sin dibujar. Si no, copiar el frame.
        if obj_in_frame is None:
            return frame
        frame = frame.copy()
        # Obtener la bounding box.
        bounding_box = obj_in_frame.object.bounding_box
        # Dibujar sobre el frame.
        color = self._objects_colors[tracked_obj.id]
        cv2.line(frame, bounding_box.top_left, bounding_box.top_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.top_right, bounding_box.bottom_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_right, bounding_box.bottom_left, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_left, bounding_box.top_left, color, 2, cv2.LINE_AA)
        return frame

    def _draw_bounding_boxes(self, fid: int, frame: Image) -> Image:
        for tracked_obj in self.tracker.objects:
            frame = self._draw_object_bounding_box(fid, frame, tracked_obj)
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
