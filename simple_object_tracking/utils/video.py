import cv2
import numpy as np
from typing import Callable, Dict, Any
from enum import Enum
from tqdm import tqdm

from simple_object_detection.typing import Image
from simple_object_detection.utils import StreamSequence
from simple_object_detection.utils.video import StreamSequenceWriter
from simple_object_tracking.datastructures import (TrackedObjectDetection,
                                                   TrackedObject,
                                                   TrackedObjects)
from simple_object_tracking.utils.text import TextFormat


class TrackingVideoProperty(Enum):
    DRAW_OBJECTS = 0
    DRAW_OBJECTS_IDS = 1
    DRAW_OBJECTS_BOUNDING_BOXES = 2
    DRAW_OBJECTS_TRACES = 3
    DRAW_OBJECTS_BOUNDING_BOXES_TRACES = 4,
    DRAW_FRAME_NUMBER = 10,
    DRAW_FRAME_TIMESTAMP = 11,
    TEXT_OBJECT_INFORMATION = 100,
    TEXT_FRAME_INFORMATION = 101,


class TrackingVideo:
    """Clase para el dibujado de información sobre un vídeo con seguimiento de objetos.
    """
    default_properties = {
        TrackingVideoProperty.DRAW_OBJECTS: True,
        TrackingVideoProperty.TEXT_FRAME_INFORMATION: TextFormat(
            font=cv2.FONT_HERSHEY_SIMPLEX,
            color=(255, 255, 255),
            linetype=cv2.LINE_AA,
            thickness=2,
            font_scale=1.2
        ),
        TrackingVideoProperty.TEXT_OBJECT_INFORMATION: TextFormat(
            font=cv2.FONT_HERSHEY_SIMPLEX,
            color=(255, 255, 255),
            linetype=cv2.LINE_AA,
            thickness=2,
            font_scale=0.7
        ),
    }

    def __init__(self, tracked_objects: TrackedObjects, input_sequence: StreamSequence):
        """

        :param tracked_objects: estructura de datos con el seguimiento de los objetos.
        :param input_sequence: secuencia de vídeo.
        """
        self.tracked_objects = tracked_objects
        self.input_sequence = input_sequence
        self._properties: Dict[TrackingVideoProperty, Any] = self.default_properties.copy()
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
        # DRAW_OBJECTS
        if self.get_property(TrackingVideoProperty.DRAW_OBJECTS):
            frame = self._draw_objects(fid, frame)
        # DRAW_FRAME_NUMBER
        if self.get_property(TrackingVideoProperty.DRAW_FRAME_NUMBER):
            frame = self._draw_frame_number(fid, frame)
        # DRAW_FRAME_TIMESTAMP
        if self.get_property(TrackingVideoProperty.DRAW_FRAME_NUMBER):
            frame = self._draw_frame_timestamp(fid, frame)
        # Devolver frame con los dibujados.
        return frame

    def _draw_objects(self, fid: int, frame: Image) -> Image:
        """Realiza el dibujado en los objetos que aparecen en el frame actual.

        :param fid: número del frame.
        :param frame: frame.
        :return: frame con los objetos dibujados.
        """
        # Objetos detectados en el frame fid.
        tracked_objects_in_frame = self.tracked_objects.frame_objects(fid)
        # Dibujar la información de los objetos si aparecen en el frame actual.
        for tracked_object_detection in tracked_objects_in_frame:
            frame = self._draw_object(fid, frame, self.tracked_objects[tracked_object_detection.id],
                                      tracked_object_detection)
        return frame

    def _draw_object(self,
                     fid: int,
                     frame: Image,
                     tracked_object: TrackedObject,
                     tracked_object_detection: TrackedObjectDetection) -> Image:
        """Realiza el dibujado de un objeto en el frame actual.

        :param fid: número del frame.
        :param frame: frame.
        :param tracked_object: objeto con el seguimiento.
        :param tracked_object_detection: detección del objeto seguido en el frame actual.
        :return: frame con el objeto dibujados.
        """
        # DRAW_OBJECTS_IDS
        if self.get_property(TrackingVideoProperty.DRAW_OBJECTS_IDS):
            frame = self._draw_object_id(frame, tracked_object_detection)
        # DRAW_OBJECTS_BOUNDING_BOXES
        if self.get_property(TrackingVideoProperty.DRAW_OBJECTS_BOUNDING_BOXES):
            frame = self._draw_object_bounding_box(frame, tracked_object_detection)
        # DRAW_OBJECTS_TRACES
        if self.get_property(TrackingVideoProperty.DRAW_OBJECTS_TRACES):
            frame = self._draw_object_trace(fid, frame, tracked_object)
        # DRAW_OBJECTS_BOUNDING_BOXES_TRACES
        if self.get_property(TrackingVideoProperty.DRAW_OBJECTS_BOUNDING_BOXES_TRACES):
            frame = self._draw_object_bounding_box_trace(fid, frame, tracked_object)
        return frame

    def _draw_frame_number(self, fid: int, frame: Image) -> Image:
        """Escribe el número del frame en el que se encuentra en cada instante.

        :param fid: número del frame.
        :param frame: frame.
        :return: frame con el número de frame.
        """
        # Propiedades del texto.
        text_format = self.get_property(TrackingVideoProperty.TEXT_FRAME_INFORMATION)
        font, color, linetype, thickness, font_scale, _ = text_format
        # Dibujar texto.
        text = f'Frame {fid}'
        position = (30, self.input_sequence.properties().height - 30)
        cv2.putText(frame, text, position, font, font_scale, color, thickness, linetype)
        return frame

    def _draw_frame_timestamp(self, fid: int, frame: Image) -> Image:
        """Escribe el instante de tiempo en el que se encuentra la secuencia (en segundos)..

        :param fid: número del frame.
        :param frame: frame.
        :return: frame con el instante de tiempo en segundos.
        """
        # Propiedades del texto.
        text_format = self.get_property(TrackingVideoProperty.TEXT_FRAME_INFORMATION)
        font, color, linetype, thickness, font_scale, _ = text_format
        # Dibujar texto.
        second = fid / self.input_sequence.properties().fps
        text = f'Timestamp {second} s'
        position = (30, self.input_sequence.properties().height - 80)
        cv2.putText(frame, text, position, font, font_scale, color, thickness, linetype)
        return frame

    def _draw_object_id(self,
                        frame: Image,
                        tracked_object_detection: TrackedObjectDetection) -> Image:
        """Dibuja el id del objeto.

        :param frame: frame.
        :param tracked_object_detection: detección del objeto seguido.
        :return: frame con el id del objeto dibujado.
        """
        # Propiedades del texto.
        text_format = self.get_property(TrackingVideoProperty.TEXT_OBJECT_INFORMATION)
        font, color, linetype, thickness, font_scale, _ = text_format
        # Dibujar texto.
        text = f'OBJECT ID {tracked_object_detection.id}'
        position = tracked_object_detection.object.bounding_box.top_left
        position = position.x, position.y - 8
        cv2.putText(frame, text, position, font, font_scale, color, thickness, linetype)
        # Devolver frame.
        return frame

    def _draw_object_trace(self, fid: int, frame: Image, tracked_object: TrackedObject) -> Image:
        """Dibuja los trazados de un objeto hasta el frame en el que se encuentra.

        :param fid: número del frame.
        :param frame: frame.
        :param tracked_object: estructura del seguimiento del objeto.
        :return: imagen con el seguimiento del objeto.
        """
        positions_centroid = [t_obj.object.center for t_obj in tracked_object if t_obj.frame <= fid]
        # Dibujar centroides.
        prev_position = positions_centroid[0]
        # Dibujar las líneas.
        color = self._objects_colors[tracked_object.id]
        prev_position = positions_centroid[0]
        for position in positions_centroid:
            cv2.line(frame, position, prev_position, color, 2, cv2.LINE_AA)
            prev_position = position
        # Dibujar los puntos.
        color = (color[0] * 1/2, color[1] * 2/3, 3/4 * color[2])
        prev_position = positions_centroid[0]
        for position in positions_centroid:
            cv2.circle(frame, position, 0, color, 5, cv2.LINE_AA)
            prev_position = position
        return frame

    def _draw_object_bounding_box_trace(self,
                                        fid: int,
                                        frame: Image,
                                        tracked_object: TrackedObject) -> Image:
        """Dibuja los trazados del bounding box de un objeto..

        :param fid: número del frame.
        :param frame: frame.
        :param tracked_object: estructura del seguimiento del objeto.
        :return: imagen con el seguimiento del objeto.
        """
        bounding_boxes = [t_obj.object.bounding_box
                          for t_obj in tracked_object if t_obj.frame <= fid]
        # Dibujar bounding boxes.
        color = self._objects_colors[tracked_object.id]
        prev_bounding_box = bounding_boxes[0]
        for bounding_box in bounding_boxes:
            for position_id in range(len(bounding_box)):
                position = bounding_box[position_id]
                prev_position = prev_bounding_box[position_id]
                cv2.line(frame, position, prev_position, color, 1, cv2.LINE_AA)
            prev_bounding_box = bounding_box
        return frame

    def _draw_object_bounding_box(self,
                                  frame: Image,
                                  tracked_object_detection: TrackedObjectDetection) -> Image:
        """Dibuja la caja delimitadora de un objeto.

        :param frame: frame.
        :param tracked_object_detection: detección del objeto seguido.
        :return: frame con la bounding box añadida al objeto.
        """
        # Obtener la bounding box.
        bounding_box = tracked_object_detection.object.bounding_box
        # Dibujar sobre el frame.
        color = self._objects_colors[tracked_object_detection.id]
        cv2.line(frame, bounding_box.top_left, bounding_box.top_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.top_right, bounding_box.bottom_right, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_right, bounding_box.bottom_left, color, 2, cv2.LINE_AA)
        cv2.line(frame, bounding_box.bottom_left, bounding_box.top_left, color, 2, cv2.LINE_AA)
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

    def generate_video(self, file_output: str) -> None:
        """Genera la secuencia de vídeo y la guarda en un archivo.

        :param file_output: archivo de salida.
        :return: None.
        """
        output_stream = StreamSequenceWriter(file_output, self.input_sequence.properties())
        t = tqdm(total=len(self.input_sequence), desc='Generating video')
        for frame in self:
            output_stream.write(frame)
            t.update()
