from typing import List, Tuple, Callable

from simple_object_detection.object import Object
from simple_object_detection.typing import Image

# Object datastructre typing.
ObjectUID = int
FrameID = int
Timestamp = int
RegisteredStatus = bool
ObjectWithUID = Tuple[ObjectUID, Object]
ObjectFrame = Tuple[FrameID, Object]
ObjectTimestamp = Tuple[Timestamp, Object]
ObjectHistory = List[ObjectFrame]
ObjectTracking = Tuple[ObjectUID, RegisteredStatus, ObjectHistory]

# Tracking typing.
Width = int
Height = int
FPS = float
Sequence = List[Image]
Timestamps = List[Timestamp]
# Información de una secuencia (ancho, alto, fps, frames).
SequenceInformation = Tuple[Width, Height, FPS, Sequence, Timestamps]
# Lista de objetos por cada frame de una secuencia.
SequenceObjectsDetections = List[List[Object]]
# Función que recibe una lista de objetos y devuelve una lista de objetos.
ObjectsFilterFunction = Callable[[List[Object]], List[Object]]
# Función para el cálculo de la distancia máxima a la que puede estar un objeto para ser emparejado.
# TODO: Añadirle la cantidad de frames elapsed.
DistanceToleranceFunction = Callable[[Width, Height, FPS], float]
