from simple_object_detection import DetectionModel
from simple_object_detection.models import YOLOv3
from simple_object_detection.utils import load_sequence, load_detections_in_sequence, save_sequence
from simple_object_tracking.trackers import CentroidLinearTracker
from simple_object_tracking.utils import *

# Cargar archivo de vídeo como secuencia de imágenes.
height, width, fps, sequence = load_sequence('../sample_data/ball-8.mp4')

# Cargar el modelo
file_detection = 'balls-8-detections-yolov3.pkl'
# file_detection = 'balls-8-detections-fastercrnn.pkl'
detections = load_detections_in_sequence('../sample_data/detections/' + file_detection)
# Eliminar las N primeras y M últimos.
N = 0
M = 1
sequence = sequence[N:-M]
detections = detections[N:-M]
# Cargar el modelo de seguimiento.
tracker = CentroidLinearTracker(
    sequence_with_information=(height, width, fps, sequence),
    object_detections=detections,
    distance_tolerance_factor=0.05,
    frames_to_unregister_object=10,
    objects_avoid_duplicated=True,    # Parámetro muy importante!!!!
    objects_min_score=0.2,
    objects_classes=None
)

tracker.run()

# Mostrar número original de frames del vídeo y número de frames eliminados
print("Número de frames del video:", len(sequence), "Frames eliminados:", N+M)
# Mostrar las posiciones del objeto.
print("Objetos registrados (incluso eliminados):", tracker.registered_objects.objects)
# Obtener todas las posiciones por las que ha pasado un objeto.
object_uid = 1
positions = positions_in_object_tracking(tracker.registered_objects.history[object_uid], sequence)
print("Lista de posiciones del objeto {}:".format(object_uid), positions)

# Generar la secuencia con las trazas y guardarlas.
sequence_with_traces = sequence_with_objects_trace(sequence, tracker)
save_sequence(sequence_with_traces, tracker.frame_width, tracker.frame_height,
              tracker.frames_per_second, '../../output.mp4')
