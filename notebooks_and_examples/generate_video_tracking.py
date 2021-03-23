import sys
from simple_object_detection.utils import load_sequence, load_detections_in_sequence, save_sequence
from simple_object_tracking.trackers import CentroidTracker
from simple_object_tracking.utils import *


def generate_video_sequence(tracker, sequence, file_output):
    # Generar la secuencia con las trazas y guardarlas.
    sequence_with_traces = sequence_with_objects_trace(sequence, tracker)
    save_sequence(sequence_with_traces, tracker.frame_width, tracker.frame_height,
                  tracker.fps, file_output)


if __name__ == '__main__':
    width, height, fps, sequence = load_sequence(sys.argv[1])
    object_detections = load_detections_in_sequence(sys.argv[2])
    # Crear y ejecutar el seguimiento de objetos.
    tracker = CentroidTracker(
        sequence_with_information=(width, height, fps, sequence),
        object_detections=object_detections,
        distance_tolerance_factor=0.05,
        frames_to_unregister_object=10,
        objects_avoid_duplicated=True,
        objects_min_score=0.2,
        objects_classes=None
    )
    # Ejecutar el seguimiento.
    tracker.run()
    # Mostrar número original de frames del vídeo y número de frames eliminados
    print("Número de frames del video:", len(sequence))
    # Mostrar las posiciones del objeto.
    print("Objetos registrados (incluso desregistrados):", tracker.registered_objects.objects)
    # Obtener todas las posiciones por las que ha pasado un objeto dado su id.
    object_uid = 1
    positions = positions_in_object_tracking(tracker.registered_objects.history[object_uid],
                                             sequence)
    print("Lista de posiciones del objeto {}:".format(object_uid), positions)
    # Guardar el vídeo de salida.
    print("Generando video de salida...")
    generate_video_sequence(tracker, sequence, sys.argv[3])


