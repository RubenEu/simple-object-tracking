from simple_object_tracking.tracker import ObjectTracker


class TrackingVideo:
    """Clase para el dibujado de información sobre un vídeo con seguimiento de objetos.

    TODO:
        ¿Cómo plantearla? ¿Clase para dibujar frame a frame? ¿ esta dibuja, guarda y todo?
    """
    def __init__(self, tracker: ObjectTracker):
        self.tracker = tracker

    def draw_bounding_boxes(self):
        ...

    def add_object_information(self):
        ...

    def add_frame_information(self):
        ...
