import numpy as np
from ..tracker import ObjectTracker
from ..utils import calculate_euclidean_distance
# DEBUG
import cv2
from simple_object_detection.utils import set_bounding_boxes_in_image


class CentroidLinearTracker(ObjectTracker):
    """Seguimiento de objetos a partir de su centroide, y realizando la
    suposición de que los objetos se mueven de manera lineal en la escena.

    DESARROLLO
    ---------------------------------
    - Para hacer matching tener en cuenta que el objeto debe seguir una trayectoria
    rectilínea, eso facilita el matching.
    - Hacer matching con N frames anteriores.
    """
    def __init__(self, distance_tolerance_factor=0.05, max_previous_frames_matching=1, *args, **kwargs):
        """

        :param distance_tolerance: factor de la distancia máxima para realizar emparejamientos.
        consecutivos.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.distance_tolerance_factor = distance_tolerance_factor

    def _register_initial_objects(self):
        # Obtener los objetos del frame 0.
        objects_initial = self.objects_in_frame(0)
        for obj in objects_initial:
            self.registered_objects.register_object(obj, 0)

    def _distance_tolerance(self):
        a = np.amax([self.width, self.height])
        return a * self.distance_tolerance_factor

    def _calculate_matches(self, objects_prev, objects_actual):
        matches_found = list()
        # Comprobar previamente que hay objetos al menos algún objeto en cada lista.
        if len(objects_prev) and len(objects_actual):
            # Inicializar distancias a 0.
            distances = np.zeros((len(objects_prev), len(objects_actual)))
            # Crear una tabla con las filas como objetos previos y columnas como objetos actuales.
            for i in range(len(objects_prev)):
                centroid_obj_i = objects_prev[i].get_centroid()
                for j in range(len(objects_actual)):
                    centroid_obj_j = objects_actual[j].get_centroid()
                    distances[i, j] = calculate_euclidean_distance(centroid_obj_i, centroid_obj_j)
            # Para cada objeto actual, intentar emparejar con el que menor distancia obtenga
            # de los objetos previos.
            for j in range(len(objects_actual)):
                # Distancias a la que están los objetos del objeto j.
                distances_with_j = distances[:, j]
                # Índice del objeto del frame previo con menor distancia al objeto j.
                best_i_match = np.argmin(distances_with_j)
                # Comprobar que la distancia sea menor que una tolerancia (px).
                if distances[best_i_match, j] < self._distance_tolerance():
                    # Agregar el emparejamiento del objeto (best_i_match, j).
                    matches_found.append((best_i_match, j))
        return matches_found

    def _resolve_matches(self, matches, objects_prev, objects_actual):
        """
        Actual criterio: el que antes se hizo.
        TODO: ¿Qué criterio seguir? ¿El que más puntuación tenga? ¿El que menor distancia?
         ¿El primero y ya?
        :param matches:
        :param objects_prev:
        :param objects_actual:
        :return:
        """
        resolved_matches = list()
        # Para evitar emparejamientos duplicados, llevar la lista de si ha sido emparejado ya.
        object_prev_matched = np.full(len(objects_prev), False)
        # Determinar para cada objeto actual cuál de los previos se le va a asignar.
        for match_i, match_j in matches:
            if not object_prev_matched[match_i]:
                resolved_matches.append((match_i, match_j))
                object_prev_matched[match_i] = True
        return resolved_matches

    def _do_matches(self, frame_actual, objects_actual):
        """

        TODO: Problema de que los frames sean muchos: puede que el objeto se encuentre a una
         distancia mayor que la tolerada, por tanto podría introducirse un factor que
         lo maneje: distancia < distance_tolerance + (num_frames_atrás * factor).
        :param frame_actual:
        :param objects_actual:
        :return: índices de los objetos actuales emparejados.
        """
        matches_done = list() # TODO: objetos actuales que han sido emparejados!!!
        # Obtener los objetos registrados.
        objects_with_uid_registered = self.registered_objects.objects_with_uid()
        # Comprobar que hay objetos registrados con los que emparejar.
        if objects_with_uid_registered:
            objects_uid_registered, objects_registered = zip(*objects_with_uid_registered)
            # Buscar los emparejamientos.
            matches = self._calculate_matches(objects_registered, objects_actual)
            # Determinar qué emparejamientos se van a hacer.
            resolved_matches = self._resolve_matches(matches, objects_registered, objects_actual)
            # Actualizar el registro de cada objeto emparejado.
            for match_i, match_j in resolved_matches:
                obj_uid = objects_uid_registered[match_i]
                self.registered_objects.update_object(objects_actual[match_j], obj_uid, frame_actual)
        return matches_done # TODO: objetos actuales que han sido emparejados!!!

    def _algorithm(self):
        # Paso 1. Registrar objetos iniciales.
        self._register_initial_objects()
        ##############
        # TODO: DEBUG PURPOSES ONLY!
        print(self.registered_objects)
        #self.show_cv(0)
        ##############
        # Paso 2. Emparejar, registrar, y desregistrar objetos en el resto de frames.
        for frame_proccessing in range(1, len(self.sequence)):
            objects_actual = self.objects_in_frame(frame_proccessing)
            # 1. Emparejar.
            objects_actual_matched = self._do_matches(frame_proccessing, objects_actual)
            # 2. Registrar: TODO

            # 3. Desregistrar.
            self.registered_objects.unregister_dissapeared_objects(frame_proccessing)
            ##############
            # TODO: DEBUG PURPOSES ONLY!
            print(self.registered_objects)
            self.show_cv(frame_proccessing)
            ##############

    def show_cv(self, frame_id):
        """TODO: BORRAR SOLO DEBUG
        """
        print("Mostrando del frame", frame_id)
        image = self.sequence[frame_id]
        # Objetos en ese frame.
        objects = self.objects_in_frame(frame_id)
        # Preparar la imagen.
        image = set_bounding_boxes_in_image(image, objects)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        # Pintar un punto en cada objeto registrado.
        for _, obj in self.registered_objects.objects_with_uid():
            image = cv2.circle(image, obj.get_centroid(), 1, (255, 0, 0), 5)
        new_shape = (int(image.shape[1] * 4 / 5), int(image.shape[0] * 4 / 5))
        # Redimensionar la salida.
        image = cv2.resize(image, new_shape)
        # Mostrar la imagen.
        cv2.imshow('Image', image)
        # Esperar a pulsar escape para cerrar la ventana.
        cv2.waitKey(0)

