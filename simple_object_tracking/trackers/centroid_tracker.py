import numpy as np
from ..tracker import ObjectTracker
from ..utils import calculate_euclidean_distance


class CentroidTracker(ObjectTracker):
    """Modelo de seguimiento de objetos basados en su centro.

    Algoritmo de seguimiento:
    1. Registra todos los objetos detectados en el primer frame.
    2. Para cada frame en la secuencia:
        2.0. Obtener los objetos del frame actual.
        2.1. Buscar los emparejamientos posibles, resolver los conflictos y emparejar, actualizando
             la lista que mantiene el seguimiento de los objetos.
        2.2. Registrar aquellos objetos detectados en el frame actual que no han sido emparejados.
        2.3. Desregistrar los objetos que llevan sin detectarse un número de frames, determinado por
        el parámetro frames_to_unregister_object.

    Principales problemas:
    - Si se detecta un objeto de manera múltiple, se genera el trazado bien, pero supone que so
    distintos objetos los que está viendo, no uno mismo.
    """
    def __init__(self, distance_tolerance_factor=0.05, *args, **kwargs):
        """Crea una instancia de este modelo de seguimiento.

        :param distance_tolerance: factor en el rango [0, 1] que indica la distancia máxima a la que
         se podrá realizar el matching de objetos en proporción a la altura y anchura de los frames.
        :param args:
        :param kwargs:
        """
        super().__init__(*args, **kwargs)
        self.distance_tolerance_factor = distance_tolerance_factor

    def _register_initial_objects(self):
        """Registra los objetos iniciales.
        """
        # Obtener los objetos del frame 0.
        for obj in self.objects_in_frame(0):
            self.registered_objects.register_object(obj, 0)

    def _distance_tolerance(self):
        """Método que calcula la distancia máxima (en píxeles) con la que se considerará que un
        objeto puede ser emparejado con otro visto en otro frame.

        Para el cálculo se multiplica el factor 'distance_tolerance_factor' por el máximo entre la
        altura y anchura.
        """
        a = np.amax([self.frame_width, self.frame_height])
        return a * self.distance_tolerance_factor

    def _calculate_matches(self, objects_prev, objects_actual):
        """Busca todos los emparejamientos posibles entre los objetos registrados (objects_prev) y
        los objetos obtenidos en un frame (objects_actual).

        Para ello se crea una matriz de len(objects_prev) x len(objects_actual) en la que se calcula
        todas las distancias entre los objectos de objects_prev y objects_actual.

        Una vez calculadas las distancias, cada uno de los objetos del frame actual se intenta
        emparejar con alguno de los objetos de los registrados. Es decir, un objeto registrado puede
        tener varios candidatos de los objetos del frame actual, pero cada uno de los del frame
        actual solo tendrá un candidato de los registrados.

        :param objects_prev: lista de objetos registrados.
        :param objects_actual: numpy array de objetos detectados en el frame actual.
        :return: lista de tuplas entre posibles emparejamientos.
        """
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

    def _resolve_matches(self, matches, objects_prev):
        """Un objeto previo puede tener como candidatos de emparejamiento varios objetos actuales,
        para ello debe resolverse el emparejamiento.

        Actualmente la implementación es: el primer objeto que devuelva la implementación de
        búsqueda de matches.

        :param matches: lista de emparejamientos [(obj_prev_i, obj_act_j), ...].
        :param objects_prev: lista de objetos previos.
        :return: lista de los matches resueltos.
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
        """Una vez se han resuelto los conflictos entre emparejamientos con un mismo objeto, se
        deben registrar y actualizar en la lista self.registered_objects.

        :param frame_actual: id del frame actual.
        :param objects_actual: lista de objetos del frame actual.
        :return: índices de los objetos del frame actual emparejados.
        """
        # Lista de índices de los objetos actuales que han sido emparejados.
        matches_done = list()
        # Obtener los objetos registrados.
        objects_with_uid_registered = self.registered_objects.objects_with_uid()
        # Comprobar que hay objetos registrados con los que emparejar.
        if objects_with_uid_registered:
            objects_uid_registered, objects_registered = zip(*objects_with_uid_registered)
            # Buscar los emparejamientos.
            matches = self._calculate_matches(objects_registered, objects_actual)
            # Determinar qué emparejamientos se van a hacer.
            resolved_matches = self._resolve_matches(matches, objects_registered)
            # Actualizar el registro de cada objeto emparejado.
            for match_i, match_j in resolved_matches:
                obj_uid = objects_uid_registered[match_i]
                self.registered_objects.update_object(objects_actual[match_j], obj_uid, frame_actual)
                matches_done.append(match_j)
        return matches_done

    def _register_not_matched_objects(self, frame_actual, objects_actual, matched_objects_ids):
        """Registra todos los objetos que no han sido emparejados.

        :param frame_actual: id del frame con el que se está trabajando.
        :param objects_actual: ndarray con los objetos del frame actual.
        :param matched_objects_ids: lista con los ids de los objetos actuales registrados.
        """
        # Registrar los objetos no emparejados.
        for obj_id, obj in enumerate(objects_actual):
            # Comprobar que ese objeto no ha sido emparejado.
            if obj_id not in matched_objects_ids:
                self.registered_objects.register_object(obj, frame_actual)

    def _algorithm(self):
        # Paso 1. Registrar objetos iniciales.
        self._register_initial_objects()
        # Paso 2. Emparejar, registrar, y desregistrar objetos en el resto de frames.
        for frame_proccessing in range(1, len(self.sequence)):
            objects_actual = self.objects_in_frame(frame_proccessing)
            # 1. Emparejar.
            objects_matched = self._do_matches(frame_proccessing, objects_actual)
            # 2. Registrar.
            self._register_not_matched_objects(frame_proccessing, objects_actual, objects_matched)
            # 3. Desregistrar.
            self.registered_objects.unregister_dissapeared_objects(frame_proccessing)
