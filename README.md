

# Pasos

1. Obtener el objeto (centroide y bounding box).
    * Este paso se realiza con la librería simple-object-detection.
    
2. Calcular la distancia euclídea entre los nuevos objetos y los ya existentes.
    * Se necesitará identificar en el seguimiento cada objeto con un ID único,
      no vale con el asignado por el detector de objetos, puesto que en dos frames
      contiguos no tiene por qué asignarse el mismo id.
      
    * Ni siquiera tiene por qué esperarse que un objeto detectado anteriormente
      vuelva a presentarse, puede no haberse detectado o desaparecido.
      
    * Podemos partir de la ventaja de que la cámara será estática, y además como
      es seguimiento de vehículos, estos irán en línea recta, asegurando así una
      mayor fiabilidad de que dos emparejamientos son correctos (suponiendo una
      detección sin errores).
      
3. Actualizar las coordenadas (x, y) de los objetos existentes.
    * Esto se puede realizar cogiendo el centroide del último objeto detectado,
      no hace falta guardar información redundante.
      
4. Registrar nuevos objetos.
    * Asignar un nuevo identificador único.
    
    * Almacenar el objeto (concretamente su bounding box y su centroide, pero
      almacenando el objeto tenemos el resto de información también, que será
      útil, ya que podemos necesitar su clase, etc.
      
5. Eliminar objetos antiguos.
    * Aquellos objetos que han desaparecido de la escena.
    
    * Añadir cierta tolerancia, puesto que un objeto puede desaparecer por unos
      frames porque no se haya detectado pero luego vuelva a aparecer, y se pueda
      realizar el emparejamiento correctamente.
    

# Ideas

- Analizar un frame con distintos detectores.
    * Esto tiene un problema y es cómo hacer realizar el matching de objetos, 
      puesto que cada detector asignará un ID distinto.
