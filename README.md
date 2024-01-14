# TonicNet


Código del proyecto de la asignatura *Aprendizaje Automático y Redes Neuronales* (AARN) del grado en Ingeniería Informática, de la UPV/EHU (curso 2023-24). Proyecto de final de asignatura. 
Desarrollado por David Cuenca Marcos, a partir del trabajo original de O. Peracha (TonicNet, https://github.com/omarperacha/TonicNet).

<b>Requisitos:</b>
- Python 3 (probado con la versión 3.11.7)
- Pytorch (probado con la versión 2.1.2)
- Music21 (probado con la versión 9.1.0)
- midi2audio (probado con la versión 0.1.1)
    - fluidsynth/jammy,now 2.2.5-1 amd64 (en Linux Mint 21.2 Victoria)

<b>Preparación del dataset:</b>

La preparación del dataset en train/validation/test splits se ejecuta mediante el comando:
```
python main.py --jsf only
```

<b>Para entrenar el modelo a partir desde cero, se puede usar el comando:</b>

Primero, es recomendable ejecutar el comando del punto anterior. De lo contrario, no será posible ejecutar el entrenamiento.
```
python main.py --train
```

Un entrenamiento de 60 epochs lleva alrededor de 3-6 horas en una sola GPU.

<b>Generación de música usando random sample: </b>

Se puede generar música usando random sample siguiendo el siguiente comando e indicando el número de muestras a generar.
Para cargar un modelo concreto, basta indicar el path a dicho modelo preentrenado después del flag ```--model```.

```
python main.py --sample N_SAMPLES
```
