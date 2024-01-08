# TonicNet


Código del proyecto de la asignatura *Aprendizaje Automático y Redes Neuronales* (AARN) del grado en Ingeniería Informática, de la UPV/EHU (curso 2023-24). Proyecto de final de asignatura. 
Desarrollado por ... (insertar autores).

El proyecto original está basado en el código de ..., que a su vez es el código del artículo ... de ....

<b>Requisitos:</b>
- Python 3 (probado con la versión 3.11.7)
- Pytorch (probado con la versión 2.1.2)
- Music21 (probado con la versión 9.1.0)
- midi2audio (probado con la versión 0.1.1)
    - fluidsynth/jammy,now 2.2.5-1 amd64 (en Linux Mint 21.2 Victoria)

<b>Preparación del dataset:</b>

To prepare the vanilla JSB Chorales dataset with canonical train/validation/test split:
```
python main.py --gen_dataset
```

To prepare dataset augmented with [JS Fake Chorales](https://github.com/omarperacha/js-fakes):
```
python main.py --gen_dataset --jsf
```

To prepare dataset for training on JS Fake Chorales only:
```
python main.py --gen_dataset --jsf_only
```

<b>Train Model from Scratch:</b>

First run `--gen_dataset` with any optional 2nd argument, then:
```
python main.py --train
```

Training requires 60 epochs, taking roughly 3-6 hours on GPU

<b>Evaluate Pre-trained Model on Test Set:</b>

```
python main.py --eval_nn
```

<b>Sample with Pre-trained Model (via random sampling):</b>

```
python main.py --sample
```
