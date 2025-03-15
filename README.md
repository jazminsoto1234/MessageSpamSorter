# MessageSpamSorter
Classify spam and non-spam emails through an api and using machine learning

## Clasificacion de variable discretas
Para su deteccion se usaran dos clases:
- Spam 
- No-Spam

Por ello se usara la regresion logisticas para predecir clases binarias.

Caracteristicas a tener en cuenta:
- words (vectorizar el texto)
- Se uso countvectorizer para trabajar con las frecuencias y matchear con las palabras dadas en el modelo de entranamiento.


Keywords:

Loss function: Es una funcion para evaluar que tan alejado esta los datos del algoritmo con el conjunto de datos de testing.

Desicion boundary: Es un trazo (linea, plano, etc) que separa en dos grupos.

Vamos a colocar como resultado predictivo a Y' y a los reales resultados como Y. 

## Mejoras a futuro

El modelo soporta a emails solo en ingles como por ejemplo: "To STOP receiving these emails from us Just hit *REPLY* and let us know Thanks." y puede haber un sesgo de error al no tener palabras nuevas para detectar como spam o no spam. Ademas que el score logrado por el modelo es de 96.

