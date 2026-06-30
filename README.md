# ESCUELA POLITÉCNICA NACIONAL

## Laboratorio 9 - Evaluación de Riesgo Crediticio con Árboles de Decisión

**Curso:** Business Intelligence  
**Grupo:** C  
**Fecha:** 29/06/2026  
**Integrantes:**
- Nayeli Leiva
- Ricardo Villarreal
- Donoban Ramon
- Daniel Jaramillo
- Leonardo Maldonado

---
# Predicción de valores en Weka

---

## Introducción

La minería de datos permite extraer conocimiento útil a partir de conjuntos de
datos mediante modelos capaces de reconocer patrones y, con ellos, predecir
valores desconocidos. Weka es una de las herramientas más utilizadas en este
campo, ya que integra un amplio conjunto de algoritmos de clasificación,
agrupamiento y asociación en un entorno gráfico sencillo.

En esta práctica se trabaja con el conjunto de datos `weather.nominal.arff`, un
ejemplo clásico que relaciona condiciones meteorológicas (estado del cielo,
temperatura, humedad y viento) con la decisión de jugar o no. El propósito es
construir una instancia de prueba con el valor de la clase desconocido y emplear
clasificadores para que el sistema prediga dicho valor de forma automática.

---

## Objetivo

Crear una instancia de prueba en Weka, dejando en blanco el atributo de clase, y
utilizar los clasificadores **NaiveBayes** y **J48** sobre el conjunto
`weather.nominal.arff` para predecir el valor desconocido, comparando los
resultados obtenidos por ambos algoritmos.

---

## Desarrollo

A continuación se describe el procedimiento realizado paso a paso, acompañado de
las capturas correspondientes.

### Preparación de la instancia de prueba

**1.** En el menú principal de Weka se hace clic en **Tools** y se selecciona
**ArffViewer**.

**2.** Se abre el archivo de datos con **File → Open**, seleccionando
`weather.nominal.arff`.

**3.** Se seleccionan todos los registros **excepto uno**, que se reserva como
registro de prueba, y se eliminan con **Edit → Delete Instances**.

**4.** Se modifican los valores del registro restante, asignándole los valores
que se desean utilizar como instancia de prueba mediante las herramientas de
edición del ArffViewer.

**5.** Se deja **en blanco** el atributo de clase, ya que ese es el valor que se
desea que prediga el clasificador.

**6.** Se guarda el archivo con el nombre **`test.arff`**.

<img width="986" height="287" alt="Captura de pantalla 2026-06-29 234632" src="https://github.com/user-attachments/assets/0890e414-757c-4d04-b48a-4ba1b16ddd7a" />

### Construcción y aplicación del clasificador (NaiveBayes)

**7.** En la pestaña **Preprocess** se hace clic en **Open file** y se carga el
conjunto de entrenamiento `weather.nominal.arff`.

**8.** En la pestaña **Classify** se elige el clasificador **NaiveBayes**.

**9.** Se selecciona la opción **Supplied test set**, se hace clic en **Set** y,
con **Open File**, se carga el archivo `test.arff`.

**10.** En **More Options** se selecciona **PlainText** en **Output
predictions**, para visualizar la predicción en texto plano.

**11.** Se hace clic en **Start** para construir el clasificador y aplicarlo
sobre la instancia de prueba.

<img width="1365" height="765" alt="Captura de pantalla 2026-06-29 235421" src="https://github.com/user-attachments/assets/c91ce780-86c6-4eea-9f46-e916270cac8f" />

### Aplicación del clasificador J48

**12.** Se repite el procedimiento seleccionando ahora el clasificador **J48**,
manteniendo el mismo conjunto de prueba (`test.arff`).

<img width="1213" height="752" alt="Captura de pantalla 2026-06-29 235649" src="https://github.com/user-attachments/assets/023e8231-e1d6-4ba3-abfb-7414d2db0fbe" />

---

## Análisis de resultados

Tras aplicar el clasificador **NaiveBayes** sobre la instancia de prueba, el
modelo predijo el valor de la clase como **Play: Yes**. Este algoritmo se basa en
el teorema de Bayes y calcula la probabilidad de cada clase a partir de los
valores de los atributos, asumiendo independencia entre ellos; la clase con
mayor probabilidad es la que se asigna como predicción.

Al repetir el experimento con el clasificador **J48**, basado en árboles de
decisión, se obtuvo el mismo resultado, **Play: Yes**. Esto indica que, para la
instancia de prueba utilizada, ambos modelos coinciden en su predicción a pesar
de emplear enfoques distintos: uno probabilístico (NaiveBayes) y otro basado en
reglas derivadas de un árbol (J48).

La coincidencia entre ambos clasificadores aporta mayor confianza en la
predicción obtenida, ya que dos algoritmos con fundamentos diferentes llegan a la
misma conclusión a partir de los mismos datos de entrenamiento.

---

## Implementación en Python

Como resultado final, el árbol de decisión **J48** obtenido en Weka se tradujo a
código Python mediante una estructura de condicionales `if/else`, junto con una
interfaz interactiva construida con `ipywidgets` que permite ingresar los valores
de los atributos y obtener la predicción de forma automática.

El árbol J48 generado para el dataset `weather.nominal` es el siguiente:

<img width="789" height="539" alt="image" src="https://github.com/user-attachments/assets/f52c1d6f-c53c-46d7-96f2-f100ac036e10" />


A partir de dicho árbol se desarrolló el siguiente código:

```python
from IPython.display import display
import ipywidgets as widgets


# Función que representa el árbol de decisión J48 generado en Weka
# para el dataset weather.nominal
def evaluar_juego(outlook, humidity, windy):

    if outlook == "sunny":
        if humidity == "high":
            return "No"
        else:  # humidity == "normal"
            return "Yes"

    elif outlook == "overcast":
        return "Yes"

    else:  # outlook == "rainy"
        if windy == "TRUE":
            return "No"
        else:  # windy == "FALSE"
            return "Yes"


# Widget Outlook (estado del cielo)
outlook_input = widgets.Dropdown(
    options=["sunny", "overcast", "rainy"],
    value="sunny",
    description="Outlook:"
)

# Widget Humidity (humedad)
humidity_input = widgets.Dropdown(
    options=["high", "normal"],
    value="high",
    description="Humidity:"
)

# Widget Windy (viento)
windy_input = widgets.Dropdown(
    options=["TRUE", "FALSE"],
    value="FALSE",
    description="Windy:"
)

# Botón
btn = widgets.Button(description="Evaluar juego")

# Salida
output = widgets.Output()


def on_button_clicked(b):
    with output:
        output.clear_output()

        resultado = evaluar_juego(
            outlook_input.value,
            humidity_input.value,
            windy_input.value
        )

        print(f"Play: {resultado}")


btn.on_click(on_button_clicked)

display(
    outlook_input,
    humidity_input,
    windy_input,
    btn,
    output
)
```

### Caso de prueba

Al ejecutar el código se despliega la interfaz con los menús desplegables; al
seleccionar los valores y presionar el botón **Evaluar juego**, el programa
muestra la predicción correspondiente.

<img width="491" height="189" alt="image" src="https://github.com/user-attachments/assets/be66017b-8e1b-4160-9e8b-7846252fc5a4" />

---

## Conclusión

Mediante el ArffViewer de Weka se construyó una instancia de prueba con el
atributo de clase en blanco y, utilizando los clasificadores **NaiveBayes** y
**J48** sobre el conjunto `weather.nominal.arff`, se logró predecir el valor de
la clase desconocida, obteniendo en ambos casos el resultado **Play: Yes**.

La práctica permitió comprender el flujo completo de un proceso de clasificación
en Weka: la preparación de los datos, la separación de una instancia de prueba,
la configuración del conjunto de evaluación y la interpretación de la predicción.
Asimismo, se evidenció que distintos algoritmos pueden coincidir en sus
resultados, lo que refuerza la utilidad de los modelos de clasificación para
predecir valores en instancias nuevas y desconocidas.
