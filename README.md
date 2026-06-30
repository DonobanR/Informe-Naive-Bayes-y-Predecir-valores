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

## 1. Introducción

La evaluación crediticia es una de las tareas más críticas en el sector financiero. Determinar si un cliente representa un riesgo al momento de otorgarle un préstamo implica analizar múltiples factores de forma simultánea, lo cual puede ser difícil de escalar manualmente.

Los árboles de decisión son modelos de aprendizaje supervisado que permiten construir reglas de clasificación a partir de datos históricos. Su estructura jerárquica y legible los convierte en una herramienta ideal para este tipo de escenarios, ya que la lógica de clasificación puede auditarse, explicarse y trasladarse directamente a código.

En este laboratorio se parte de un dataset histórico de decisiones crediticias (`dataset.arff`), se entrena un árbol de decisión en Weka usando el algoritmo J48 (implementación de C4.5), y finalmente se traduce la lógica del árbol a una aplicación interactiva en Python.

---

## 2. Objetivo

Construir un modelo de árbol de decisión capaz de clasificar solicitudes de crédito como **seguras (safe)** o **riesgosas (risky)**, partiendo del dataset `dataset.arff`, y representar dicho modelo en código Python funcional con interfaz interactiva para el ingreso de datos.

---

## 3. Desarrollo

### 3.1 Dataset — `dataset.arff`

El archivo `dataset.arff` define la relación `loan_risk` con **1000 instancias** y los siguientes atributos categóricos:

| Atributo       | Valores posibles               | Rol        |
|----------------|-------------------------------|------------|
| `Age`          | young, middle-aged, senior    | Predictor  |
| `Income`       | low, medium, high             | Predictor  |
| `Loan_History` | poor, average, good           | Predictor  |
| `Loan_Decision`| risky, safe                   | Clase      |

El formato ARFF es el estándar de Weka para describir conjuntos de datos. La sección `@RELATION` nombra el dataset, `@ATTRIBUTE` define cada columna con sus posibles valores, y `@DATA` contiene las instancias:

```
@RELATION loan_risk

@ATTRIBUTE Age          {middle-aged,senior,young}
@ATTRIBUTE Income       {high,low,medium}
@ATTRIBUTE Loan_History {average,good,poor}
@ATTRIBUTE Loan_Decision {risky,safe}

@DATA
young,low,poor,risky
middle-aged,low,average,risky
young,medium,poor,risky
...
```

Algunos patrones observables en el dataset:
- Clientes `young` con ingresos `low` e historial `poor` son consistentemente clasificados como `risky`.
- Clientes `middle-aged` o `senior` con ingresos `high` tienden a ser `safe`.
- El historial crediticio resulta ser el factor más discriminante.

### 3.2 Árbol de Decisión generado en Weka (J48)

Al cargar el dataset en Weka y aplicar el clasificador **J48**, se genera el siguiente árbol:

<img width="1250" height="676" alt="image" src="https://github.com/user-attachments/assets/28e1bcdc-55af-4327-8b04-5e902539fd4b" />


**Interpretación del árbol:**

El nodo raíz es `Loan_History`, lo que indica que esta variable tiene la mayor ganancia de información sobre la decisión crediticia:

- Un historial **poor** lleva directamente a `risky` sin necesidad de evaluar otros atributos.
- Un historial **good** lleva directamente a `safe`.
- Un historial **average** requiere evaluar el nivel de `Income`:
  - Si el ingreso es **low**, el resultado es `risky`.
  - Si el ingreso es **high** o **medium**, se evalúa la `Age`: los clientes `young` resultan `risky`, mientras que `middle-aged` y `senior` resultan `safe`.

### 3.3 Implementación en Python

La lógica del árbol se tradujo a una función Python y se integró con `ipywidgets` para generar una interfaz interactiva en Jupyter Notebook.

#### Función de clasificación

```python
def evaluar_riesgo(historial, ingreso, edad):

    if historial == "poor":
        return "Risky"

    else:
        if historial == "good":
            return "Safe"

        else:  # historial == "average"

            if ingreso == "low":
                return "Risky"

            else:
                if ingreso == "high":
                    if edad == "young":
                        return "Risky"
                    else:  # middle-aged o senior
                        return "Safe"

                else:  # ingreso == "medium"
                    if edad == "young":
                        return "Risky"
                    else:  # middle-aged o senior
                        return "Safe"
```

La función replica fielmente la estructura del árbol J48: primero ramifica por historial, luego por ingreso, y finalmente por edad cuando corresponde.

#### Interfaz interactiva con ipywidgets

```python
from IPython.display import display
import ipywidgets as widgets

# Dropdowns de entrada
historial_input = widgets.Dropdown(
    options=["poor", "average", "good"],
    value="average",
    description="Historial:"
)

ingreso_input = widgets.Dropdown(
    options=["low", "medium", "high"],
    value="medium",
    description="Ingreso:"
)

edad_input = widgets.Dropdown(
    options=["young", "middleaged", "senior"],
    value="young",
    description="Edad:"
)

btn    = widgets.Button(description="Evaluar riesgo")
output = widgets.Output()

def on_button_clicked(b):
    with output:
        output.clear_output()
        resultado = evaluar_riesgo(
            historial_input.value,
            ingreso_input.value,
            edad_input.value
        )
        print(f"Resultado: {resultado}")

btn.on_click(on_button_clicked)

display(historial_input, ingreso_input, edad_input, btn, output)
```

La interfaz despliega tres menús desplegables (uno por atributo predictor) y un botón que invoca la función de clasificación al hacer clic, mostrando el resultado en el widget de salida.

### 3.4 Casos de Prueba

Se ejecutaron dos casos representativos para verificar el comportamiento del modelo:

<img width="800" height="707" alt="image" src="https://github.com/user-attachments/assets/2596a2a3-7099-4a9d-8ef9-d2e3a72d1a21" />


**Caso 1 — trayectoria en el árbol:**  
`Loan_History = good` → nodo terminal → **Safe**

**Caso 2 — trayectoria en el árbol:**  
`Loan_History = average` → `Income = medium` → `Age = young` → **Risky**

Ambos resultados coinciden con la lógica del árbol generado por Weka.

---

## 4. Análisis

### Variable más relevante: Historial Crediticio

El árbol J48 seleccionó `Loan_History` como nodo raíz porque es el atributo con mayor ganancia de información. Esto tiene sentido contextualmente: el comportamiento previo de pago de un cliente es el predictor más directo de su riesgo futuro. Los valores `poor` y `good` resuelven el 67.5% de las instancias (675 de 1000) sin necesidad de atributos adicionales.

### Rol secundario del Ingreso y la Edad

Cuando el historial es `average`, el modelo recurre al nivel de ingreso como segundo discriminante. Un ingreso `low` es suficiente para clasificar como `risky` (114 instancias), mientras que ingresos `high` o `medium` requieren también la edad del cliente. La variable `Age` actúa como desempate final: los clientes `young` con historial promedio y buenos ingresos siguen siendo considerados riesgosos, posiblemente por menor estabilidad financiera.

### Legibilidad del modelo

Una ventaja clave de los árboles de decisión frente a modelos como redes neuronales o SVM es su interpretabilidad directa. La función `evaluar_riesgo` es una traducción línea a línea del árbol; cualquier analista puede seguir el razonamiento del modelo sin conocimientos técnicos avanzados.

### Limitaciones observadas

- El árbol no captura interacciones más complejas (e.g., un cliente `young` con historial `good` e ingreso `low` podría tener un perfil distinto del árbol actual).
- Al tratarse de variables todas categóricas y discretas, el modelo no generaliza a rangos numéricos reales (edad en años, ingreso en dólares).
- El dataset de 1000 instancias es suficiente para una práctica, pero pequeño para un sistema crediticio real.

---

## 5. Conclusión

- El laboratorio demostró el flujo completo de aplicación de un árbol de decisión para clasificación binaria en un contexto financiero: desde la carga del dataset en formato ARFF hasta la construcción visual del árbol en Weka y su posterior implementación como función Python con interfaz interactiva.

- El modelo J48 identificó correctamente que el historial crediticio es el factor más determinante para la evaluación de riesgo, seguido del nivel de ingreso y, en casos ambiguos, de la edad del solicitante. Los dos casos de prueba ejecutados confirmaron que la implementación en Python replica fielmente la lógica del árbol generado por Weka.

- Este tipo de modelos representa una herramienta valiosa para automatizar decisiones crediticias de forma auditable y explicable, lo que es especialmente importante en contextos regulados donde se debe justificar el rechazo o aprobación de un préstamo.
