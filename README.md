# WAVEMIND: Diagnóstico Automático de Trastornos Neurológicos con EEG

**WAVEMIND** es una aplicación de escritorio que permite cargar señales EEG, procesarlas, y diagnosticar posibles trastornos neurológicos como epilepsia, esquizofrenia, insomnio, o identificar actividad normal. Utiliza algoritmos de aprendizaje automático y genera un informe en PDF automáticamente.

---

## 🧠 Funcionalidades

- 📥 Carga de archivos EEG en formato `.csv`.
- 📊 Visualización de la señal EEG en un gráfico.
- 🧪 Procesamiento de datos mediante transformada wavelet.
- 🤖 Clasificación con modelo `RandomForestClassifier` de `scikit-learn`.
- 📈 Muestra la precisión del modelo.
- 🧾 Generación de reporte PDF con resultado y descripción del diagnóstico.
- 💾 El reporte se guarda automáticamente y se abre en el navegador (Chrome si es predeterminado).

---

## 🧪 Trastornos detectados

- **Normal**
- **Epilepsia**
- **Esquizofrenia**
- **Insomnio**

---

## 📂 Estructura esperada del archivo.

El archivo EEG debe estar en formato ".csv", ".edf" , ".mat", donde cada fila representa una muestra de señal.

---

## 🛠 Requisitos

- Python 3.7+
- Bibliotecas necesarias:
  ```bash
  pip install matplotlib numpy pandas scikit-learn pywt fpdf
