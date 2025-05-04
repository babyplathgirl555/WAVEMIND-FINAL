import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from fpdf import FPDF
import mne
import scipy.io

class EEGApp:
    def __init__(self, root):
        self.root = root
        self.root.title("WaveMind - Diagnóstico Automatizado EEG")
        self.root.state('zoomed')
        self.root.configure(bg="#eef2f7")

        self.data = None
        self.model = None

        self.mapa_clases = {
            0: "Normal",
            1: "Epilepsia",
            2: "Esquizofrenia",
            3: "Insomnio"
        }

        tk.Label(self.root, text="WaveMind - Sistema Clínico de Apoyo al Diagnóstico Neurológico",
                 bg="#264653", fg="white", font=("Helvetica", 18, "bold"), pady=12).pack(fill=tk.X)

        top_frame = tk.Frame(self.root, bg="#eef2f7")
        top_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        left_frame = tk.Frame(top_frame, bg="white", padx=15, pady=15)
        left_frame.pack(side=tk.LEFT, fill=tk.Y)

        right_frame = tk.Frame(top_frame, bg="#eef2f7")
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        boton_estilo = {"font": ("Arial", 10, "bold"), "bg": "#2a9d8f", "fg": "white", "padx": 10, "pady": 5}
        tk.Button(left_frame, text="Cargar Señal (.csv, .edf, .mat)", command=self.cargar_archivo, **boton_estilo).pack(pady=5)
        tk.Button(left_frame, text="Entrenar Modelo y Diagnosticar", command=self.entrenar_modelo, **boton_estilo).pack(pady=5)

        tk.Label(left_frame, text="Datos del Paciente", font=("Arial", 12, "bold"), bg="white").pack(pady=(10, 0))
        self.nombre_entry = self._crear_entrada(left_frame, "Nombre completo")
        self.edad_entry = self._crear_entrada(left_frame, "Edad")
        self.cedula_entry = self._crear_entrada(left_frame, "Cédula")

        tk.Button(left_frame, text="Generar Reporte PDF", command=self.generar_reporte, **boton_estilo).pack(pady=10)

        tk.Label(right_frame, text="Resultados del Modelo", font=("Arial", 12, "bold"), bg="#eef2f7").pack()
        self.resultados_text = scrolledtext.ScrolledText(right_frame, height=12, font=("Courier", 10))
        self.resultados_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        tk.Label(right_frame, text="Visualización de señal EEG (primer registro)", font=("Arial", 12), bg="#eef2f7").pack()
        self.canvas_frame = tk.Frame(right_frame, bg="#eef2f7")
        self.canvas_frame.pack(fill=tk.BOTH, expand=True)

        self.fig = plt.Figure(figsize=(8, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.canvas_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _crear_entrada(self, parent, label_text):
        tk.Label(parent, text=label_text + ":", bg="white").pack(anchor="w")
        entry = tk.Entry(parent, width=30)
        entry.pack(pady=2)
        return entry

    def cargar_archivo(self):
        file_path = filedialog.askopenfilename(filetypes=[("Todos los formatos", "*.csv *.edf *.mat")])
        if not file_path:
            return
        try:
            ext = os.path.splitext(file_path)[-1].lower()
            if ext == ".csv":
                self.data = pd.read_csv(file_path)
            elif ext == ".edf":
                raw = mne.io.read_raw_edf(file_path, preload=True)
                data, _ = raw[:, :]
                df = pd.DataFrame(data.T)
                df['label'] = 0
                self.data = df
            elif ext == ".mat":
                mat = scipy.io.loadmat(file_path)
                for key in mat:
                    if isinstance(mat[key], np.ndarray) and mat[key].ndim == 2:
                        df = pd.DataFrame(mat[key])
                        df['label'] = 0
                        self.data = df
                        break
            else:
                raise ValueError("Formato no soportado.")
            messagebox.showinfo("Éxito", "Archivo cargado correctamente.")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {str(e)}")

    def entrenar_modelo(self):
        if self.data is None:
            messagebox.showwarning("Advertencia", "Primero cargue un archivo.")
            return

        if 'label' not in self.data.columns:
            messagebox.showerror("Error", "El archivo debe contener una columna 'label'.")
            return

        X = self.data.drop(columns=['label'])
        y = self.data['label']

        if len(X) > 1000:
            indices_muestra = np.random.choice(X.index, 1000, replace=False)
            X = X.loc[indices_muestra]
            y = y.loc[indices_muestra]

        clases_unicas = y.nunique()
        self.resultados_text.delete(1.0, tk.END)

        if clases_unicas < 2:
            self.resultados_text.insert(tk.END, "Advertencia: Solo hay una clase en los datos.\n")
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            X_fake = X + np.random.normal(0, 0.5, X.shape)
            y_fake = [(y.iloc[0] + 1) % len(self.mapa_clases)] * len(X_fake)
            X_aug = pd.concat([pd.DataFrame(X), pd.DataFrame(X_fake)], ignore_index=True)
            y_aug = pd.Series(list(y) + y_fake)
            self.model.fit(X_aug, y_aug)
            y_pred = self.model.predict(X)
            diagnostico = y_pred[0]
            nombre_diagnostico = self.mapa_clases.get(diagnostico, "Desconocido")
            self.resultados_text.insert(tk.END, f"Diagnóstico sugerido: {nombre_diagnostico}\n")
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            reporte_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            df_reporte = pd.DataFrame(reporte_dict).T.round(2)
            diagnostico = self.model.predict(X.iloc[[0]])[0]
            nombre_diagnostico = self.mapa_clases.get(diagnostico, "Desconocido")
            self.resultados_text.insert(tk.END, f"Modelo entrenado exitosamente.\n")
            self.resultados_text.insert(tk.END, f"Diagnóstico sugerido para el primer registro: {nombre_diagnostico}\n\n")
            self.resultados_text.insert(tk.END, df_reporte.to_string())

        self.visualizar_senal()

    def visualizar_senal(self):
        if self.data is None:
            return
        self.ax.clear()
        muestra = self.data.iloc[0, :-1].values
        self.ax.plot(muestra, color="#264653")
        self.ax.set_xlabel("Muestras")
        self.ax.set_ylabel("Amplitud")
        self.ax.set_title("Primer registro EEG")
        self.canvas.draw()

    def generar_reporte(self):
        if self.model is None:
            messagebox.showwarning("Advertencia", "Primero entrene el modelo.")
            return

        nombre = self.nombre_entry.get().strip()
        edad = self.edad_entry.get().strip()
        cedula = self.cedula_entry.get().strip()

        if not (nombre and edad and cedula):
            messagebox.showwarning("Campos faltantes", "Complete todos los campos del paciente.")
            return

        try:
            X = self.data.drop(columns=['label'])
            pred = self.model.predict([X.iloc[0].values])[0]
            enfermedad = self.mapa_clases.get(pred, "Desconocido")
        except Exception as e:
            messagebox.showerror("Error", f"Error durante la predicción: {str(e)}")
            return

        descripcion = {
            "Epilepsia": "Se han identificado patrones compatibles con actividad epiléptica.",
            "Esquizofrenia": "La señal EEG sugiere actividad relacionada con trastornos del movimiento.",
            "Insomnio": "Señal EEG indica alteraciones propias del insomnio.",
            "Normal": "Actividad cerebral dentro de los parámetros normales.",
            "Desconocido": "No se pudo determinar el diagnóstico con precisión."
        }

        try:
            pdf = FPDF()
            pdf.add_page()
            pdf.set_font("Arial", size=12)
            pdf.set_text_color(0)

            pdf.cell(0, 10, "WaveMind - Reporte Clínico EEG", ln=True, align='C')
            pdf.ln(10)
            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(0, 10, "Datos del paciente:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, f"Nombre: {nombre}", ln=True)
            pdf.cell(0, 10, f"Edad: {edad}", ln=True)
            pdf.cell(0, 10, f"Cédula: {cedula}", ln=True)
            pdf.ln(10)

            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(0, 10, "Diagnóstico sugerido:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(0, 10, enfermedad, ln=True)
            pdf.ln(5)
            pdf.multi_cell(0, 10, descripcion.get(enfermedad, "Sin descripción disponible."))
            pdf.ln(10)

            pdf.set_font("Arial", style='B', size=12)
            pdf.cell(0, 10, "Resumen técnico:", ln=True)
            pdf.set_font("Arial", size=10)
            for linea in self.resultados_text.get("1.0", tk.END).strip().split("\n"):
                pdf.multi_cell(0, 10, linea)

            nombre_archivo = f"Reporte_{nombre.replace(' ', '_')}_{cedula}.pdf"
            ruta_guardado = os.path.join("C:/Users/mardi/OneDrive/Documentos", nombre_archivo)
            pdf.output(ruta_guardado)
            os.startfile(ruta_guardado, "open")
            messagebox.showinfo("PDF generado", f"Reporte guardado como {nombre_archivo}")
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo generar el PDF: {str(e)}")

if __name__ == "__main__":
    root = tk.Tk()
    app = EEGApp(root)
    root.mainloop()
