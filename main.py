
from pathlib import Path
from datetime import datetime
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk

# ==================================================
# CONFIGURACION
# ==================================================

BASE_DIR = Path(__file__).resolve().parent
MODELO_PATH = BASE_DIR / "keras_model.h5"
ETIQUETAS_PATH = BASE_DIR / "labels.txt"

CAMARA_INDICE = 1
TAMANO_IMAGEN = (224, 224)
ANCHO_VIDEO = 720
ALTO_VIDEO = 480

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

FONDO = "#111714"
PANEL = "#1A2420"
PANEL_2 = "#222F29"
VERDE = "#35D07F"
VERDE_OSCURO = "#173F2B"
BLANCO = "#F1F5F2"
GRIS = "#9BAAA1"
ROJO = "#FF6262"
AMARILLO = "#F5C451"


# ==================================================
# MODELO Y ETIQUETAS
# ==================================================

def cargar_modelo():
    if not MODELO_PATH.exists():
        raise FileNotFoundError(f"No se encontro: {MODELO_PATH}")

    if not ETIQUETAS_PATH.exists():
        raise FileNotFoundError(f"No se encontro: {ETIQUETAS_PATH}")

    with open(ETIQUETAS_PATH, "r", encoding="utf-8") as archivo:
        clases = [linea.strip() for linea in archivo if linea.strip()]

    modelo = tf.keras.models.load_model(
        MODELO_PATH,
        compile=False
    )

    print("Modelo cargado correctamente.")
    print("Clases:", clases)

    return modelo, clases


def abrir_camara():
    # Intentar identificar Camo por su nombre en Windows.
    try:
        from pygrabber.dshow_graph import FilterGraph

        dispositivos = FilterGraph().get_input_devices()

        print("Camaras detectadas:")
        for i, nombre in enumerate(dispositivos):
            print(f"  [{i}] {nombre}")

        for i, nombre in enumerate(dispositivos):
            if "camo" in nombre.lower():
                camara = cv2.VideoCapture(i, cv2.CAP_DSHOW)

                if camara.isOpened():
                    print(f"Camo abierta en el indice {i}.")
                    return camara

                camara.release()

    except Exception as error:
        print("No se pudo detectar Camo por nombre:", error)

    # Alternativa: indice conocido de tu configuracion.
    camara = cv2.VideoCapture(CAMARA_INDICE, cv2.CAP_DSHOW)

    if not camara.isOpened():
        camara.release()
        camara = cv2.VideoCapture(CAMARA_INDICE)

    if not camara.isOpened():
        raise RuntimeError(
            "No se pudo abrir Camo. Abre Camo Studio "
            "y comprueba la conexion del iPhone."
        )

    return camara


def limpiar_etiqueta(etiqueta):
    partes = etiqueta.split(" ", 1)

    if len(partes) == 2 and partes[0].isdigit():
        return partes[1]

    return etiqueta


def clasificar_imagen(frame, modelo, clases):
    imagen = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    imagen = cv2.resize(
        imagen,
        TAMANO_IMAGEN,
        interpolation=cv2.INTER_AREA
    )

    imagen = np.asarray(imagen, dtype=np.float32)
    imagen = imagen / 127.5 - 1.0
    imagen = np.expand_dims(imagen, axis=0)

    predicciones = modelo.predict(imagen, verbose=0)[0]
    indice = int(np.argmax(predicciones))

    resultados = []

    for i, probabilidad in enumerate(predicciones):
        etiqueta = limpiar_etiqueta(clases[i])
        resultados.append(
            (etiqueta, float(probabilidad) * 100)
        )

    return resultados[indice][0], resultados[indice][1], resultados


# ==================================================
# APLICACION
# ==================================================

class QCVision(ctk.CTk):

    def __init__(self):
        super().__init__()

        self.title("QC VISION | Control de calidad")
        self.geometry("1280x820")
        self.minsize(1050, 700)
        self.configure(fg_color=FONDO)

        self.modelo = None
        self.clases = []
        self.camara = None
        self.frame_actual = None
        self.resultado_actual = None
        self.historial = []
        self.total_inspecciones = 0
        self.after_id = None
        self.cerrando = False

        self.crear_interfaz()
        self.iniciar_sistema()

        self.protocol("WM_DELETE_WINDOW", self.cerrar)
        self.bind("<space>", self.atajo_espacio)
        self.bind("<Escape>", lambda event: self.cerrar())

    def crear_interfaz(self):
        # ---------- CABECERA ----------
        cabecera = ctk.CTkFrame(
            self, fg_color=PANEL, corner_radius=14
        )
        cabecera.pack(fill="x", padx=18, pady=(18, 12))

        marca = ctk.CTkFrame(cabecera, fg_color="transparent")
        marca.pack(side="left", padx=20, pady=15)

        ctk.CTkLabel(
            marca,
            text="QC VISION",
            font=("Segoe UI", 27, "bold"),
            text_color=VERDE
        ).pack(anchor="w")

        ctk.CTkLabel(
            marca,
            text="SISTEMA INTELIGENTE DE INSPECCION",
            font=("Segoe UI", 11),
            text_color=GRIS
        ).pack(anchor="w")

        self.estado_sistema = ctk.CTkLabel(
            cabecera,
            text="● INICIANDO",
            text_color=AMARILLO,
            font=("Segoe UI", 13, "bold")
        )
        self.estado_sistema.pack(side="right", padx=22)

        # ---------- CONTENIDO PRINCIPAL ----------
        contenido = ctk.CTkFrame(
            self, fg_color="transparent"
        )
        contenido.pack(
            fill="both", expand=True, padx=18, pady=(0, 12)
        )
        contenido.grid_columnconfigure(0, weight=3)
        contenido.grid_columnconfigure(1, weight=2)
        contenido.grid_rowconfigure(0, weight=1)

        # ---------- CAMARA ----------
        panel_camara = ctk.CTkFrame(
            contenido, fg_color=PANEL, corner_radius=14
        )
        panel_camara.grid(
            row=0, column=0, sticky="nsew", padx=(0, 8)
        )

        titulo_camara = ctk.CTkFrame(
            panel_camara, fg_color="transparent"
        )
        titulo_camara.pack(fill="x", padx=18, pady=15)

        ctk.CTkLabel(
            titulo_camara,
            text="INSPECCION VISUAL",
            font=("Segoe UI", 16, "bold"),
            text_color=BLANCO
        ).pack(side="left")

        self.estado_camara = ctk.CTkLabel(
            titulo_camara,
            text="● CONECTANDO",
            text_color=AMARILLO,
            font=("Segoe UI", 11, "bold")
        )
        self.estado_camara.pack(side="right")

        self.video_label = ctk.CTkLabel(
            panel_camara,
            text="Esperando imagen de Camo...",
            fg_color="#0A0E0C",
            corner_radius=10
        )
        self.video_label.pack(
            fill="both", expand=True, padx=16, pady=(0, 12)
        )

        ctk.CTkLabel(
            panel_camara,
            text="Coloca la pieza frente a la camara y pulsa ANALIZAR",
            font=("Segoe UI", 11),
            text_color=GRIS
        ).pack(pady=(0, 14))

        # ---------- PANEL DE RESULTADOS ----------
        panel_derecho = ctk.CTkFrame(
            contenido, fg_color="transparent"
        )
        panel_derecho.grid(
            row=0, column=1, sticky="nsew", padx=(8, 0)
        )

        resultado = ctk.CTkFrame(
            panel_derecho, fg_color=PANEL, corner_radius=14
        )
        resultado.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            resultado,
            text="RESULTADO DEL ANALISIS",
            font=("Segoe UI", 14, "bold"),
            text_color=BLANCO
        ).pack(anchor="w", padx=18, pady=(16, 8))

        self.resultado_label = ctk.CTkLabel(
            resultado,
            text="Sin inspeccion",
            font=("Segoe UI", 25, "bold"),
            text_color=GRIS,
            wraplength=320
        )
        self.resultado_label.pack(
            anchor="w", padx=18, pady=(6, 3)
        )

        self.confianza_label = ctk.CTkLabel(
            resultado,
            text="Confianza: --",
            font=("Segoe UI", 13),
            text_color=GRIS
        )
        self.confianza_label.pack(
            anchor="w", padx=18, pady=(0, 15)
        )

        ctk.CTkLabel(
            resultado,
            text="PROBABILIDAD POR CLASE",
            font=("Segoe UI", 11, "bold"),
            text_color=GRIS
        ).pack(anchor="w", padx=18, pady=(0, 8))

        self.barras = {}

        for etiqueta in [
            "Pieza_OK", "Chatter", "Rayadura", "Rebaba"
        ]:
            fila = ctk.CTkFrame(
                resultado, fg_color="transparent"
            )
            fila.pack(fill="x", padx=18, pady=5)

            ctk.CTkLabel(
                fila,
                text=etiqueta,
                font=("Segoe UI", 12),
                text_color=BLANCO,
                width=100,
                anchor="w"
            ).pack(side="left")

            barra = ctk.CTkProgressBar(
                fila,
                height=10,
                progress_color=VERDE,
                fg_color=PANEL_2
            )
            barra.pack(
                side="left", fill="x", expand=True, padx=8
            )
            barra.set(0)

            porcentaje = ctk.CTkLabel(
                fila,
                text="--%",
                width=48,
                text_color=GRIS,
                font=("Segoe UI", 11)
            )
            porcentaje.pack(side="right")

            self.barras[etiqueta] = (barra, porcentaje)

        self.boton_analizar = ctk.CTkButton(
            resultado,
            text="▶   ANALIZAR PIEZA",
            height=48,
            corner_radius=10,
            fg_color=VERDE_OSCURO,
            hover_color="#21633D",
            text_color=BLANCO,
            font=("Segoe UI", 15, "bold"),
            command=self.analizar
        )
        self.boton_analizar.pack(
            fill="x", padx=18, pady=(18, 8)
        )

        ctk.CTkLabel(
            resultado,
            text="Atajo: ESPACIO",
            text_color=GRIS,
            font=("Segoe UI", 10)
        ).pack(pady=(0, 14))

        # ---------- HISTORIAL ----------
        panel_historial = ctk.CTkFrame(
            panel_derecho, fg_color=PANEL, corner_radius=14
        )
        panel_historial.pack(
            fill="both", expand=True
        )

        ctk.CTkLabel(
            panel_historial,
            text="HISTORIAL DE INSPECCIONES",
            font=("Segoe UI", 14, "bold"),
            text_color=BLANCO
        ).pack(anchor="w", padx=16, pady=(14, 8))

        self.lista_historial = ctk.CTkTextbox(
            panel_historial,
            fg_color="#111714",
            text_color=BLANCO,
            font=("Consolas", 11),
            corner_radius=8,
            wrap="word"
        )
        self.lista_historial.pack(
            fill="both", expand=True, padx=12, pady=(0, 12)
        )
        self.lista_historial.insert(
            "end", "Todavia no hay inspecciones.\n"
        )
        self.lista_historial.configure(state="disabled")

        # ---------- BARRA INFERIOR ----------
        pie = ctk.CTkFrame(
            self, fg_color=PANEL, corner_radius=10
        )
        pie.pack(fill="x", padx=18, pady=(0, 18))

        self.total_label = ctk.CTkLabel(
            pie,
            text="INSPECCIONES: 0",
            text_color=GRIS,
            font=("Segoe UI", 11, "bold")
        )
        self.total_label.pack(side="left", padx=16, pady=10)

        ctk.CTkLabel(
            pie,
            text="MODELO IA  •  4 CLASES  •  CAMO",
            text_color=VERDE,
            font=("Segoe UI", 11, "bold")
        ).pack(side="right", padx=16, pady=10)

    def iniciar_sistema(self):
        try:
            self.modelo, self.clases = cargar_modelo()
            self.camara = abrir_camara()

            self.estado_sistema.configure(
                text="● SISTEMA LISTO", text_color=VERDE
            )
            self.estado_camara.configure(
                text="● CONECTADA", text_color=VERDE
            )

            self.actualizar_video()

        except Exception as error:
            self.estado_sistema.configure(
                text="● ERROR", text_color=ROJO
            )
            self.estado_camara.configure(
                text="● SIN CONEXION", text_color=ROJO
            )
            messagebox.showerror(
                "Error al iniciar QC Vision",
                str(error)
            )

    def actualizar_video(self):
        if self.cerrando or self.camara is None:
            return

        ok, frame = self.camara.read()

        if ok and frame is not None:
            self.frame_actual = frame.copy()

            alto, ancho = frame.shape[:2]
            escala = min(
                ANCHO_VIDEO / ancho,
                ALTO_VIDEO / alto
            )

            nuevo_ancho = int(ancho * escala)
            nuevo_alto = int(alto * escala)

            frame_mostrar = cv2.resize(
                frame, (nuevo_ancho, nuevo_alto)
            )
            frame_mostrar = cv2.cvtColor(
                frame_mostrar, cv2.COLOR_BGR2RGB
            )

            imagen = Image.fromarray(frame_mostrar)
            foto = ImageTk.PhotoImage(imagen)

            self.video_label.configure(image=foto, text="")
            self.video_label.image = foto

        else:
            self.estado_camara.configure(
                text="● SIN IMAGEN", text_color=ROJO
            )

        self.after_id = self.after(15, self.actualizar_video)

    def atajo_espacio(self, event):
        # Evitar analizar cuando el boton no esta disponible.
        if str(self.boton_analizar.cget("state")) == "normal":
            self.analizar()

    def analizar(self):
        if self.frame_actual is None or self.modelo is None:
            messagebox.showwarning(
                "Camara no disponible",
                "Espera a que llegue una imagen de Camo."
            )
            return

        self.boton_analizar.configure(
            state="disabled", text="ANALIZANDO..."
        )
        self.estado_sistema.configure(
            text="● ANALIZANDO", text_color=AMARILLO
        )

        # Copiar la imagen para analizar exactamente ese instante.
        frame = self.frame_actual.copy()

        try:
            nombre, confianza, resultados = clasificar_imagen(
                frame, self.modelo, self.clases
            )

            self.mostrar_resultado(nombre, confianza, resultados)
            self.guardar_historial(nombre, confianza)

        except Exception as error:
            messagebox.showerror(
                "Error de clasificacion", str(error)
            )

        finally:
            self.boton_analizar.configure(
                state="normal", text="▶   ANALIZAR PIEZA"
            )
            self.estado_sistema.configure(
                text="● SISTEMA LISTO", text_color=VERDE
            )

    def mostrar_resultado(self, nombre, confianza, resultados):
        self.resultado_actual = nombre

        nombre_normalizado = nombre.lower()

        if "ok" in nombre_normalizado:
            color = VERDE
        elif any(
            defecto in nombre_normalizado
            for defecto in ["chatter", "rayadura", "rebaba"]
        ):
            color = ROJO
        else:
            color = AMARILLO

        self.resultado_label.configure(
            text=nombre, text_color=color
        )
        self.confianza_label.configure(
            text=f"Confianza: {confianza:.1f}%"
        )

        for etiqueta, probabilidad in resultados:
            if etiqueta in self.barras:
                barra, texto = self.barras[etiqueta]
                barra.set(max(0, min(1, probabilidad / 100)))
                texto.configure(text=f"{probabilidad:.1f}%")

        # Mantener tambien visibles clases con otras etiquetas.
        for etiqueta, (barra, texto) in self.barras.items():
            if etiqueta not in [r[0] for r in resultados]:
                barra.set(0)
                texto.configure(text="0.0%")

    def guardar_historial(self, nombre, confianza):
        self.total_inspecciones += 1
        hora = datetime.now().strftime("%H:%M:%S")

        self.historial.insert(
            0, f"{hora} | {nombre} | {confianza:.1f}%"
        )
        self.historial = self.historial[:100]

        self.lista_historial.configure(state="normal")
        self.lista_historial.delete("1.0", "end")

        for i, registro in enumerate(self.historial, start=1):
            self.lista_historial.insert(
                "end", f"{i:02d}. {registro}\n"
            )

        self.lista_historial.configure(state="disabled")
        self.total_label.configure(
            text=f"INSPECCIONES: {self.total_inspecciones}"
        )

        print(f"[{hora}] {nombre}: {confianza:.1f}%")

    def cerrar(self):
        self.cerrando = True

        if self.after_id is not None:
            try:
                self.after_cancel(self.after_id)
            except Exception:
                pass

        if self.camara is not None:
            self.camara.release()

        self.destroy()


if __name__ == "__main__":
    app = QCVision()
    app.mainloop()
