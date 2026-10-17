import cv2
import numpy as np
from tensorflow.keras.models import load_model

MODELO = "keras_model.h5"
ETIQUETAS = "labels.txt"


def encontrar_camo():
    """Devuelve el índice de la cámara Camo (Windows). Si no la halla, devuelve None."""
    try:
        from pygrabber.dshow_graph import FilterGraph
        dispositivos = FilterGraph().get_input_devices()
        print("Cámaras detectadas:")
        for i, nombre in enumerate(dispositivos):
            print(f"  [{i}] {nombre}")
        for i, nombre in enumerate(dispositivos):
            if "camo" in nombre.lower():
                return i
    except Exception as e:
        print("No se pudo listar cámaras por nombre:", e)
    return None


def abrir_camara(indice):
    # En Windows CAP_DSHOW funciona mejor con cámaras virtuales
    cap = cv2.VideoCapture(indice, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(indice)
    return cap


def main():
    np.set_printoptions(suppress=True)

    model = load_model(MODELO, compile=False)
    with open(ETIQUETAS, "r", encoding="utf-8") as f:
        clases = [line.strip() for line in f.readlines()]

    indice = encontrar_camo()
    if indice is None:
        # Si no se encontró por nombre, pon aquí el número manualmente (0, 1, 2...)
        indice = 1
        print(f"No se detectó Camo por nombre, probando índice {indice}")
    else:
        print(f"Camo detectada en el índice {indice}")

    cap = abrir_camara(indice)
    if not cap.isOpened():
        print("No se pudo abrir la cámara. Abre Camo Studio y prueba otro índice.")
        return

    print("Presiona 'q' para salir.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer el frame.")
            break

        # Preprocesado igual que Teachable Machine
        img = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, dtype=np.float32).reshape(1, 224, 224, 3)
        img = (img / 127.5) - 1

        prediccion = model.predict(img, verbose=0)
        i = int(np.argmax(prediccion))
        nombre = clases[i].split(" ", 1)[-1]  # quita el número del inicio
        confianza = float(prediccion[0][i]) * 100

        texto = f"{nombre}: {confianza:.1f}%"
        cv2.putText(frame, texto, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow("Teachable Machine + Camo", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()