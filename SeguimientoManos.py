import math
import cv2
import mediapipe as mp
import time

class detectormanos():
    # Inicializamos los parámetros de la detección
    def __init__(self, mode=False, maxManos=2, Confdeteccion=0.5, Confsegui=0.5):
        self.mode = mode  
        self.maxManos = maxManos  
        self.Confdeteccion = Confdeteccion
        self.Confsegui = Confsegui

        # Creamos los objetos que detectarán las manos y las dibujarán
        self.mpmanos = mp.solutions.hands
        self.manos = self.mpmanos.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxManos,
            min_detection_confidence=self.Confdeteccion,
            min_tracking_confidence=self.Confsegui)
        self.dibujo = mp.solutions.drawing_utils
        self.tip = [4, 8, 12, 16, 20]  # Lista de índices de las puntas de los dedos

    # Función para encontrar las manos
    def encontrarmanos(self, frame, dibujar=True):
        imgcolor = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.resultados = self.manos.process(imgcolor)
        
        if self.resultados.multi_hand_landmarks:
            for mano in self.resultados.multi_hand_landmarks:
                if dibujar:
                    # Dibujamos las conexiones de los puntos de la mano en el frame
                    self.dibujo.draw_landmarks(frame, mano, self.mpmanos.HAND_CONNECTIONS)
        
        return frame

    # Función para encontrar la posición de la mano
    def encontrarposicion(self, frame, ManoNum=0, dibujar=True):
        xlista = []
        ylista = []
        bbox = []
        self.lista = []

        if self.resultados.multi_hand_landmarks:
            miMano = self.resultados.multi_hand_landmarks[ManoNum]
            for id, lm in enumerate(miMano.landmark):
                alto, ancho, c = frame.shape  # Extraemos las dimensiones del frame
                cx, cy = int(lm.x * ancho), int(lm.y * alto)  # Convertimos la información en píxeles
                xlista.append(cx)
                ylista.append(cy)
                self.lista.append([id, cx, cy])

                if dibujar:
                    # Dibujamos un círculo en cada punto de referencia
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 0), cv2.FILLED)

            # Determinamos las coordenadas mínimas y máximas para definir una caja de contorno
            xmin, xmax = min(xlista), max(xlista)
            ymin, ymax = min(ylista), max(ylista)
            bbox = xmin, ymin, xmax, ymax

            if dibujar:
                # Dibujamos un rectángulo alrededor de la mano detectada
                cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return self.lista, bbox
    
    # Función para detectar y dibujar los dedos que están arriba
    def dedosarriba(self):
        dedos = []

        # Comprobamos el pulgar
        if self.lista[self.tip[0]][1] > self.lista[self.tip[0] - 1][1]:
            dedos.append(1)  # Pulgar arriba
        else:
            dedos.append(0)  # Pulgar abajo

        # Comprobamos los otros cuatro dedos
        for id in range(1, 5):
            if self.lista[self.tip[id]][2] < self.lista[self.tip[id] - 2][2]:
                dedos.append(1)  # Dedo arriba
            else:
                dedos.append(0)  # Dedo abajo

        return dedos

    # Función para detectar la distancia entre dos puntos específicos (dedos)
    def distancia(self, p1, p2, frame, dibujar=True, r=15, t=3):
        x1, y1 = self.lista[p1][1:]  # Coordenadas del primer punto
        x2, y2 = self.lista[p2][1:]  # Coordenadas del segundo punto
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Coordenadas del punto medio

        if dibujar:
            # Dibujar una línea entre los dos puntos
            cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), t)
            # Dibujar círculos en los puntos y en el centro
            cv2.circle(frame, (x1, y1), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (x2, y2), r, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, (cx, cy), r, (0, 0, 255), cv2.FILLED)

        # Calcular la longitud de la línea (distancia euclidiana entre los dos puntos)
        length = math.hypot(x2 - x1, y2 - y1)

        return length, frame, [x1, y1, x2, y2, cx, cy]

def main():
    ptiempo = 0
    ctiempo = 0

    # Leemos la cámara web
    cap = cv2.VideoCapture(0)
    # Creamos el objeto detector de manos
    detector = detectormanos()

    while True:
        ret, frame = cap.read()
        frame = detector.encontrarmanos(frame)
        lista, bbox = detector.encontrarposicion(frame)

        # Si hay al menos un punto detectado, imprimimos las coordenadas del quinto punto
        if len(lista) != 0:
            print(lista[4])

        # Mostramos los FPS
        ctiempo = time.time()
        fps = 1 / (ctiempo - ptiempo)
        ptiempo = ctiempo

        # Mostrar los FPS en el frame
        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        # Mostrar el frame en una ventana
        cv2.imshow("Manos", frame)
        k = cv2.waitKey(1)

        # Salir del bucle si se presiona la tecla 'ESC' (código 27)
        if k == 27:
            break

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
