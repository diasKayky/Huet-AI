import cv2
import numpy as np
import tensorflow as tf

# Carregar o modelo
def carregar_modelo_com_camadas_personalizadas(caminho_modelo):
    with tf.keras.utils.custom_object_scope({
        'GlobalAveragePooling3D': tf.keras.layers.GlobalAveragePooling3D,
        'AtencaoEspacoTemporal': AtencaoEspacoTemporal
    }):
        modelo_carregado = tf.keras.models.load_model(caminho_modelo)
    return modelo_carregado

modelo = carregar_modelo_com_camadas_personalizadas("HuetAI_v0.h5")

# Inicializar a captura da webcam
cap = cv2.VideoCapture(0)  # 0 representa a webcam padrão

# Inicializar uma lista para armazenar os quadros
sequencia_quadros = []

while True:
    ret, quadro = cap.read()  # Ler um quadro da webcam
    if not ret:
        break

    # Pré-processar o quadro
    quadro_redimensionado = cv2.resize(quadro, (320, 320))
    quadro_normalizado = quadro_redimensionado / 255.0

    sequencia_quadros.append(quadro_normalizado)

    # Manter apenas os 10 quadros mais recentes
    if len(sequencia_quadros) > 10:
        sequencia_quadros.pop(0)

    if len(sequencia_quadros) == 10:
        # Converter a sequência de quadros em um tensor de entrada
        dados_entrada = np.array([sequencia_quadros])

        # Realizar a previsão
        previsoes = modelo.predict(dados_entrada)
        indice_classe_prevista = np.argmax(previsoes)

        # Exibir o quadro com a previsão
        cv2.putText(quadro, f'Classe Prevista: {indice_classe_prevista}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', quadro)

    # Sair do loop quando 'q' for pressionado
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar as janelas do OpenCV
cap.release()
cv2.destroyAllWindows()
