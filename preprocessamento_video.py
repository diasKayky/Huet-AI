# Importações gerais
import os
import cv2
import glob
import pandas as pd
import cv2
import gc
import seaborn as sns
import numpy as np
import random
import imageio
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm
from tensorflow_docs.vis import embed
import matplotlib.pyplot as plt


def redimensionar_e_cortar_videos(diretorio_entrada, diretorio_saida, proporcao_corte=0.8, tamanho_alvo=(320, 320)):
    # Cria o diretório de saída se ele não existir
    if not os.path.exists(diretorio_saida):
        os.makedirs(diretorio_saida)

    # Percorre cada diretório de classe
    for nome_classe in os.listdir(diretorio_entrada):
        diretorio_classe = os.path.join(diretorio_entrada, nome_classe)

        # Cria o diretório de classe de saída
        diretorio_classe_saida = os.path.join(diretorio_saida, nome_classe)
        if not os.path.exists(diretorio_classe_saida):
            os.makedirs(diretorio_classe_saida)

        # Percorre cada vídeo no diretório da classe
        for arquivo_video in os.listdir(diretorio_classe):
            caminho_video = os.path.join(diretorio_classe, arquivo_video)

            # Lê o vídeo usando o OpenCV
            cap = cv2.VideoCapture(caminho_video)

            # Obtém as dimensões do vídeo
            largura = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            altura = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Calcula o número de linhas a serem cortadas
            linhas_corte = int(altura * (1 - proporcao_corte))

            # Percorre cada quadro no vídeo
            quadros = []
            while True:
                ret, quadro = cap.read()
                if not ret:
                    break

                # Corta o quadro da parte inferior
                quadro_cortado = quadro[:-linhas_corte, :]

                # Redimensiona o quadro cortado
                quadro_redimensionado = cv2.resize(quadro_cortado, tamanho_alvo, interpolation=cv2.INTER_AREA)

                quadros.append(quadro_redimensionado)

            # Salva o vídeo processado
            caminho_video_saida = os.path.join(diretorio_classe_saida, arquivo_video)
            out = cv2.VideoWriter(caminho_video_saida, cv2.VideoWriter_fourcc(*'mp4v'), cap.get(cv2.CAP_PROP_FPS), tamanho_alvo)

            for quadro in quadros:
                out.write(quadro)

            out.release()
            cap.release()

if __name__ == "__main__":
    diretorio_entrada = "dados"
    diretorio_saida = "dados_processados"
    redimensionar_e_cortar_videos(diretorio_entrada, diretorio_saida)

    
