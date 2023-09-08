import tensorflow as tf
import cv2
import numpy as np
import random

def formatar_quadros(quadro, tamanho_saida):
    quadro = tf.image.convert_image_dtype(quadro, tf.float32)
    quadro = tf.image.resize_with_pad(quadro, *tamanho_saida)
    return quadro

def quadros_do_arquivo_de_video(caminho_video, n_quadros, tamanho_saida=(320, 320), passo_quadro=8):
    # Lê cada quadro do vídeo quadro a quadro
    resultado = []
    src = cv2.VideoCapture(str(caminho_video))

    comprimento_video = src.get(cv2.CAP_PROP_FRAME_COUNT)
    comprimento_necessario = 1 + (n_quadros - 1) * passo_quadro

    if comprimento_necessario > comprimento_video:
        inicio = 0
    else:
        inicio_maximo = comprimento_video - comprimento_necessario
        inicio = random.randint(0, inicio_maximo + 1)

    src.set(cv2.CAP_PROP_POS_FRAMES, inicio)
    # ret é um booleano que indica se a leitura foi bem-sucedida, quadro é a imagem em si
    ret, quadro = src.read()
    resultado.append(formatar_quadros(quadro, tamanho_saida))

    for _ in range(n_quadros - 1):
        for _ in range(passo_quadro):
            ret, quadro = src.read()
        if ret:
            quadro = formatar_quadros(quadro, tamanho_saida)
            resultado.append(quadro)
        else:
            resultado.append(np.zeros_like(resultado[0]))
    src.release()
    resultado = np.array(resultado)[..., [2, 1, 0]]

    return resultado


def criar_gif(imagens):
    imagens_convertidas = np.clip(imagens * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./animação.gif', imagens_convertidas, fps=8)

    return embed.embed_file('./animação.gif')

caminhos_arquivos = []
alvos = []
classes = [
    "oi", "a", "k", "y", "tudo_bem", "obrigado", "nome", "y"
]
for i, classe in enumerate(classes):
    sub_caminhos_arquivos = glob.glob(f'dados_processados/{classe}/**.mp4')
    caminhos_arquivos += sub_caminhos_arquivos
    alvos += [i] * len(sub_caminhos_arquivos)

caracteristicas = []
for caminho_arquivo in tqdm(caminhos_arquivos):
    caracteristicas.append(quadros_do_arquivo_de_video(caminho_arquivo, n_quadros=10))
caracteristicas = np.array(caracteristicas)

caracteristicas_treino, caracteristicas_validacao, alvos_treino, alvos_validacao = train_test_split(caracteristicas, alvos, test_size=0.2, random_state=42)
caracteristicas_treino.shape, caracteristicas_validacao.shape, len(alvos_treino), len(alvos_validacao)


# Conjunto de Dados
conjunto_dados_treino = tf.data.Dataset.from_tensor_slices((caracteristicas_treino, alvos_treino)).shuffle(10 * 4).batch(10).cache().prefetch(tf.data.AUTOTUNE)
conjunto_dados_validacao = tf.data.Dataset.from_tensor_slices((caracteristicas_validacao, alvos_validacao)).batch(10).cache().prefetch(tf.data.AUTOTUNE)

