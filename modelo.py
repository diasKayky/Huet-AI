import tensorflow as tf


class GlobalAveragePooling3D(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalAveragePooling3D, self).__init__()

    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=[1, 2, 3])


class AtencaoEspacoTemporal(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AtencaoEspacoTemporal, self).__init__()
        self.camada_atencao_espacial = tf.keras.layers.Conv3D(1, kernel_size=(1, 1, 1), activation='sigmoid')
        self.camada_atencao_temporal = tf.keras.layers.Dense(1, activation='softmax')

    def build(self, input_shape):

        _, self.dimensao_temporal, self.altura, self.largura, _ = input_shape
        self.camada_agrupamento = GlobalAveragePooling3D()

    def call(self, inputs):
        # Atenção Espacial
        atencao_espacial = self.camada_atencao_espacial(inputs)

        # Atenção Temporal
        caracteristicas_agrupadas = self.camada_agrupamento(inputs)
        atencao_temporal = self.camada_atencao_temporal(caracteristicas_agrupadas)
        atencao_temporal = tf.keras.layers.Reshape((1, 1, 1, 1))(atencao_temporal)  # Altere o redimensionamento aqui

        # Combinando Atenção Espacial e Temporal
        atencao = tf.keras.layers.Multiply()([atencao_espacial, atencao_temporal])

        # Aplicando a atenção às características de entrada
        caracteristicas_atendidas = tf.keras.layers.Multiply()([inputs, atencao])

        # Retornando as características atendidas
        return caracteristicas_atendidas


def construir_modelo(input_shape, num_classes=8, kernel_size=(3, 3, 3), pool_size=(1, 1, 1)):
    inputs = tf.keras.Input(shape=input_shape)

    # Primeiras camadas de CNN 3D
    x = tf.keras.layers.Conv3D(5, kernel_size=kernel_size, padding="same", activation='relu',
                               kernel_regularizer=tf.keras.regularizers.l2(0.1))(inputs)
    x = tf.keras.layers.MaxPooling3D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout3D(0.1)(x)

    # Atenção Espaço-Temporal
    x = AtencaoEspacoTemporal()(x)

    # Segundas camadas de CNN 3D
    x = tf.keras.layers.Conv3D(2, kernel_size=kernel_size, activation='relu', padding="same",
                               kernel_regularizer=tf.keras.regularizers.l2(0.1))(x)
    x = tf.keras.layers.MaxPooling3D(pool_size=pool_size)(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.SpatialDropout3D(0.1)(x)

    # Achatar e adicionar camadas Dense para classificação
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

    modelo = tf.keras.Model(inputs=inputs, outputs=outputs)

    return modelo


# Exemplo de uso com formato de entrada e número de classes
input_shape = (10, 320, 320, 3)  # Substitua pelo formato de entrada desejado
modelo = construir_modelo(input_shape)
modelo.summary()
