from modelo import modelo
import tensorflow as tf

modelo.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              optimizer=tf.keras.optimizers.SGD(0.001),
              metrics=["accuracy"])

ponto_de_verificacao = tf.keras.callbacks.ModelCheckpoint(
    "HuetAI_v0.h5",
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    restore_best_weights=True
)

if __name__ == "__main__":

    historico = modelo.fit(
        conjunto_dados_treino,
        epochs=10,
        validation_data=conjunto_dados_validacao,
        callbacks=[ponto_de_verificacao]
    )
