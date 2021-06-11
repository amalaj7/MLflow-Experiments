import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import mlflow
import seaborn as sns

mlflow.set_experiment("Fashion MNIST Experiments")

fashion_mnist = tf.keras.datasets.fashion_mnist
(xtrain_full,ytrain_full),(xtest,ytest)=fashion_mnist.load_data()

X_valid, X_train = xtrain_full[:5000] / 255., xtrain_full[5000:] / 255.
y_valid, y_train = ytrain_full[:5000], ytrain_full[5000:]

xtest = xtest / 255.

plt.figure(figsize=(15,15))
sns.heatmap(X_train[0], annot=True, cmap="binary")
plt.show()

mlflow.tensorflow.autolog()
with mlflow.start_run():
    LAYERS = [tf.keras.layers.Flatten(input_shape=[28,28],name = "InputLayer"),
              tf.keras.layers.Dense(300,activation="relu",name="HiddenLayer1"),
              tf.keras.layers.Dense(150,activation="relu",name="HiddenLayer2"),
              tf.keras.layers.Dense(10,activation="softmax",name="OutputLayer")]

    clf = tf.keras.models.Sequential(LAYERS)
    print(clf.summary())

    LOSS = "sparse_categorical_crossentropy"
    OPTIMIZER = "SGD"
    METRICS = ["accuracy"]

    clf.compile(loss=LOSS, optimizer=OPTIMIZER,metrics=METRICS)
    model_checkpoint_cb = tf.keras.callbacks.ModelCheckpoint("fashion_mnist_model.h5",save_best_only=True)
    early_stopping_cb = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir="logs")

    EPOCHS = 30

    Validation = (X_valid,y_valid)
    history = clf.fit(X_train,y_train,epochs=EPOCHS,validation_data=Validation,callbacks=[model_checkpoint_cb,early_stopping_cb,tensorboard_cb])
    pd.DataFrame(history.history).plot(figsize=(8, 5))
    plt.grid(True)
    plt.gca().set_ylim(0, 1)
    plt.show()
