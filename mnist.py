import os
import numpy as np
import cv2
import keras
import keras_tuner
from keras import layers

# general config params for saving files
config = {'model_path': "mnist.keras",
          'model_graph': "mnist.png",
          'logs': os.path.abspath('./logs')}

# tuneable model hyperparameters
def hyperparameters(hp):
  return {'neurons': hp.Int("neurons", min_value=32, max_value=512, step=32),
          'activation': hp.Choice("activation", ["relu", "tanh"]),
          'dropout': hp.Boolean("dropout"),
          'dropout_rate': hp.Float("droprate", min_value=0.1, max_value=0.3, sampling="log"),
          'learning_rate': hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")}

def load_mnist_data():
  "Loads the MNIST data, flattens it into a row vector and normalizes as float values."
  pixels, colorspace = (28**2, 255.0)
  (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
  x_train = x_train.reshape(60000, pixels).astype("float32") / colorspace
  x_test = x_test.reshape(10000, pixels).astype("float32") / colorspace
  return ((x_train, y_train), (x_test, y_test))

def save_image_array(path, image_data):
  "Saves an image to the path from an array with width x height x 3 (channels) dimensions"
  image = keras.utils.array_to_img(image_data)
  keras.utils.save_img(path, image, "channels_last")

def save_mnist_dataset_image(idx, dataset="train"):
  "Loads the MNIST grayscale training image by index, converts to RGB and saves it."
  train, test = keras.datasets.mnist.load_data()
  data, labels = (train if dataset == "train" else test)
  arr, label = (data[idx] * 255).astype(np.uint8), labels[idx]
  rgb = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
  path = f"mnist/{dataset}/{label}/{idx}.png"
  save_image_array(path, rgb)

def mnist_model(hyperparams):
  "Returns the neural network model for the MNIST dataset."
  inputs = keras.Input(shape=(784,), name="inputs")
  layer1 = layers.Dense(hyperparams['neurons'],
                         activation=hyperparams['activation'], name="hidden_1")(inputs)
  layer2 = layers.Dense(hyperparams['neurons'],
                         activation=hyperparams['activation'], name="hidden_2")(layer1)
  if hyperparams['dropout']:
    last_layer =  layers.Dropout(rate=hyperparams['dropout_rate'])(layer2)
  else:
    last_layer = layer2
  outputs = layers.Dense(10, name="outputs")(last_layer)
  model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
  return model

def compile_model(model, hyperparams):
  "Returns the model compiled with an optimized learning rate and loss function."
  model.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.RMSprop(hyperparams['learning_rate']),
    metrics=["sparse_categorical_accuracy"])
  return model

def build_tunable_model(hp):
  "Builds and compiles a model with autotuneable hyperparameters."
  hyperparams = hyperparameters(hp)
  model = mnist_model(hyperparams)
  return compile_model(model, hyperparams)

def evaluate_test_data(model, x_test, y_test, batch_size=64, verbose=2):
  "Evaluates the model using the test data and its labels."
  test_scores = model.evaluate(x_test, y_test,
                               batch_size=batch_size,
                               verbose=verbose)
  print("Test loss:", test_scores[0])
  print("Test accuracy:", test_scores[1])

def train_model(x_train, y_train):
  "Trains a model with the Keras autotuner and saves the best model."
  callbacks = [
    keras.callbacks.EarlyStopping(
      monitor="val_loss",
      min_delta=1e-3,
      patience=2,
      verbose=1
    ),
    keras.callbacks.TensorBoard(
      log_dir=config['logs'],
      histogram_freq=0,   # how often to log histogram visualizations
      embeddings_freq=0,  # how often to log embedding visualizations
      update_freq="epoch")]
  tuner = keras_tuner.RandomSearch(
    hypermodel=build_tunable_model,
    objective="sparse_categorical_accuracy",
    max_trials=5,
    executions_per_trial=2,
    overwrite=False,
    directory="tuner",
    project_name="mnist-keras",
  )
  print("Tuning hyperparameters for optimal accuracy.")
  print(tuner.search_space_summary())
  tuner.search(x_train, y_train,
               epochs=10,
               callbacks=callbacks,
               validation_split=0.2)
  best_models = tuner.get_best_models(num_models=2)
  model = best_models[0]
  print("Found best hyperparameters:")
  print(model.summary())
  keras.utils.plot_model(model, config['model_graph'],
                         show_shapes=True,
                         show_dtype=True,
                         show_layer_names=True,
                         show_layer_activations=True,
                         show_trainable=True,)
  model.save(config['model_path'])
  return model

def load_model(x_train, y_train, retrain=False):
  "Returns a previously saved model if one exists, or trains a new one."
  if os.path.isfile(config['model_path']) and not retrain:
    model = keras.models.load_model(config['model_path'])
  else:
    model = train_model(x_train, y_train)
  return model

def evaluate_handdrawn_images(model, verbose=True):
  "Evaluates the model's performance on 10 hand-drawn images I made."
  correct = 0
  for n in range(10):
    image = keras.utils.load_img(f"mnist/mine/{n}.png", color_mode="grayscale")
    data = keras.utils.img_to_array(image).reshape(1, 28**2) / 255.0
    prediction = np.argmax(model.predict(data))
    if verbose:
      print(f'Image: {n}, Prediction: {prediction}')
    if prediction == n:
      correct += 1
  accuracy = correct/10
  print(f"Accuracy: {accuracy:.0%}")

# use all the things
(x_train, y_train), (x_test, y_test) = load_mnist_data()
model = load_model(x_train, y_train, retrain=True)
evaluate_test_data(model, x_test, y_test)
evaluate_handdrawn_images(model, verbose=False)
