"""Create and train Neural Network."""

from os import getcwd

from create_model import initial_model
from load_data import x_train1, x_train2, y_train, x_test1, x_test2, y_test

from keras.callbacks import ModelCheckpoint
from keras.models import load_model

cwd = getcwd() + '/facial_recognition'

try:
    model = load_model(cwd + '/models/fr.hdf5')
except OSError:
    model = initial_model


def train_model(model):
    """Train the neural network."""
    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['categorical_accuracy'])
    checkpoint = ModelCheckpoint(filepath=cwd + '/models/fr.hdf5',
                                 monitor='categorical_accuracy',
                                 save_best_only=True,
                                 mode='max')
    model.fit([x_train1, x_train2],
              y_train,
              epochs=1000,
              batch_size=500,
              validation_split=0.25,
              callbacks=[checkpoint])

    return model

print('\nTraining model...\n')
trained_model = train_model(initial_model)


def evaluate_model():
    """
    Evaluate the models accuracy.

    Test on both the train and test data.
    Return accuracies: {'training': acc, 'testing': acc}
    Test on all combinations of available pictures.
    """
    best_model = load_model(cwd + '/models/fr.hdf5')
    train_score = best_model.evaluate([x_train1, x_train2], y_train)
    test_score = best_model.evaluate([x_test1, x_test2], y_test)
    return train_score, test_score

print('\nEvaluating model...\n')
train_score, test_score = evaluate_model()

print('\nAccuracy training data:\n', train_score[1] * 100, '%')
print('\nAccuracy testing data:\n', test_score[1] * 100, '%')
