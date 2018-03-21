"""
Create the model.
"""

from keras.layers import (
    concatenate, Conv2D, Dense, Dropout,
    Flatten, Input, MaxPooling2D)

from keras.models import Model
from keras.utils import plot_model


def create_model():
    """
    Create the Convolutional Neural Network.
    Return the initial model.
    The network will have two inputs and two possible outputs.
    """

    def create_tower(input_img, kernel_size, levels):
        """
        Create a tower of Convolutional, MaxPooling and Flatten layers.
        Each level has 2 convolutional and one maxpooling layer.
        """

        tower_objects = [input_img]
        filters, pool_size, dropout = 64, (2, 2), 0.4
        structure = {
            1: 2,
            2: 2,
            3: 2,
            4: 3,
            5: 3,
        }

        def create_cnn_layer():
            """
            Create a convolutional layer and its dropout layer.
            """
            conv_layer1 = Conv2D(filters=filters,
                                 kernel_size=kernel_size,
                                 padding='same',
                                 activation='relu')(tower_objects[-1])
            tower_objects.append(conv_layer1)
            dropout_layer = Dropout(dropout)(tower_objects[-1])
            tower_objects.append(dropout_layer)

        for i in range(1, levels+1):
            for j in range(structure[i]):
                create_cnn_layer()
            maxpooling_layer = MaxPooling2D(
                pool_size=pool_size)(tower_objects[-1])
            tower_objects.append(maxpooling_layer)
            filters = filters*2
        tower = Flatten()(tower_objects[-1])
        return tower

    inputs = [Input(shape=(128, 128, 1)) for i in range(2)]
    kernel_sizes = [(2, 2), (2, 2)]
    towers = [create_tower(input_img, kernel_size, 5)
              for input_img, kernel_size in zip(inputs, kernel_sizes)]
    merged = concatenate(towers)
    dense = Dense(4096, activation='relu')(merged)
    dense = Dropout(0.4)(dense)
    dense = Dense(1024, activation='relu')(dense)
    dense = Dropout(0.4)(dense)
    output = Dense(2, activation='softmax')(dense)
    model = Model(inputs=inputs, outputs=output)
    return model

model = create_model()
plot_model(model)
