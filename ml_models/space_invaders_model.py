import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError
# TODO from ml_metrics

def create_SpaceInvaders_model(env_shape, action_shape, **kwargs):
    # Define layers
    input_layer = Input(shape=env_shape)
    c1 = Conv2D(16, (3,3), strides=(1,1), 
                            padding='same', 
                            activation='relu')(input_layer)
    c2 = Conv2D(16, (3,3), strides=(1,1), 
                            padding='same', 
                            activation='relu')(c1)
    fl = Flatten()(c2)
    d1 = Dense(24, activation='relu')(fl)
    d2 = Dense(24, activation='relu')(d1)
    output_layer = Dense(action_shape, activation='linear')(d2)

    model = Model(input_layer, output_layer, name='CartPol_model')
    
    # create loss
    delta = kwargs.get('loss_delta', 1.0)
        # delta loss is the value at which
        # the huber loss function transitions from
        # a quadratic function to a linear funtion
    # loss = Huber(delta=delta)
    loss = MeanSquaredError()

    # create optimizer
    LR = kwargs.get('LR', 0.001)
        # learning rate
    LR_decay1 = kwargs.get('LR_decay1', 0.9)
        # Learning rate decay
    LR_decay2 = kwargs.get('LR_decay2', 0.999)
        # The exponential decay rate for the 2nd moment estimates
    optimizer = Adam(learning_rate=LR, beta_1=LR_decay1, beta_2=LR_decay2)

    model.compile(optimizer=optimizer, loss=loss, metrics=None)
    print(model.summary())
    return model

