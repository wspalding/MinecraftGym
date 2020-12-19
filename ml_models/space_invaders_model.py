import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, TimeDistributed
from tensorflow.keras.layers import Flatten, Input, Conv2D
from tensorflow.keras.layers import LSTM, RNN, Reshape
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber, MeanSquaredError
# TODO from ml_metrics

def create_SpaceInvaders_model(env_shape, action_shape, **kwargs):
    # Define layers
    input_layer = Input(shape=env_shape)
    c1 = Conv2D(16, (10,10), strides=(1,1), 
                            padding='same', 
                            activation='relu')(input_layer)
    c2 = Conv2D(32, (5,5), strides=(1,1), 
                            padding='same', 
                            activation='relu')(c1)
    c3 = Conv2D(64, (4,4), strides=(1,1), 
                            padding='same', 
                            activation='relu')(c2)
    c4 = Conv2D(16, (3,3), strides=(1,1), 
                            padding='same', 
                            activation='relu')(c3)
    fl = Flatten()(c4)
    d1 = Dense(64, activation='relu')(fl)
    d2 = Dense(256, activation='relu')(d1)
    d2 = Dense(256, activation='relu')(d2)
    d2 = Dense(512, activation='relu')(d2)
    d2 = Dense(64, activation='relu')(d2)
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
















def create_SpaceInvaders_lstm(env_shape, action_shape, **kwargs):
    # Define layers
    input_layer = Input(shape=env_shape)
    c1 = Conv2D(16, (3,3), strides=(1,1), 
                            padding='same', 
                            activation='relu')(input_layer)
    c2 = Conv2D(32, (4,4), strides=(1,1), 
                            padding='same', 
                            activation='relu')(c1)
    c3 = Conv2D(64, (5,5), strides=(1,1), 
                            padding='same', 
                            activation='relu')(c2)
    c4 = Conv2D(16, (10,10), strides=(10,10), 
                            padding='same', 
                            activation='relu')(c3)
    
    fl = Flatten()
    td = TimeDistributed(fl)(c4)
    #d1 = Dense(64, activation='relu')(fl)
    lstm1 = LSTM(16, return_sequences=True)(td)
    lstm2 = LSTM(64, return_sequences=True)(lstm1)
    #d2 = Dense(256, activation='relu')(d1)
    lstm3 = LSTM(256, return_sequences=True)(lstm2)
    lstm4 = LSTM(512, return_sequences=True)(lstm3)
    lstm5 = LSTM(64)(lstm4)
    output_layer = Dense(action_shape, activation='linear')(lstm5)

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

