import tensorflow as tf 
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
# from ml_metrics

def create_CartPol_model_from_config(config):
    input_layer = Input(shape=config.input_shape)
    d1 = Dense(24, activation='relu')(input_layer)
    d2 = Dense(24, activation='relu')(d1)
    output_layer = Dense(config.action_shape, activation='softmax')(d2)

    model = Model(input_layer, output_layer, name='CartPol_model')
    
    loss = Huber()
    optimizer = Adam(learning_rate=config.LR, beta_1=config.LR_decay1)

    model.compile(optimizer=optimizer, loss=loss, metrics=None)
    return model

    