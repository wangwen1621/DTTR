# -*- codeing = utf-8 -*-
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dropout,Dense
from tensorflow.keras.layers import MultiHeadAttention, Add, Flatten,LSTM,Conv1D,ZeroPadding1D, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential, layers, utils
from sklearn.preprocessing import MinMaxScaler
import statsmodels.api as sm
from scipy.stats import pearsonr
#  model evaluation metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error ,r2_score

from scipy import io


def create_new_dataset(dataset, seq_len):
    X = []
    y = []

    start = 0
    end = dataset.shape[0] - seq_len

    for i in range(start, end):  # for loop to construct feature dataset
        sample = dataset[i: i + seq_len]  # Create samples based on time span seq_len
        label = dataset[i + seq_len]
        X.append(sample)
        y.append(label)

    return np.array(X), np.array(y)

# Training and test set partition function
def split_dataset(X, y, train_ratio=0.8):
    X_len = len(X)

     train_data_len = int(X_len * train_ratio)  # Number of samples in the training set

    X_train = X[:train_data_len]  # training set
    y_train = y[:train_data_len]

    X_test = X[train_data_len:]  # test set
    y_test = y[train_data_len:]

    return X_train, X_test, y_train, y_test

# DCRN
def DCRN(input_layer, n_filters, kernel_size, dilation_rate):

    padding_layer = ZeroPadding1D(padding=(dilation_rate * (kernel_size - 1), 0))(input_layer)
    # Convolutional layer
    conv_layer = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='valid',
                        activation='relu')(padding_layer)
    # Approved standardisation layer
    bn_layer = BatchNormalization()(conv_layer)
    # Dropout Layer
    dropout_layer = Dropout(drop)(bn_layer)

    padding_layer_2 = ZeroPadding1D(padding=(dilation_rate * (kernel_size - 1), 0))(dropout_layer)
    #  Convolutional Layer II
    conv_layer_2 = Conv1D(filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation_rate, padding='valid',
                        activation='relu')(padding_layer_2)
    # Approval of standardisation layer II
    bn_layer_2 = BatchNormalization()(conv_layer_2)
    # Dropout Layer II
    dropout_layer_2 = Dropout(drop)(bn_layer_2)

    # residual link
    Res = Conv1D(filters=n_filters, kernel_size=kernel_size, padding='same')(input_layer)
    residual_layer = Add()([Res, dropout_layer_2])
    return residual_layer


# TAM layer
def attention_block(input_layer):
    # relay operation
    transpose_layer = tf.keras.layers.Permute((2, 1))(input_layer)
    # Dot multiplication
    dot_layer = tf.keras.layers.Dot(axes=[2, 1])([input_layer, transpose_layer])
    # Softmax
    softmax_layer = tf.keras.layers.Softmax()(dot_layer)
    # weighted multiplication
    attention_layer = tf.keras.layers.Dot(axes=[1, 1])([input_layer, softmax_layer])
    return attention_layer


# Building the Transformer Model
class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerEncoder, self).__init__()

        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)


        self.dense1 = layers.Dense(dense_dim, activation="tanh")
        self.dense2 = layers.Dense(embed_dim)
        self.lstm1=layers.LSTM(dense_dim, activation="relu")
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training):
        attn_output = self.mha(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        dense_output = self.dense1(out1)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output, training=training)
        out2 = self.layernorm2(out1 + dense_output)

        dense_output = self.dense1(out2)
        dense_output = self.dense2(dense_output)
        dense_output = self.dropout2(dense_output, training=training)
        out3 = self.layernorm2(out2 + dense_output)

        dense_output = self.dense1(out3)
        dense_output = self.dense2(dense_output)
        out4 = self.layernorm2(out3 + dense_output)

        return out4

class TransformerDecoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate):
        super(TransformerDecoder, self).__init__()

        self.mha1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.mha2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)
        self.dropout3 = layers.Dropout(dropout_rate)


        self.dense1 = layers.Dense(dense_dim, activation="relu")
        self.dense2 = layers.Dense(embed_dim)
        self.layernorm4 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout4 = layers.Dropout(dropout_rate)

    def call(self, inputs, encoder_outputs, training):
        attn1 = self.mha1(inputs, inputs)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(inputs + attn1)

        attn2 = self.mha2(out1, encoder_outputs)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        dense_output = self.dense1(out2)
        dense_output = self.dense2(dense_output)
        # dense_output = self.dropout3(dense_output, training=training)
        out3 = self.layernorm3(out2 + dense_output)

        decoder_output = self.dense1(out3)
        decoder_output = self.dense2(decoder_output)
        # decoder_output = self.dropout4(decoder_output, training=training)
        out4 = self.layernorm4(out3 + decoder_output)

        # decoder_output = self.dense1(out4)
        # decoder_output = self.dense2(decoder_output)
        # decoder_output = self.dropout4(decoder_output, training=training)
        # out5 = self.layernorm4(out4 + decoder_output)
        return out4

class Transformer(keras.Model):
    def __init__(self, embed_dim, dense_dim, num_heads, dropout_rate,num_blocks, output_sequence_length):
        super(Transformer, self).__init__()

        self.embedding = layers.Dense(embed_dim, activation="relu")
        self.transformer_encoder = [TransformerEncoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)]

        self.transformer_decoder = [TransformerDecoder(embed_dim, dense_dim, num_heads, dropout_rate) for _ in range(num_blocks)]

        self.final_layer = layers.Dense(output_sequence_length)

    def call(self, inputs, training):
        encoder_inputs = inputs
        decoder_inputs = inputs

        encoder_outputs = self.embedding(encoder_inputs)
        for i in range(len(self.transformer_encoder)):
            encoder_outputs = self.transformer_encoder[i](encoder_outputs, training=training)

        decoder_outputs = self.embedding(decoder_inputs)
        for i in range(len(self.transformer_decoder)):
            decoder_outputs = self.transformer_decoder[i](decoder_outputs, encoder_outputs, training=training)

        decoder_outputs = tf.reshape(decoder_outputs, [-1, decoder_outputs.shape[1] * decoder_outputs.shape[2]])
        decoder_outputs = self.final_layer(decoder_outputs)
        decoder_outputs = tf.reshape(decoder_outputs, [-1, T])
        return decoder_outputs

# Building a  model DTTR
def build_model(input_shape,n_filters,embed_dim,dense_dim,num_heads,num_blocks):
    # Input layer
    input_layer = Input(shape=input_shape)
    #DCRN
    DCRNlayer = DCRN(input_layer, n_filters=n_filters, kernel_size=3, dilation_rate=1)
    # TAM
    attention = attention_block(DCRNlayer)
    #Transformer
    Trans_layer=Transformer(embed_dim=embed_dim, dense_dim=dense_dim, num_heads=num_heads, dropout_rate=0.001,num_blocks=num_blocks, output_sequence_length=T)(attention)
    # Dropout
    dropout = Dropout(drop)(Trans_layer)
    # Output layer
    output_layer = Dense(1, activation='linear')(dropout)
    # Defining the model
    model = Model(inputs=input_layer, outputs=output_layer)
    return model
