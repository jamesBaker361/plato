import tensorflow as tf

from tensorflow.keras.layers import *

class VGGPreProcess(Layer):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self,inputs):
        return tf.keras.applications.vgg19.preprocess_input(inputs)

class ResnetPreProcess(Layer):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self,inputs):
        return tf.keras.applications.resnet50.preprocess_input(inputs)

class Identity(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return inputs

def clone_layer(layer):
    config = layer.get_config()
    weights = layer.get_weights()
    cloned_layer = type(layer).from_config(config)
    cloned_layer.build(layer.input_shape)
    cloned_layer.set_weights(weights)
    cloned_layer.trainable=False
    return cloned_layer

def resnet_layers(layer_names,input_shape):
    resnet=tf.keras.applications.resnet50.ResNet50(include_top=False,input_shape=input_shape)

def vgg_layers(layer_names,input_shape):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet',input_shape=input_shape)
    vgg.trainable = False
    inputs=tf.keras.Input(shape=input_shape)
    identity=Identity(name="identity")
    preproc=VGGPreProcess(name="prepocess")
    #x=vgg(x)

    #_model.layer_names=set([no_block,no_block_raw])

    layers=[inputs,identity,preproc]
    for layer in vgg.layers[1:]:
        layers.append(clone_layer(layer))

    _model= tf.keras.Sequential(layers)
    #vgg=Concatenate()([inputs,preprocess,vgg])
    #if name in _model.layer_names else _model.layers[-1].get_layer(name).output

    outputs = [_model.get_layer(name).output for name in layer_names ]

    return tf.keras.Model(_model.input, outputs)