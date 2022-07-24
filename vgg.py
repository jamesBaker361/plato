import tensorflow as tf

from tensorflow.keras.layers import *

class VGGPreProcess(Layer):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self,inputs):
        return tf.keras.applications.vgg19.preprocess_input(255*inputs)

class ResnetPreProcess(Layer):
    def __init__(self,*args,**kwargs) -> None:
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self,inputs):
        return tf.keras.applications.resnet50.preprocess_input(255*inputs)

class Identity(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        return super().build(input_shape)

    def call(self, inputs, *args, **kwargs):
        return inputs

def clone_layer(layer):
    '''It takes a layer and returns a copy of that layer
    
    Parameters
    ----------
    layer
        The layer to be cloned
    
    Returns
    -------
        A cloned layer with the same weights and configuration as the original layer.
    
    '''
    config = layer.get_config()
    weights = layer.get_weights()
    cloned_layer = type(layer).from_config(config)
    cloned_layer.build(layer.input_shape)
    cloned_layer.set_weights(weights)
    cloned_layer.trainable=False
    return cloned_layer

def resnet_layers(layer_names,input_shape):
    '''> We take a ResNet50 model, remove the top layers, and then clone the remaining layers
    
    Parameters
    ----------
    layer_names
        a list of strings, each of which is the name of a layer in the ResNet50 model.
    input_shape
        The shape of the input image.
    
    Returns
    -------
        A model with the input and output layers specified.
    
    '''
    resnet=tf.keras.applications.resnet50.ResNet50(include_top=False,input_shape=input_shape)
    resnet.trainable=False
    inputs=tf.keras.Input(shape=input_shape)
    identity=Identity(name="identity")
    preproc=ResnetPreProcess(name="preprocess")

    layers=[inputs,identity,preproc]
    for layer in resnet.layers[1:]:
        layers.append(clone_layer(layer))

    outputs = [resnet.get_layer(name).output for name in layer_names ]

    res= tf.keras.Model(resnet.input, outputs)
    x=identity(inputs)
    x=preproc(x)
    x=res(x)

    return tf.keras.Model(inputs,x)



def vgg_layers(layer_names,input_shape):
    '''> It takes a list of layer names and an input shape, and returns a model that outputs the
    activations of those layers
    
    Parameters
    ----------
    layer_names
        The names of the layers we want to extract from the VGG19 model.
    input_shape
        The shape of the input image.
    
    Returns
    -------
        A model with the input and output layers specified.
    
    '''
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet',input_shape=input_shape)
    vgg.trainable = False
    inputs=tf.keras.Input(shape=input_shape)
    identity=Identity(name="identity")
    preproc=VGGPreProcess(name="prepocess")

    layers=[inputs,identity,preproc]
    for layer in vgg.layers[1:]:
        layers.append(clone_layer(layer))

    _model= tf.keras.Sequential(layers)

    outputs = [_model.get_layer(name).output for name in layer_names ]

    return tf.keras.Model(_model.input, outputs)

if __name__ =="__main__":
    def test_resnet():
        r_layers=resnet_layers(["conv2_block2_out","conv5_block1_out"],(64,64,3))
        img=tf.random.normal((1,64,64,3))
        result=r_layers(img)
        print([r.shape for r in result])

    test_resnet()