import tensorflow as tf
from tensorflow.keras.layers import Dense, Reshape, Conv2DTranspose
from tensorflow.keras import Model, Input
from random import randrange
from numpy import log2
from cvae import *



def adain(content, style, epsilon=1e-5): #stolen from https://github.com/ftokarev/tf-adain/blob/master/adain/norm.py
    axes = [2,1]

    batch_size=content.shape[0]
    #print('content.shape',content.shape)
    if batch_size is None:
        try:
            batch_size=tf.shape(content)[0]
            #print('tf.shape(content)',tf.shape(content))
        except:
            try:
                batch_size=len(content)
            except:
                print("couldnt figure out batch size")
                pass
    #print('style.shape',style.shape)
    #print('tf.shape(style)',tf.shape(style))
    c_mean, c_var = tf.nn.moments(content, axes=axes, keepdims=True)
    s_mean, s_var = tf.nn.moments(style,axes=[1] ,keepdims=True)
    s_mean=tf.reshape(s_mean,[-1,1,1,1])
    s_var=tf.reshape(s_var,[-1,1,1,1])

    c_std= tf.sqrt(c_var + epsilon)
    s_std =tf.sqrt(s_var + epsilon)

    return s_std * ((content - c_mean) / c_std) + s_mean

class C3VAE(CVAE): #class conditioned convolutional VAE
    def __init__(self, n_classes,class_latent_dim,latent_dim,max_dim,*args,**kwargs):
        super(C3VAE,self).__init__(latent_dim,max_dim,*args,**kwargs)
        self.class_latent_dim=class_latent_dim
        self.n_classes=n_classes
        encoder_filters=[min(2** (i+5),512) for i in range(int(log2(max_dim))-1)]
        decoder_filters=encoder_filters[::-1]
        class_inputs=Input(shape=n_classes)
        class_noise=Dense(class_latent_dim)(class_inputs)
        decoder_input=Input(shape=(latent_dim,))
        total_feature_space=4*decoder_filters[0]
        img=Dense(units=total_feature_space, activation=tf.nn.relu)(decoder_input)
        img=Reshape(target_shape=(2, 2, decoder_filters[0]))(img)
        class_noise_layers=[Dense(class_latent_dim//4,activation='relu')(class_noise)]
        for d,f in enumerate(decoder_filters):
            img=Conv2DTranspose(
                                filters=f, kernel_size=5, strides=2, padding='same',
                                activation='relu')(img)
            shape=img.shape[1:]
            img=adain(img,class_noise)
        img=tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')(img)
        self.decoder=Model(inputs=[decoder_input,class_inputs],outputs=img)

    @tf.function
    def sample(self,eps,apply_sigmoid,class_label=None):
        batch=100
        if eps is None:
            eps = tf.random.normal(shape=(batch, self.latent_dim))
        else:
            try:
                batch=eps.shape[0]
            except:
                try:
                    batch = tf.shape(eps)[0]
                except:
                    try:
                        batch = len(eps)
                    except:
                        pass
        if class_label is None:
            indices=[randrange(self.n_classes) for _ in range(batch)]
            class_label=tf.one_hot(indices,self.n_classes)
        return self.decode(eps,class_label,apply_sigmoid)

    def decode(self, z,class_label, apply_sigmoid=False):
        logits=self.decoder([z,class_label])
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self, pair):
        [img,class_label]=pair
        m,l=self.encode(img)
        z=self.reparameterize(m,l)
        return self.decode(z,class_label)


if __name__=="__main__":
    model=C3VAE(15,32,32,64)
    #model.encoder.summary()
    model.sample(None,False,None)
    img=tf.random.uniform([1,64,64,3])
    label=tf.random.uniform([1,15])
    model(img,label)