from c3vae import *
from tensorflow.keras.layers import Dropout, Conv2D, LeakyReLU, Layer

class LPCS(Layer): #learned per-channel scaling factors to the noise input
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self,input_shape):
        self.scaling_factor=self.add_weight('scaling_factor',shape=[int(input_shape[-1])])

    def call(self, inputs):
        return self.scaling_factor * inputs

class Affine(Layer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, input_shape):
        self.affine_weights=self.add_weight('affine_weights',shape=[int(input_shape[-1])])
        self.affine_bias=self.add_weight('affine_bias',shape=[int(input_shape[-1])])

    def call(self, inputs):
        w=self.affine_weights * inputs
        return w + self.affine_bias

class Constant(Layer):
    def __init__(self,shape=[4,4,512], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shape=shape

    def build(self, input_shape):
        self._constant=self.add_weight("constant",shape=self.shape)

    def call(self, inputs):
        return self._constant

class Noise(Layer):
    def __init__(self,width, block_channels, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.width=width
        self.block_channels=block_channels

    def build(self, input_shape):
        pass

    def call(self, inputs, *args, **kwargs):
        return tf.random.normal([self.width,self.width,self.block_channels])

class StyleVAE(CVAE):
    def __init__(self,fc_layers, fc_dim, dropout_rate, fc_activation, latent_dim, max_dim, *args, **kwargs):
        super().__init__(latent_dim, max_dim, *args, **kwargs)
        decoder_input=Input(shape=(latent_dim,))
        w=decoder_input
        for _ in range(fc_layers):
            w=Dense(fc_dim, activation=fc_activation)(w)
            w=Dropout(dropout_rate)(w)
        #constant = tf.Variable(tf.ones([4,4,512]))
        img=Constant()(None)
        def block(img,block_channels,width):
            noise_0=Noise(width,block_channels)(None)
            #img=img+(tf.Variable(tf.random.normal([block_channels])) * noise_0) #learned per-channel scaling factors to the noise input
            img=img+LPCS()(noise_0)
            #affine_0 =tf.Variable(tf.random.normal([fc_dim]),name='affine_weights')*w + tf.Variable(tf.random.normal([fc_dim]), name='affine_bias') #learned affine transform
            img=adain(img,Affine()(w))
            img=LeakyReLU()(img)
            img =Conv2D(block_channels,3,padding='same')(img)
            noise_1=Noise(width,block_channels)(None)
            #img=img+(tf.Variable(tf.random.normal([block_channels])) * noise_1) #learned per-channel scaling factors to the noise input
            img=img+LPCS()(noise_1)
            #affine_1=tf.Variable(tf.random.normal([fc_dim]),name='affine_weights')*w + tf.Variable(tf.random.normal([fc_dim]), name='affine_bias') #learned affine transform
            return LeakyReLU()(adain(img,Affine()(w)))
        current_width=4
        current_channels=512
        img=block(img,current_channels,current_width)
        while current_width < max_dim:
            current_width *=2
            current_channels=current_channels//2
            img=Conv2DTranspose(current_channels,5,(2,2),padding='same')(img)
            img =Conv2D(current_channels,3,padding='same')(img)
            img=block(img,current_channels,current_width)
        img=tf.keras.layers.Conv2DTranspose(filters=3, kernel_size=3, strides=1, padding='same')(img)
        self.decoder=Model(inputs=decoder_input,outputs=img)


if __name__=='__main__':
    def test0():
        for dim in [64,128,256]:
            model=StyleVAE(64,4,16,0.1,'sigmoid',32,dim)
        print('test0 :)')

    def test1():
        batch_size=2
        for dim in [64,128,256]:
            model=StyleVAE(batch_size,4,16,0.1,'sigmoid',32,dim)
            noise=tf.random.normal([batch_size,dim,dim,3])
            print("shape =", tf.shape(model(noise)))
        print('test 1 :)')

    def test2():
        pass

    test1()



        

