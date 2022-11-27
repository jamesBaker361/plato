import tensorflow as tf
from numpy import log2

normal_init=tf.keras.initializers.RandomNormal(mean=0., stddev=0.01)

class CVAE(tf.keras.Model):
    """Convolutional variational autoencoder."""

    def __init__(self, latent_dim,max_dim,*args,**kwargs):
        super(CVAE, self).__init__(*args,**kwargs)
        self.latent_dim = latent_dim
        encoder_filters=[min(2** (i+5),512) for i in range(int(log2(max_dim))-1)]
        self.encoder = tf.keras.Sequential(
                [
                        tf.keras.layers.InputLayer(input_shape=(max_dim, max_dim, 3)),
                        *[tf.keras.layers.Conv2D(
                                filters=f, kernel_size=5, strides=2, padding='same',
                                activation='relu') for f in encoder_filters]
                        ,
                        tf.keras.layers.Flatten(),
                        # No activation
                        tf.keras.layers.Dense(latent_dim + latent_dim),
                ]
        )

        decoder_filters=encoder_filters[::-1]
        self.decoder = tf.keras.Sequential(
                [
                        tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                        tf.keras.layers.Dense(units=4*decoder_filters[0], activation=tf.nn.relu),
                        tf.keras.layers.Reshape(target_shape=(2, 2, decoder_filters[0])),
                        *[tf.keras.layers.Conv2DTranspose(
                                filters=f, kernel_size=5, strides=2, padding='same',
                                activation='relu') for f in decoder_filters]
                        ,
                        # No activation
                        tf.keras.layers.Conv2DTranspose(
                                filters=3, kernel_size=3, strides=1, padding='same'),
                ]
        )

    @tf.function
    def sample(self,eps=None,apply_sigmoid=False):
        '''The sample function takes in a random noise vector and returns a generated image
        
        Parameters
        ----------
        eps
            a random normal tensor.
        apply_sigmoid
            Boolean, whether to apply the sigmoid activation function after the decoder network.
        
        Returns
        -------
            The decoder is being returned.
        
        '''
        batch=100
        if eps is None:
            eps = tf.random.normal(shape=(batch, self.latent_dim))
        return self.decode(eps, apply_sigmoid)

    def encode(self, x):
        '''The encoder takes in an input, and returns the mean and log variance of the latent distribution
        
        Parameters
        ----------
        x
            the input data
        
        Returns
        -------
            The mean and log variance of the encoder
        
        '''
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        '''The encoder function takes in an image and returns the mean and log of the variance of the latent
        space. The decoder function takes in a latent vector and returns the logits of the reconstruction
        
        Parameters
        ----------
        z
            the latent space representation of the input image
        apply_sigmoid, optional
            If true, the output of the decoder is a sigmoid function.
        
        Returns
        -------
            The logits of the reconstruction.
        
        '''
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def call(self,inputs):
        '''> The encoder takes in an input, and returns a mean and log variance. The reparameterization trick
        is then used to sample from the distribution, and the decoder is used to reconstruct the input
        
        Parameters
        ----------
        inputs
            the input data
        
        Returns
        -------
            The output of the decoder.
        
        '''
        m,l=self.encode(inputs)
        z=self.reparameterize(m,l)
        return self.decode(z)

if __name__ =="__main__":
    model=CVAE(32,64)
    #model(tf.random.uniform((1,64,64,3)))
    #model.encoder.summary()
    model.decoder.summary()