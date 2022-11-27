#thanks to https://github.com/emla2805/vision-transformer/blob/master/model.py

import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    LayerNormalization,
    Reshape,
    Conv2DTranspose,
    Add,
    Flatten
)
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from cvae import *


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim)
        self.key_dense = Dense(embed_dim)
        self.value_dense = Dense(embed_dim)
        self.combine_heads = Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)

        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(
            x, (batch_size, -1, self.num_heads, self.projection_dim)
        )
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)

        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.mlp = tf.keras.Sequential(
            [
                Dense(mlp_dim, activation=tfa.activations.gelu),
                Dropout(dropout),
                Dense(embed_dim),
                Dropout(dropout),
            ]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs, training):
        inputs_norm = self.layernorm1(inputs)
        attn_output = self.att(inputs_norm)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = attn_output + inputs

        out1_norm = self.layernorm2(out1)
        mlp_output = self.mlp(out1_norm)
        mlp_output = self.dropout2(mlp_output, training=training)
        return mlp_output + out1


class AttentionVAE(CVAE):
    def __init__(self, patch_size, num_layers ,d_encoder,d_decoder, latent_dim, max_dim, *args, **kwargs):
        channels=3
        num_heads=4
        mlp_dim=8
        super().__init__(latent_dim, max_dim, *args, **kwargs)
        self.patch_size = patch_size
        num_patches_sqrt=max_dim // patch_size
        num_patches = num_patches_sqrt ** 2
        self.patch_dim = channels * patch_size ** 2

        self.num_layers = num_layers

        self.pos_emb = self.add_weight(
            "pos_emb", shape=(1, num_patches, d_encoder)
        )

        self.pos_emb_dec= self.add_weight(
            "pos_emb_dec", shape=(1, num_patches, d_decoder)
        )

        self.enc_layers = [
            TransformerBlock(d_encoder, num_heads, mlp_dim)
            for _ in range(num_layers)
        ]

        self.dec_layers=[ TransformerBlock(d_decoder, num_heads, mlp_dim) for _ in range(num_layers)]

        self.patch_proj = Dense(d_encoder)
        self.patch_proj_dec=Dense(d_decoder)

        self.encoder = tf.keras.Sequential(
            [
                *self.enc_layers,
                LayerNormalization(epsilon=1e-6),
                Dense(d_encoder, activation=tfa.activations.gelu),
                Flatten(),
                Dense(latent_dim*2)
            ]
        )

        self.pre_decoder=tf.keras.Sequential([
            Dense(num_patches*self.patch_dim),
            Reshape((num_patches, self.patch_dim))
        ])
        self.decoder = tf.keras.Sequential([
            *self.dec_layers,
            LayerNormalization(epsilon=1e-6),
            Reshape((num_patches_sqrt,num_patches_sqrt,d_decoder)),
            Conv2DTranspose(d_decoder,(patch_size,patch_size),strides=(patch_size,patch_size),padding='same'),
            #Conv2DTranspose(d_decoder,(2,2),strides=(patch_size //2,patch_size //2),padding='same'),
            Dense(3)
        ])

    def extract_patches(self, images):
        try:
            batch_size=images.shape[0]
        except:
            try:
                batch_size = tf.shape(images)[0]
            except:
                try:
                    batch_size = len(images)
                except:
                    pass
        #print('batch_size',batch_size)
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        #print('pre-shaped',patches.shape)
        patches = tf.reshape(patches, [batch_size, -1, self.patch_dim])
        return patches

    def encode(self, x):
        x=self.extract_patches(x)
        #print('self.extract_patches(x)',x.shape)
        x=self.patch_proj(x)
        x=x+self.pos_emb
        x=self.encoder(x)
        #print('self.encoder(x)',x.shape)
        mean, logvar = tf.split(x, num_or_size_splits=2, axis=1)
        return mean, logvar

    def decode(self, z, apply_sigmoid=False):
        z=self.pre_decoder(z)
        z=self.patch_proj_dec(z)
        z=z+self.pos_emb_dec
        z=self.decoder(z)
        if apply_sigmoid:
            return tf.sigmoid(z)
        return z



if __name__ == '__main__':
    models=[AttentionVAE(16,4,8,8,32,64),AttentionVAE(8,4,8,8,32,64)]
    v=tf.random.uniform((10,64,64,3))
    for m in models:
        print(v.shape)
        print(m(v).shape)
        print(m.patch_size)