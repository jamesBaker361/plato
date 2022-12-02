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
from optimizer_config import *
from data_helper import *
from generate_img_helpers import *
from cvae import *


class MultiHeadSelfAttention(tf.keras.layers.Layer):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = Dense(embed_dim,name='query')
        self.key_dense = Dense(embed_dim,name='key')
        self.value_dense = Dense(embed_dim,name='value')
        self.combine_heads = Dense(embed_dim,name='combine')

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
    def __init__(self, embed_dim, num_heads, mlp_dim):
        dropout=0.1
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
        self.patch_size = patch_size
        num_patches_sqrt=max_dim // patch_size
        num_patches = num_patches_sqrt ** 2
        self.patch_dim = channels * patch_size ** 2

        self.num_layers = num_layers

        super(AttentionVAE,self).__init__(latent_dim, max_dim, *args, **kwargs)
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
    strategy = tf.distribute.MirroredStrategy()

    class Args(object):
        def __init__(self, _dict):
            self.__dict__.update(_dict)

    def test1():
        models=[AttentionVAE(16,4,8,8,32,64,strategy),AttentionVAE(8,4,8,8,32,64)]

        v=tf.random.uniform((10,64,64,3))

        models[0](v)
        for var in models[0].trainable_variables:
            print(var.name)

        for m in models:
            print(v.shape)
            print(m(v).shape)
            print(m.patch_size)

    
    def test2():
        with strategy.scope():
            model=AttentionVAE(16,4,8,8,32,64)
            optimizer=tf.keras.optimizers.Adam()
            dataset=tf.data.Dataset.from_tensor_slices([tf.random.uniform((64,64,3)) for _ in range(4)]).batch(1)
            dataset=strategy.experimental_distribute_dataset(dataset)

            def compute_loss(x):
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,False)
                cross_ent=(x-x_logit)**2

                return tf.nn.compute_average_loss(cross_ent, global_batch_size=1)

        def train_step(x):
            with tf.GradientTape() as tape:
                loss = compute_loss(x)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(x):
            per_replica_losses = strategy.run(train_step, args=(x,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

        for x in dataset:
            distributed_train_step(x)

    def test3():
        global_batch_size=1
        num_examples_to_generate=1
        args=Args({
            'c3vae':False,
            "oversample":False,
            "dataset":"faces2",
            "batch_size":1,
            "genres":["ukiyo-e"],"max_dim":64,"use_smote":False,
            "quantity":10
        })
        def print_debug(*a):
            print(*a)
        with strategy.scope():
            model=AttentionVAE(16,4,8,8,32,64)
            optimizer=tf.keras.optimizers.Adam()
            #dataset=tf.data.Dataset.from_tensor_slices([tf.random.uniform((64,64,3)) for _ in range(4)]).batch(global_batch_size)
            #dataset=strategy.experimental_distribute_dataset(dataset)
            train_dataset,test_dataset,validate_dataset,test_sample,validate_sample=get_data_loaders(args,global_batch_size,print_debug,num_examples_to_generate,strategy)
            def compute_loss(x):
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,False)
                cross_ent=(x-x_logit)**2

                return tf.nn.compute_average_loss(cross_ent, global_batch_size=global_batch_size)

        def train_step(x):
            with tf.GradientTape() as tape:
                loss = compute_loss(x)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(x):
            per_replica_losses = strategy.run(train_step, args=(x,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

        for x in train_dataset:
            distributed_train_step(x)


    def test5():
        global_batch_size=1
        num_examples_to_generate=1
        args=Args({
            'c3vae':False,
            "oversample":False,
            "dataset":"faces2",
            "batch_size":1,
            "genres":["ukiyo-e"],"max_dim":64,"use_smote":False,
            "quantity":10,
            "kl_weight":1.0
        })
        def print_debug(*a):
            print(*a)
        with strategy.scope():
            model=AttentionVAE(16,4,8,8,32,64)
            #optimizer=tf.keras.optimizers.Adam()
            optimizer=get_optimizer('adam','vanilla',0.01,0.01,0.85,0.95,0.4,1000,0.9,1.0)
            #dataset=tf.data.Dataset.from_tensor_slices([tf.random.uniform((64,64,3)) for _ in range(4)]).batch(global_batch_size)
            #dataset=strategy.experimental_distribute_dataset(dataset)
            train_dataset,test_dataset,validate_dataset,test_sample,validate_sample=get_data_loaders(args,global_batch_size,print_debug,num_examples_to_generate,strategy)
            def compute_loss(x):
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,False)
                cross_ent=(x-x_logit)**2
                reconst = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                kl= -0.5* args.kl_weight * tf.reduce_sum(1+logvar - mean**2 - tf.exp(logvar),axis=1)

                per_example_loss= reconst+kl

                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        def train_step(x):
            with tf.GradientTape() as tape:
                loss = compute_loss(x)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(x):
            per_replica_losses = strategy.run(train_step, args=(x,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

        for x in train_dataset:
            distributed_train_step(x)


    def test5():
        global_batch_size=2
        num_examples_to_generate=2
        args=Args({
            'c3vae':False,
            "oversample":False,
            "dataset":"faces2",
            "batch_size":2,
            "genres":["ukiyo-e"],
            "max_dim":64,
            "use_smote":False,
            "quantity":20,
            "kl_weight":1.0
        })
        def print_debug(*a):
            print(*a)
        with strategy.scope():
            model=AttentionVAE(16,4,8,8,32,64)
            #optimizer=tf.keras.optimizers.Adam()
            optimizer=get_optimizer('adam','vanilla',0.01,0.01,0.85,0.95,0.4,1000,0.9,1.0)
            #dataset=tf.data.Dataset.from_tensor_slices([tf.random.uniform((64,64,3)) for _ in range(4)]).batch(global_batch_size)
            #dataset=strategy.experimental_distribute_dataset(dataset)
            train_dataset,test_dataset,validate_dataset,test_sample,validate_sample=get_data_loaders(args,global_batch_size,print_debug,num_examples_to_generate,strategy)
            def compute_loss(x):
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,False)
                cross_ent=(x-x_logit)**2
                reconst = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                kl= -0.5* args.kl_weight * tf.reduce_sum(1+logvar - mean**2 - tf.exp(logvar),axis=1)

                per_example_loss= reconst+kl

                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)

        
        def train_step(x):
            with tf.GradientTape() as tape:
                loss = compute_loss(x)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(x):
            per_replica_losses = strategy.run(train_step, args=(x,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

        for epoch in range(2):
            for x in train_dataset:
                distributed_train_step(x)

    def test6():
        num_examples_to_generate=1
        args=Args({
            'c3vae':False,
            "oversample":False,
            "dataset":"faces2",
            "batch_size":2,
            "genres":["ukiyo-e"],
            "max_dim":64,
            "use_smote":False,
            "quantity":20,
            "kl_weight":1.0,
            "begin_epoch":0,
            "latent_dim":32
        })
        logical_gpus = tf.config.list_logical_devices('GPU')
        strategy = tf.distribute.MirroredStrategy()

        if len(logical_gpus)>0:
            global_batch_size=len(logical_gpus)*args.batch_size
        else:
            global_batch_size=args.batch_size
        def print_debug(*a):
            print(*a)

        random_vector_for_generation = tf.random.normal(
        shape=[num_examples_to_generate, args.latent_dim])
        with strategy.scope():
            model=AttentionVAE(16,4,8,8,32,64)
            #optimizer=tf.keras.optimizers.Adam()
            optimizer=get_optimizer('adam','vanilla',0.01,0.01,0.85,0.95,0.4,1000,0.9,1.0)
            #dataset=tf.data.Dataset.from_tensor_slices([tf.random.uniform((64,64,3)) for _ in range(4)]).batch(global_batch_size)
            #dataset=strategy.experimental_distribute_dataset(dataset)
            train_dataset,test_dataset,validate_dataset,test_sample,validate_sample=get_data_loaders(args,global_batch_size,print_debug,num_examples_to_generate,strategy)
            def compute_loss(x):
                mean, logvar = model.encode(x)
                z = model.reparameterize(mean, logvar)
                x_logit = model.decode(z,False)
                cross_ent=(x-x_logit)**2
                reconst = tf.reduce_sum(cross_ent, axis=[1, 2, 3])
                kl= -0.5* args.kl_weight * tf.reduce_sum(1+logvar - mean**2 - tf.exp(logvar),axis=1)

                per_example_loss= reconst+kl

                return tf.nn.compute_average_loss(per_example_loss, global_batch_size=global_batch_size)


        def train_step(x):
            with tf.GradientTape() as tape:
                loss = compute_loss(x)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            return loss

        @tf.function
        def distributed_train_step(x):
            per_replica_losses = strategy.run(train_step, args=(x,))
            return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses,
                         axis=None)

        generate_and_save_images=get_generate_and_save_images(print_debug,args)
        generate_from_noise=get_generate_from_noise(args,num_examples_to_generate)

        generate_and_save_images(model, args.begin_epoch, test_sample,False)
        generate_from_noise(model,args.begin_epoch,random_vector_for_generation,False)

        for epoch in range(2):
            for x in train_dataset:
                distributed_train_step(x)


    test5()
    print("done :)")

