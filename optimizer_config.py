import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
from tensorflow.keras.optimizers import SGD,Adam,RMSprop

#https://www.avanwyk.com/tensorflow-2-super-convergence-with-the-1cycle-policy/

def falling_cos(steps,base=0,amplitude=1):
    cos= [1+np.cos(s*np.pi/steps) for s in range(steps)]
    return [amplitude*c + base for c in cos]

def rising_cos(steps,base=0,amplitude=1):
    cos= [1-np.cos(s*np.pi/steps) for s in range(steps)]
    return [amplitude*c + base for c in cos]

class OptimizerWrapper:
    def __init__(self,_optimizer):
        self._optimizer=_optimizer
        self.steps=0
        self.epoch=0

    def apply_gradients(self,grads_and_vars):
        self.steps+=1
        self._optimizer.apply_gradients(grads_and_vars)

    def get_lr(self):
        try:
            return tf.keras.backend.get_value(self._optimizer.lr)
        except AttributeError:
            return None
        
    def get_momentum(self):
        try:
            return tf.keras.backend.get_value(self._optimizer.momentum)
        except AttributeError:
            return None
        
    def set_lr(self, lr):
        try:
            tf.keras.backend.set_value(self._optimizer.lr, lr)
        except AttributeError:
            pass # ignore
        
    def set_momentum(self, mom):
        try:
            tf.keras.backend.set_value(self._optimizer.momentum, mom)
        except AttributeError:
            pass # ignore

class OptimizerWrapperDecay(OptimizerWrapper):
    def __init__(self,_optimizer,decay_rate,cycle_steps):
        super().__init__(_optimizer)
        self.decay_rate=decay_rate
        self.cycle_steps=cycle_steps

    def apply_gradients(self, grads_and_vars):
        super().apply_gradients(grads_and_vars)
        if self.steps % self.cycle_steps==0:
            lr=self.get_lr()
            lr*=self.decay_rate
            self.set_lr(lr)

class OptimizerWrapperTriangular(OptimizerWrapper):
    def __init__(self,_optimizer,init_lr,max_lr,cycle_steps):
        super().__init__(_optimizer)
        self.init_lr=init_lr
        self.max_lr=max_lr
        half_cycle=cycle_steps//2
        change_per_step=(self.max_lr-self.init_lr)/half_cycle
        rising_schedule=[init_lr+ (j * change_per_step) for j in range(half_cycle)]
        falling_schedule=[max_lr-(j*change_per_step) for j in range(half_cycle)]
        self.lr_schedule=rising_schedule+falling_schedule

    def apply_gradients(self, grads_and_vars):
        super().apply_gradients(grads_and_vars)
        lr=self.lr_schedule[self.steps % len(self.lr_schedule)]
        self.set_lr(lr)

class OptimizerWrapperSuper(OptimizerWrapper):
    def __init__(self, _optimizer,init_lr,max_lr,cycle_steps,phase_one_pct,min_mom,max_mom,):
        super().__init__(_optimizer)
        phase_one_length=int(phase_one_pct*cycle_steps)
        phase_two_length=cycle_steps-phase_one_length
        lr_amplitude=max_lr-init_lr
        lr_amplitude=lr_amplitude/2
        mom_amplitude=max_mom-min_mom
        mom_amplitude=mom_amplitude/2
        self.lr_schedule=rising_cos(phase_one_length,init_lr,lr_amplitude)+falling_cos(phase_two_length,init_lr,lr_amplitude)
        self.mom_schedule=falling_cos(phase_one_length,min_mom,mom_amplitude)+rising_cos(phase_two_length,min_mom,mom_amplitude)

    def apply_gradients(self, grads_and_vars):
        super().apply_gradients(grads_and_vars)
        lr=self.lr_schedule[self.steps % len(self.lr_schedule)]
        self.set_lr(lr)
        mom=self.mom_schedule[self.steps % len(self.mom_schedule)]
        self.set_momentum(mom)





def get_optimizer(opt_name,opt_type,init_lr,max_lr,min_mom,max_mom,decay_rate,cycle_steps,phase_one_pct,clipnorm):
    if opt_name in set(['adam','Adam','ADAM']):
        _opt=Adam(init_lr,clipnorm=clipnorm)
    elif opt_name in set(['rms','RMS','Rms']):
        _opt=RMSprop(init_lr,momentum=max_mom,clipnorm=clipnorm)
    elif opt_name in set(['sgd','SGD']):
        _opt=SGD(init_lr,max_mom,True,clipnorm=clipnorm)
    else:
        raise Exception("opt_name isnt in (adam,rms,sgd)")
    if opt_type in set(["vanilla"]):
        return _opt
    elif opt_type in set(['decay','Decay']):
        return OptimizerWrapperDecay(_opt,decay_rate,cycle_steps)
    elif opt_type in set(['triangular','cyclical']):
        return OptimizerWrapperTriangular(_opt,init_lr,max_lr,cycle_steps)
    elif opt_type in set(["super","superconvergence","cosine"]):
        return OptimizerWrapperSuper(_opt,init_lr,max_lr,cycle_steps,phase_one_pct,min_mom,max_mom)
    else:
        raise Exception("opt_type {} opt_name {}".format(opt_type,opt_name))
    

if __name__ =='__main__':
    

    def data_set(batches,batch_size):
        b=0
        while b<batches:
            b+=1
            yield tf.random.normal((batch_size,28,28,1)),tf.random.uniform((batch_size,10))


    def test(optimizer,batches,batch_size,epochs=5):
        #x_train=x_train[:limit]
        model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(10,activation="sigmoid")])
        for epoch in range(epochs):
            for batch,labels in data_set(batches,batch_size):
                with tf.GradientTape() as tape:
                    logits=model(batch)
                    loss=tf.nn.sigmoid_cross_entropy_with_logits(labels,logits)
                gradients=tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            print('finished epoch {} lr ={}'.format(epoch,optimizer.get_lr()))

    '''optimizer_decay=OptimizerWrapperDecay(SGD(),0.9,8)
    test(optimizer_decay,20,2)
    optimizer_triangle=OptimizerWrapperTriangular(SGD(),0.1,0.2,10)
    print("we should see every other")
    test(optimizer_triangle,5,2)
    optimizer_triangle=OptimizerWrapperTriangular(SGD(),0.1,0.2,10)
    print("we should see cyclical")
    test(optimizer_triangle,10,2)
    optimizer_super=OptimizerWrapperSuper(SGD(),0.1,0.2,10,0.4,0.1,0.2)
    print("should be nice cycle")
    test(optimizer_super,10,2)
    print("should be weird asf")
    optimizer_super=OptimizerWrapperSuper(SGD(),0.1,0.2,100,0.4,0.1,0.2)
    test(optimizer_super,40,2)'''