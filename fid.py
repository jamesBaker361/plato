import numpy
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy import asarray
from numpy.random import randint
from scipy.linalg import sqrtm
import tensorflow as tf
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input

 
#https://machinelearningmastery.com/how-to-implement-the-frechet-inception-distance-fid-from-scratch/

# calculate frechet inception distance
def calculate_fid(images1, images2, input_shape):
    if input_shape[0]<128:
        images1=tf.image.resize_with_pad(images1,128,128,"nearest")
        images2=tf.image.resize_with_pad(images2,128,128,"nearest")
        input_shape=(128,128,3)
    model = InceptionV3(include_top=False, pooling='avg', input_shape=input_shape)
    # pre-process images
    images1 = preprocess_input(255*images1)
    images2 = preprocess_input(255*images2)
    # calculate activations
    act1 = model.predict(images1)
    act2 = model.predict(images2)
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
    mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
    # calculate sum squared difference between means
    ssdiff = numpy.sum((mu1 - mu2)**2.0)
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
    # calculate score
    fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid

if __name__=="__main__":

    from data_loader import *

    loader=get_loader(64,["baroque","expressionism"],100,faces_npz_dir).batch(100)
    loader2=get_loader(64,["baroque","high-renaissance"],100,faces_npz_dir).batch(100)
    for i in loader:
        imgs=i
        break
    for i in loader2:
        imgs2=i
        break

    def test_calc():
        f=calculate_fid(imgs,imgs2,(64,64,3))
        print(f)

    test_calc()