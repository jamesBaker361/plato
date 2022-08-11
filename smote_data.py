from data_loader import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import NearestNeighbors
import random

def average_imgs(first,second,imgs,*args):
    #print(tf.shape(imgs[first]))
    return imgs[first]/2 + imgs[second]/2

def average_decoded(first,second,imgs,latent,model,*args):
    z =latent[first]/2 + latent[second]/2
    #print(tf.shape(z))
    return model.decode(tf.expand_dims(z,0))[0]

def average_mean_logs(first,second,imgs,latent,model,latent_mean_logs):
    [m1,l1]=latent_mean_logs[first]
    [m2,l2]=latent_mean_logs[second]
    mean =m1/2+m2/2
    logvar=l1/2+l2/2
    z=model.reparameterize(mean,logvar)
    return model.decode(z)[0]



def make_synthetic(style,max_dim,merge_function,model,total_size=2500,limit=100000):
    loader=get_loader(max_dim,[style],limit,faces_npz_dir,True).batch(1)
    imgs=[]
    latent=[]
    latent_mean_logs=[]
    for i in loader:
        imgs.append(i[0])
        mean, logvar=model.encode(i)
        z=model.reparameterize(mean,logvar)
        latent.append(z[0])
        latent_mean_logs.append([mean,logvar])

    neigh = NearestNeighbors(n_neighbors=5)
    neigh.fit(latent)
    dist,ind=neigh.kneighbors(n_neighbors=5)
    ret=[]
    n_samples=total_size-len(imgs)
    if n_samples<0:
        return ret
    for _ in range(n_samples):
        first=random.randint(0,len(imgs)-1)
        second=ind[first][random.randint(0,4)]
        ret.append(merge_function(first,second,imgs,latent,model,latent_mean_logs))
    return ret

def save16(images,name):
    fig = plt.figure(figsize=(4, 4))

    for i,img in enumerate(images):
        plt.subplot(4, 4, i + 1)
        plt.imshow(img)
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig(name)
    plt.show()

def test_smote(model,max_dim=64,style="pointillism"):
    synthetic_avg=make_synthetic(style,max_dim,average_imgs,model)
    synthetic_decoded=make_synthetic(style,max_dim,average_decoded,model)
    synthetic_mean_log=make_synthetic(style,max_dim,average_mean_logs,model)
    for i in range(0,64,16):
        save16(synthetic_avg[i:i+16],"./smote_test/{}-avg-{}.png".format(style,i))
        save16(synthetic_decoded[i:i+16],"./smote_test/{}-decoded-{}.png".format(style,i))
        save16(synthetic_mean_log[i:i+16],"./smote_test/{}-mean_log-{}.png".format(style,i))
    return synthetic_avg,synthetic_decoded,synthetic_mean_log

def make_smote(model,max_dim,style,total_size):
    synthetic_avg=make_synthetic(style,max_dim,average_imgs,model,total_size)
    return synthetic_avg


if __name__=="__main__":
    ret=[]