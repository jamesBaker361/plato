from string_globals import *
import matplotlib.pyplot as plt

def generate_and_save_images(model, epoch, test_sample,apply_sigmoid):
    mean, logvar = model.encode(test_sample)
    z = model.reparameterize(mean, logvar)
    print("z shape ",z.shape)
    predictions = model.sample(z,apply_sigmoid)
    print("real prediction shape",predictions.shape)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('{}/{}/image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
    plt.show()

def generate_from_noise(model,epoch,random_vector,apply_sigmoid):
    predictions = model.sample(random_vector,apply_sigmoid)
    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i])
        plt.axis('off')

    # tight_layout minimizes the overlap between 2 sub-plots
    plt.savefig('{}/{}/gen_image_at_epoch_{:04d}.png'.format(gen_img_dir,args.name,epoch))
    plt.show()