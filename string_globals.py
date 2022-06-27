
import os
npz_root='../../../../../scratch/jlb638/plato/imgs_npz' #where the npz matrices are; CHANGE THIS FOR UR OWN USES
img_dir='../../../../../scratch/jlb638/plato/images' #where the actual images themselves are; CHANGE THIS
mnist_dir='../../../../../scratch/jlb638/trainingSet/trainingSet'
mnist_npz_root='../../../../../scratch/jlb638/mnist'
checkpoint_dir='../../../../../scratch/jlb638/plato/checkpoints' #where saved models are stored CHANGE THIS FOR YOUR OWN PURPOSES
all_styles=[s for s in os.listdir('{}'.format(img_dir)) if s[0]!='.']
all_styles_npz=[npz_root+"/"+s for s in all_styles]
all_digits=[str(i) for i in range(10)]
all_digits_npz=[mnist_npz_root+"/"+d for d in all_digits]
gen_img_dir='./gen_imgs' #generated images
err_dir="./slurm/err"
out_dir="./slurm/out"

#raw_image_dir='/scratch/jlb638/images/
for d in [checkpoint_dir,npz_root,gen_img_dir,err_dir,out_dir,mnist_npz_root,mnist_dir]+all_styles_npz+all_digits_npz:
    if not os.path.exists(d):
        os.makedirs(d)