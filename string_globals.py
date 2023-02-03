
import os
import json
npz_root='../../../../../scratch/jlb638/plato/imgs_npz' #where the npz matrices are; CHANGE THIS FOR UR OWN USES
img_dir='../../../../../scratch/jlb638/plato/images' #where the actual images themselves are; CHANGE THIS
mnist_dir='../../../../../scratch/jlb638/trainingSet/trainingSet'
mnist_npz_root='../../../../../scratch/jlb638/mnist'
faces_dir='../../../../../scratch/jlb638/artfaces'
faces_npz_dir='../../../../../scratch/jlb638/npz_artfaces'
faces_npz_dir_2='../../../../../scratch/jlb638/npz_artfaces2/npz_artfaces'
bird_npz_dir='../../../../../scratch/jlb638/birds'
checkpoint_dir='../../../../../scratch/jlb638/plato/checkpoints' #where saved models are stored CHANGE THIS FOR YOUR OWN PURPOSES
all_styles=[s for s in os.listdir('{}'.format(img_dir)) if s[0]!='.']
all_styles_npz=[npz_root+"/"+s for s in all_styles] #marina, figurative, portrait,etc
all_digits=[str(i) for i in range(10)]
all_digits_npz=[mnist_npz_root+"/"+d for d in all_digits]
all_styles_faces=[s for s in os.listdir(faces_dir) if s[0]!='.' if len(os.listdir(os.path.join(faces_dir,s))) >100]
all_styles_faces_smote=[s for s in os.listdir(faces_dir) if s[0]!='.' if len(os.listdir(os.path.join(faces_dir,s))) >250 and len(os.listdir(os.path.join(faces_dir,s)))<1000]
all_styles_faces_npz=[faces_npz_dir+"/"+s for s in all_styles_faces]
all_styles_faces_2=[s for s in os.listdir(faces_npz_dir_2) if s[0]!='.' and len(os.listdir(os.path.join(faces_npz_dir_2,s))) >=100]
all_styles_weeds=[i for i in range(9)]
#smote_styles_faces_2=[s for s in os.listdir(faces_npz_dir_2) if s[0]!='.' and len(os.listdir(os.path.join(faces_npz_dir_2,s))) >250 and len(os.listdir(os.path.join(faces_npz_dir_2,s))) <2500]

all_styles_faces_3=json.load(open("all_styles_faces.json"))['styles']

all_styles_birds=[s for s in os.listdir(bird_npz_dir)if s[0]!='.']
smote_sample="SMOTE_SAMPLE"
gen_img_dir='./gen_imgs' #generated images
err_dir="./slurm/err"
out_dir="./slurm/out"

root_dict={
    "birds":bird_npz_dir,
    "mnist":mnist_npz_root,
    "art":npz_root,
    "faces":faces_npz_dir,
    "faces2":faces_npz_dir_2,
    "deep_weeds":'deep_weeds',
    "faces3":"faces3"
}

dataset_default_all_styles={
    "birds":all_styles_birds,
    "faces": all_styles_faces,
    "art": all_styles,
    "mnist":all_digits,
    "faces2":all_styles_faces_2,
    "deep_weeds":all_styles_weeds,
    "faces3":all_styles_faces_3
}

style_quantity_dicts={
    "faces2":{s:5000 for s in all_styles_faces_2},
    "deep_weeds": {s:5000 for s in all_styles_weeds},
    "faces3":{s:5000 for s in all_styles_faces_3}
}

#raw_image_dir='/scratch/jlb638/images/
for d in [checkpoint_dir,npz_root,gen_img_dir,err_dir,out_dir,mnist_npz_root,mnist_dir,faces_dir,faces_npz_dir]+all_styles_npz+all_digits_npz+all_styles_faces_npz:
    if not os.path.exists(d):
        os.makedirs(d)