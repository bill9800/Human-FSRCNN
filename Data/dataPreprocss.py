import os
import cv2
from mtcnn.mtcnn import MTCNN
import numpy as np
from skimage.measure import compare_ssim as ssim
from skimage.measure import compare_psnr as psnr

# parameter
ORIGINAL_PATH = "HR_img"
STORE_PATH = "LR_img"
DOWN_SCALE_FACTOR = 0.25

def is_human_img(img,detector):
    # detect whether there is a face in this image
    faces = detector.detect_faces(img)
    if(len(faces)!=0):
        return True
    print("no human in the image")
    return False

def init_dir(store_path=STORE_PATH):
    if not os.path.isdir(store_path):
        os.mkdir(store_path)

def two_K_human_img_selection(original_dir,store_dir,begin_idx,detect_human=False):
    # select 2K images
    init_dir(store_dir)
    if detect_human:
        detector = MTCNN()
    files = [f for f in os.listdir(original_dir)]
    for file in files:
        path = original_dir + "/" + file # img path
        img = cv2.imread(path)
        height, width, channels = img.shape
        if height > 2000 or width > 2000:
            # check the img size
            if detect_human:
                if is_human_img(img,detector):
                    store_path = store_dir + "/" + str(begin_idx) + ".jpg"
                    cv2.imwrite(store_path,img)
                    begin_idx += 1
            else:
                store_path = store_dir + "/" + str(begin_idx) + ".jpg"
                cv2.imwrite(store_path, img)
                begin_idx += 1

def create_database(original_dir,store_dir,down_factor):
    init_dir(store_dir)
    size = len(os.listdir(original_dir))

    for i in range(size):
        path = original_dir + '/' + str(i) + '.jpg'
        img = cv2.imread(path)
        height, width, channels = img.shape
        img = cv2.resize(img,(int(down_factor*width),int(down_factor*height)),interpolation= cv2.INTER_CUBIC)
        store_path = store_dir + "/" + str(i) + ".jpg"
        cv2.imwrite(store_path,img)


def crop_with_scale(original_dir,store_dir,size_factor):
    # ensure the data is factors of size_factor
    init_dir(store_dir)
    #size = len(os.listdir(original_dir))
    for name in os.listdir(original_dir):
        path = original_dir + '/' + name
        img = cv2.imread(path)
        height, width, channels = img.shape
        new_h = int(height/size_factor)*size_factor
        new_w = int(width/size_factor)*size_factor
        img = img[:new_h,:new_w,:]
        store_path = store_dir + "/" + name
        cv2.imwrite(store_path, img)


def data_augment(img_dir,store_dir,flip=True,crop_aug=True):
    # could add more augmentation
    init_dir(store_dir)
    imgs = os.listdir(img_dir)
    for name in imgs:
        img = cv2.imread(img_dir+'/'+name)
        aug_imgs = []
        aug_imgs.append(img)
        # flip
        if flip:
            aug_imgs.append(np.fliplr(img))
        # crop part of the image
        if crop_aug:
            height = img.shape[0]
            width = img.shape[1]
            crop_1 = img[:int(height*0.9),:int(width*0.9)]
            crop_2 = img[int(height*0.1):,int(width*0.1):]
            aug_imgs.append(crop_1)
            aug_imgs.append(crop_2)
            if flip:
                # get flip img
                flip_img = aug_imgs[1]
                crop_3 = flip_img[:int(height * 0.9), :int(width * 0.9)]
                crop_4 = flip_img[int(height * 0.1):, int(width * 0.1):]
                aug_imgs.append(crop_3)
                aug_imgs.append(crop_4)
        # store aug_imgs
        prefix = name.split('.')[0]
        for i in range(len(aug_imgs)):
            aug_img = aug_imgs[i]
            store_path = store_dir + "/" + prefix + '_' + str(i) + ".jpg"
            cv2.imwrite(store_path,aug_img)


def face_crop(img_dir,store_dir):
    init_dir(store_dir)
    detector = MTCNN()
    imgs = os.listdir(img_dir)
    for name in imgs:
        img = cv2.imread(img_dir+'/'+name)
        try:
            faces = detector.detect_faces(img)
        except:
            print('do next face detection')
            continue
        print('detect img - ' + name)
        print('detected len:',len(faces))
        for i in range(len(faces)):
            face = faces[i]
            print(face)
            box = face['box']
            confidence = face['confidence']
            if confidence < 0.95:
                # threshold to get just high confidence image
                continue
            skip = False
            for par in box:
                # box is out of range
                if par < 0 :
                    skip = True
            if skip:
                continue
            crop_img = img[box[1]:box[1]+box[3],box[0]:box[0]+box[2]]
            prefix = name.split('.')[0]
            store_path = store_dir + "/" + prefix + '_' + str(i) + ".jpg"
            cv2.imwrite(store_path, crop_img)

def img_transform(img_dir,store_dir,size_factor,transform = 'bicubic'):
    # decrease the size of img by the factor and recover back using the transform method
    init_dir(store_dir)
    if transform == 'bicubic':
        imgs = os.listdir(img_dir)
        for name in imgs:
            img = cv2.imread(img_dir + '/' + name)
            width = int(img.shape[1] * size_factor)
            height = int(img.shape[0] * size_factor)
            dim = (width, height)
            trans_img = cv2.resize(img,dim,interpolation=cv2.INTER_CUBIC)
            original_dim = img.shape
            trans_img = cv2.resize(trans_img,(original_dim[1],original_dim[0]),interpolation=cv2.INTER_CUBIC)
            store_path = store_dir + "/" + name
            cv2.imwrite(store_path,trans_img)


def train_test_split(img_dir,store_train_dir,store_test_dir,split_ratio=0.7):
    init_dir('./dataset')
    init_dir(store_train_dir)
    init_dir(store_test_dir)
    imgs = os.listdir(img_dir)
    size = len(imgs)
    imgs_train = imgs[:int(size*split_ratio)]
    imgs_test = imgs[int(size*split_ratio):]
    for name in imgs_train:
        img = cv2.imread(img_dir + '/' + name)
        store_path = store_train_dir + "/" + name
        cv2.imwrite(store_path,img)
    for name in imgs_test:
        img = cv2.imread(img_dir + '/' + name)
        store_path = store_test_dir + "/" + name
        cv2.imwrite(store_path, img)

def compare_img(source_path,target_path):
    src_img = cv2.imread(source_path)
    tar_img = cv2.imread(target_path)
    ssim_const = ssim(src_img,tar_img,multichannel=True)
    psnr_const = psnr(src_img,tar_img)
    print('ssim : ',ssim_const)
    print('psnr : ',psnr_const)

if __name__ == "__main__":
    #two_K_human_img_selection("original_img","input_img2",0)
    #crop_with_scale('face_img','face_img_4',4)
    #create_database('HR_img_4','LR_img_0.25',0.25)
    #face_crop('HR_img','face_img')
    #data_augment('HR_img','HR_img_aug')
    #crop_with_scale('HR_img_aug','HR_img_aug_4',4)
    #img_transform('HR_img','HR_img_bicubic',0.25)
    train_test_split('face_img_4','./dataset/HR_img_train','./dataset/HR_img_test')
    #compare_img('HR_img/1.jpg','HR_img/1.jpg')



















