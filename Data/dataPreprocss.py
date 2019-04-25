import os
import cv2
from mtcnn.mtcnn import MTCNN

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
    size = len(os.listdir(original_dir))
    for i in range(size):
        path = original_dir + '/' + str(i) + '.jpg'
        img = cv2.imread(path)
        height, width, channels = img.shape
        new_h = int(height/size_factor)*size_factor
        new_w = int(width/size_factor)*size_factor
        img = img[:new_h,:new_w,:]
        store_path = store_dir + "/" + str(i) + ".jpg"
        cv2.imwrite(store_path, img)


def data_augment(img_dir):
    # temp function, need to modify
    detector = MTCNN()
    imgs = os.listdir(img_dir)
    for name in imgs:
        img = cv2.imread('HR_img/244.jpg')
        #img = cv2.imread(img_dir+'/'+name)
        faces = detector.detect_faces(img)
        print('detected len;',len(faces))
        for face in faces:
            print(img.shape)
            print(face)
            box = face['box']
            cv2.rectangle(img,(box[0],box[1]),(box[0]+box[2],box[1]+box[3]),(255,0,0),2)
            img = cv2.resize(img,(int(img.shape[0]/4),int(img.shape[1]/4)))
            cv2.imshow('image',img)
            k = cv2.waitKey(0)

def face_crop(img_dir,store_dir):
    init_dir(store_dir)
    detector = MTCNN()
    imgs = os.listdir(img_dir)
    for name in imgs:
        img = cv2.imread(img_dir+'/'+name)
        faces = detector.detect_faces(img)
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




if __name__ == "__main__":
    #two_K_human_img_selection("original_img","input_img2",0)
    #crop_with_scale('HR_img','HR_img_4',4)
    #create_database('HR_img_4','LR_img_0.25',0.25)
    #data_augment('HR_img')
    face_crop('HR_img','face_img')
























