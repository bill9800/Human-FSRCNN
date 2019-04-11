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
    detector = MTCNN()
    imgs = os.listdir(img_dir)
    for name in imgs:
        img = cv2.imread(img_dir+'/'+name)
        faces = detector.detect_faces(img)
        for face in faces:
            print(img.shape)
            print(face)
            box = face['box']
            cv2.rectangle(img,(box[0],box[1]),(box[0]+box[3],box[1]+box[2]),(255,0,0),2)
            img = cv2.resize(img,(int(img.shape[0]/4),int(img.shape[1]/4)))
            cv2.imshow('image',img)
            k = cv2.waitKey(0)


if __name__ == "__main__":
    #two_K_human_img_selection("original_img","input_img2",0)
    #crop_with_scale('HR_img','HR_img_4',4)
    create_database('HR_img_4','LR_img_0.25',0.25)
    #data_augment('HR_img')
























