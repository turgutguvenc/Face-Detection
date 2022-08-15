import cv2
import joblib
import json
import numpy as np
import base64
import wavelet
import matplotlib.pyplot as plt

__class_name_to_number = dict() #__variable is a private variable assigned this file
__class_number_to_name = dict()
__model = None #This private will store model
def classify_image(image_base64_data,file_path=None):
    images = get_cropped_image_if_2_eyes( image_base64_data)
    results = list()
    for img in images:
        scalled_raw_img = cv2.resize(img, (32, 32))
        img_har = wavelet.w2d(img, 'db1', 5)
        scalled_img_har = cv2.resize(img_har, (32, 32))
        combined_img = np.vstack((scalled_raw_img.reshape(32 * 32 * 3, 1), scalled_img_har.reshape(32 * 32, 1)))
        #len_image_array = 32*32*3*32*32*1
        final_img = np.array( combined_img).reshape(1,4096).astype(float)
        results.append({"class":class_number_to_name(__model.predict(final_img)[0]),
                        "class dictionary": __class_number_to_name})
    return results


# This will load our trained machine learning model from artifacts file
def load_saved_artifacts():
    print("loading saved artifacts...start")
    global __class_name_to_number
    global __class_number_to_name

    with open("./artifacts/class_dictionary.json", "r") as f:
        __class_name_to_number = json.load(f)
        __class_number_to_name = {v:k for k,v in __class_name_to_number.items()}

    global __model
    if __model is None:
        with open('./artifacts/famous_person_classifier.pkl', 'rb') as f:
            __model = joblib.load(f)
    print("loading saved artifacts...done")

# This function gets images in base64 format and transform them into cv2 images:
def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    encoded_data = b64str.split(',')[0]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img


def get_base64_for_Khabib():
    with open("base64.txt") as file:
        return file.read()



def get_cropped_image_if_2_eyes(image_base64_data):
    face_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('./opencv/haarcascades/haarcascade_eye.xml')


    img = get_cv2_image_from_base64_string(image_base64_data)

    #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)

    cropped_faces = []
    for (x,y,w,h) in faces:
            #roi_gray = gray[y:y+h, x:x+w]
            roi_color = img[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(img)
            if len(eyes) >= 2:
                cropped_faces.append(roi_color)
    return cropped_faces


def class_number_to_name(class_num):
    return __class_number_to_name[class_num]

def main():
    load_saved_artifacts()
    print(classify_image(get_base64_for_Khabib(),None))
    print(classify_image(""))






if __name__ == "__main__":
    main()