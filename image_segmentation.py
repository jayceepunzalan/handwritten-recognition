import matplotlib.pyplot as plt
import cv2
import numpy as np
import imutils
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import load_model


image = cv2.imread('handwritten_digit.png')
image_height = image.shape[0]
image_width = image.shape[1]


def predict_image(image):
    model_f = 'cnn_handwritten_recog.h5'
    cur_model = load_model(model_f)

    image =cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    image = np.asarray(image)  
    image = cv2.resize(image,(28,28))
    image = np.array(image).reshape(-1,28,28,1)
    image = image/255.0

    prediction = cur_model.predict(image)
    predicted_digit = np.argmax(prediction)

    return predicted_digit


def image_recognition(image, height, width):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur_image = cv2.GaussianBlur(gray_image, (7,7), 0)
    ret, thresh2 = cv2.threshold(blur_image, 200, 255, cv2.THRESH_BINARY)
    # Find Canny edges
    edged = cv2.Canny(thresh2, 100, 200)
      
    # Finding Contours
    # Use a copy of the image e.g. edged.copy() 
    # since findContours alters the image
    contours, hierarchy = cv2.findContours(edged.copy(), 
        cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
      
    print("Number of Contours found = " + str(len(contours)))

    orig = image.copy()
    i = 0

    predicted_digit_list = []
    for cnt in contours:

        # Filtered countours are detected
        x,y,w,h = cv2.boundingRect(cnt)
        x = x - 30
        y = y - 30
        w = w + 60
        h = h + 60
        # x,y,w,h = int(x-(w/4)), int(y-(h/4)), int((3*w)/2), int((3*h)/2)

        # Taking ROI of the cotour
        roi = image[y:y+h, x:x+w]
        roi = cv2.resize(roi, (400,400))

        # Save your contours or characters
        cv2.imwrite("roi" + str(i) + ".png", roi)
        print('image successfully saved')
        print(f'image shape is: {roi.shape}')
        print()
        i = i + 1 

    for i in range(len(contours)):
        image = cv2.imread("roi" + str(i) + ".png")
        cv2.imshow("Sorted", image)
        cv2.waitKey(0)


    for i in range(len(contours)):
        image = cv2.imread("roi" + str(i) + ".png")
        predicted_digit = predict_image(image)
        print(predicted_digit)
        predicted_digit_list.append(predicted_digit)


    predicted_final = ''.join(str(digit) for digit in predicted_digit_list)
    print(f'Predicted digit is: {predicted_final}')
      
    # # Draw all contours
    # # -1 signifies drawing all contours
    # cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
      
    # cv2.imshow('Contours', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

def main():
    
    image_recognition(image, image_height, image_width) 
    for file in os.listdir('./'):
        if file.startswith('roi') and file.endswith('.png'):
            os.remove(file)

if __name__ == '__main__':
    main()