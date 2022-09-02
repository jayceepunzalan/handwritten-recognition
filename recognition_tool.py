# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import os
import cv2
import sys
import numpy as np

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
from keras.models import load_model

# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwritten Digit Recognition Tool")
        self.setGeometry(200, 200, 1000, 800)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)

        # variables
        self.drawing = False
        self.brushSize = 24
        self.brushColor = Qt.white
        self.image_filename = 'handwritten_digit.png'

        self.lastPoint = QPoint()

        # creating menu bar
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu("File")

        # creating clear action
        clearAction = QAction("Clear", self)
        fileMenu.addAction(clearAction)
        clearAction.triggered.connect(self.clear)


    # method for checking mouse cicks
    def mousePressEvent(self, event):
        # if left mouse button is pressed
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()


    # method for tracking mouse activity
    def mouseMoveEvent(self, event):
        # checking if left button is pressed and drawing flag is true
        if (event.buttons() & Qt.LeftButton) & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize,
                            Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()


    # method for mouse left button release
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False


    # paint event
    def paintEvent(self, event):
        canvasPainter = QPainter(self)
        canvasPainter.drawImage(self.rect(), self.image, self.image.rect())


    # method for clearing every thing on canvas
    def clear(self):
        self.image.fill(Qt.black)
        self.update()


    # method for removing existing region of 
    # interest (ROI) images in directory
    def delete_existing_roi(self):
        for file in os.listdir('./'):
            if file.startswith('roi') and file.endswith('.png'):
                os.remove(file)


    # method for preprocessing handwritten image
    def predict_image(self, image):
        model_f = 'cnn_handwritten_recog.h5'
        self.cur_model = load_model(model_f)

        image_data = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image_data = np.asarray(image_data)  
        image_data = cv2.resize(image_data,(28,28))
        image_data = np.array(image_data).reshape(-1,28,28,1)
        image_data = image_data/255.0
        
        prediction = self.cur_model.predict(image_data)
        predicted_digit = np.argmax(prediction)

        return predicted_digit


    # method for counting objects in an image
    def image_segmentation(self, image):
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

        return contours


    # method for sorting objects from left-to-right
    def sort_contours(self, contours):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in contours]
        (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes),
            key=lambda b:b[1][i], reverse=False))

        return contours


    # method for predicting the handwritten digit
    def predict_drawing(self):
        image = cv2.imread(self.image_filename)
        contours = self.image_segmentation(image)
        sorted_contours = self.sort_contours(contours)

        self.predicted_digit_list = []

        orig = image.copy()
        i = 0

        for cnt in sorted_contours:
            # Filtered countours are detected
            x,y,w,h = cv2.boundingRect(cnt)
            x = x - 30
            y = y - 30
            w = w + 60
            h = h + 60

            # Taking ROI of the cotour
            roi = orig[y:y+h, x:x+w]
            roi = cv2.resize(roi, (400,400))

            # Save your contours or characters
            cv2.imwrite("roi" + str(i) + ".png", roi)
            print('image successfully saved')
            print()
            i = i + 1 

        for i in range(len(sorted_contours)):
            image = cv2.imread("roi" + str(i) + ".png")

        for i in range(len(sorted_contours)):
            image = cv2.imread("roi" + str(i) + ".png")
            predicted_digit = self.predict_image(image)
            self.predicted_digit_list.append(predicted_digit)

        self.delete_existing_roi()
        self.show_popup(self.predicted_digit_list)


    def show_popup(self, prediction):
        msg = QMessageBox()
        msg.setWindowTitle("Prediction result")
        predicted_final = ''.join(str(digit) for digit in prediction)
        msg.setText(f'The predicted image is: {predicted_final}')
        msg.setIcon(QMessageBox.Information)
        msg.exec()


    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == Qt.Key_Return:
            self.image.save(self.image_filename)
            self.predict_drawing()


if __name__ == '__main__':
    # create pyqt5 app
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
