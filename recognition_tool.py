# importing libraries
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import sys
import os
import cv2
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from keras.models import load_model

# window class
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Handwritten Digit Recognition Tool")
        self.setGeometry(1200, 800, 1000, 800)
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.black)

        # variables
        self.drawing = False
        self.brushSize = 36
        self.brushColor = Qt.white

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

    # method for predicting the handwrittend digit
    def predict_drawing(self):
        model_f = 'cnn_handwritten_recog.h5'
        self.cur_model = load_model(model_f)

        image_data = cv2.imread('handwritten_digit.png')
        image_data =cv2.cvtColor(image_data,cv2.COLOR_BGR2GRAY)
        image_data = np.asarray(image_data)  
        image_data = cv2.resize(image_data,(28,28))
        image_data = np.array(image_data).reshape(-1,28,28,1)
        image_data = image_data/255.0

        prediction = self.cur_model.predict(image_data)
        self.show_popup(prediction)

    def show_popup(self, prediction):
        msg = QMessageBox()
        msg.setWindowTitle("Prediction result")
        msg.setText(f'The predicted image is: {np.argmax(prediction)}')
        msg.setIcon(QMessageBox.Information)
        msg.exec()

    def keyPressEvent(self, qKeyEvent):
        if qKeyEvent.key() == Qt.Key_Return:
            self.image.save('handwritten_digit.png')
            self.predict_drawing()


if __name__ == '__main__':
    # create pyqt5 app
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())
