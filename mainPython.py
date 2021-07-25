import sys
import time

import cv2
import numpy as np
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt
import pytesseract
from pytesseract import Output
from PIL import Image
import io


class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi("MainUI.ui", self)
        self.Image = None
        self.btnScan.setEnabled(False)
        self.btnConvert.clicked.connect(self.fungsi)
        self.actionExit.triggered.connect(self.exit)
        self.actionOpen_Image.triggered.connect(self.openFile)
        self.btnScan.clicked.connect(self.OCR)
        a = (pytesseract.get_languages())
        listToStr = ', '.join(map(str, a))
        print("Language Installed : " + listToStr)

    def openFile(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "Pilih Gambar", "",
                                                  "All Files (*);;Gambar JPEG (*.jpeg);;Gambar JPG (*.jpg);;Gambar PNG (*.png)",
                                                  options=options)
        if fileName:
            self.Image = cv2.imread(fileName)
            self.displayImage(1)
            print("FilePath : " + fileName)

    def exit(self):
        sys.exit(0)

    def fungsi(self):
        a = self.radBinaryImage.isChecked()
        b = self.radThresholding.isChecked()
        c = self.radOpening.isChecked()
        d = self.radGaussianBlur.isChecked()

        if self.Image is None:
            QMessageBox.warning(self, "Error", "Masukkan Gambar Terlebih Dahulu!")
        else:
            if a == False | b == False | c == False | d == False :
                QMessageBox.warning(self, "Error", "Pilih Salah Satu Metode Processing")
            elif a == True:
                QMessageBox.information(self, "Info", "Binary Image")
                self.binary()
            elif b == True:
                QMessageBox.information(self, "Info", "Thresholding")
                self.thresholding()
            elif c == True:
                QMessageBox.information(self, "Info", "Opening")
                self.opening()
            elif d == True:
                QMessageBox.information(self, "Info", "Gaussian Filtering")
                self.gaussianBlur()
            self.btnScan.setEnabled(True)

    def displayImage(self, windows=1):
        qformat = QImage.Format_Indexed8

        if len(self.Image.shape) == 3:
            if (self.Image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.Image, self.Image.shape[1], self.Image.shape[0], self.Image.strides[0], qformat)

        img = img.rgbSwapped()

        if windows == 1:
            self.mainLabel.setPixmap(QPixmap.fromImage(img).scaled(436, 551, Qt.KeepAspectRatio))
            self.mainLabel.setAlignment(QtCore.Qt.AlignCenter)
        if windows == 2:
            self.procLabel.setPixmap(QPixmap.fromImage(img).scaled(436, 551, Qt.KeepAspectRatio))
            self.procLabel.setAlignment(QtCore.Qt.AlignCenter)

    def grayscale(self):
        try:
            H, W = self.Image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.Image[i, j, 0] +
                                         0.587 * self.Image[i, j, 1] +
                                         0.114 * self.Image[i, j, 2], 0, 255)
            self.Image = gray
            self.displayImage(2)
        except:
            pass

    def opening(self):
        _, thresh = cv2.threshold(self.Image, 160, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        self.Image = opening
        self.displayImage(2)

    def thresholding(self):
        self.grayscale()
        neighbour = 11
        c = 2
        thresh = cv2.adaptiveThreshold(self.Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, neighbour, c)
        self.Image = thresh
        self.displayImage(2)

    def binary(self):
        self.grayscale()
        H, W = self.Image.shape[:2]
        thres = 150
        for i in range(H - 1):
            for j in range(W - 1):
                a = self.Image.item(i, j)
                if a < thres:
                    b = 0
                elif a > thres:
                    b = 255
                else:
                    b = a

                self.Image.itemset((i, j), b)

        self.displayImage(2)

    def gaussianBlur(self):
        gauss = (1.0 / 273) * np.array(
            [[1, 4, 7, 4, 1],
             [4, 16, 26, 16, 4],
             [7, 26, 42, 26, 7],
             [4, 16, 26, 16, 4],
             [1, 4, 7, 4, 1]]
        )

        self.grayscale()
        output = self.mainKonvolusi(self.Image, gauss)

        fig = plt.imshow(output, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='off')
        fig.axes.get_xaxis().set_visible(False)
        fig.axes.get_yaxis().set_visible(False)

        plt.savefig('Temp/gauss.jpg', bbox_inches='tight', pad_inches=0)
        plt.close()
        self.Image = cv2.imread('Temp/gauss.jpg')
        self.displayImage(2)

    def mainKonvolusi(self, X, F):
        x_height = X.shape[0]
        x_width = X.shape[1]

        f_height = F.shape[0]
        f_width = F.shape[1]

        H = (f_height) // 2
        W = (f_width) // 2

        out = np.zeros ((x_height, x_width))

        for i in np.arange(H + 1, x_height - H):
            #print(i)
            for j in np.arange(W + 1, x_width - W):
                #print(j)
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)
                out[i, j] = sum

        return out

    def OCR(self):
        if self.Image is None:
            QMessageBox.warning(self, "Error", "Masukkan Gambar Terlebih Dahulu!")
        else:
            self.btnScan.setEnabled(False)
            lang = self.cbbBahasa.currentIndex()

            if lang == 0:
                x = pytesseract.image_to_string(self.Image, lang='eng')
            else:
                x = pytesseract.image_to_string(self.Image, lang='ind')

            self.textEdit.setPlainText(x)

            d = pytesseract.image_to_data(self.Image, output_type=Output.DICT)
            print(d.keys())

            n_boxes = len(d['text'])
            for i in range(n_boxes):
                #print(d['conf'][i])
                if int(d['conf'][i]) > 60:
                    (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
                    img = cv2.rectangle(self.Image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                else :
                    img = None

            if img is None :
                QMessageBox.warning(self, "Info", "Tulisan Tidak Terdeteksi")
                pass
            else :
                self.Image = img
                self.displayImage(2)

app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setFixedSize(1360, 662)
window.show()
sys.exit(app.exec_())
