from PyQt5 import QtGui, QtCore, QtWidgets
import numpy as np
import sys


class DisplayImageWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super(DisplayImageWidget, self).__init__(parent)
        self.four_points = []

        self.button_clear = QtWidgets.QPushButton('clear_points')
        self.button_clear.clicked.connect(self.clear_points)
        self.button_confirm = QtWidgets.QPushButton('Confirm')
        self.button_confirm.clicked.connect(self.set_points)
        self.image_frame = QtWidgets.QLabel()

        self.layout = QtWidgets.QVBoxLayout()
        self.layout.addWidget(self.image_frame)
        self.layout.addWidget(self.button_clear)
        self.layout.addWidget(self.button_confirm)
        self.setLayout(self.layout)
        self.show_image()
    def mousePressEvent(self, QMouseEvent):
        x = QMouseEvent.pos().x                      # actual cursor position
        y = QMouseEvent.pos().y
        x_offset = self.image_frame.pos().x   # image offset
        y_offset = self.image_frame.pos().y
        self.four_points.append((x-x_offset, y-y_offset))
    @QtCore.pyqtSlot()
    def show_image(self):
        self.image = np.zeros((500, 500, 3))
        self.image = QtGui.QImage(self.image.data, self.image.shape[1], self.image.shape[0], QtGui.QImage.Format_RGB888).rgbSwapped()
        self.image_frame.setPixmap(QtGui.QPixmap.fromImage(self.image))

    @QtCore.pyqtSlot()
    def clear_points(self):
        self.four_points = []
    @QtCore.pyqtSlot()
    def set_points(self):
        print(self.four_points)
        self.four_points = []
if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    display_image_widget = DisplayImageWidget()
    display_image_widget.show()
    sys.exit(app.exec_())