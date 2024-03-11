import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QPushButton, QWidget, QFileDialog, QMessageBox
from PyQt5.QtGui import QPixmap
import requests


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Приложение для распознавания изображений")
        self.setGeometry(100, 100, 400, 300)

        self.image_label = QLabel(self)
        self.image_label.setGeometry(50, 50, 300, 200)

        self.upload_button = QPushButton("Загрузить изображение", self)
        self.upload_button.setGeometry(150, 250, 200, 30)
        self.upload_button.clicked.connect(self.upload_image)

        self.prediction_label = QLabel(self)
        self.prediction_label.setGeometry(50, 270, 300, 20)
        self.prediction_label.setText("Класс: ")

        self.confidence_label = QLabel(self)
        self.confidence_label.setGeometry(50, 290, 300, 20)
        self.confidence_label.setText("Точность: ")

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Изображение", "", "Файлы (*.jpg)")
        if file_path:
            pixmap = QPixmap(file_path)
            self.image_label.setPixmap(pixmap)
            self.image_label.setScaledContents(True)

            # Send the image to the API for prediction
            url = 'http://127.0.0.1:5000/predict'
            files = {'image': open(file_path, 'rb')}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                data = response.json()
                prediction = data['prediction']
                confidence = data['confidence']
                self.prediction_label.setText("Класс: " + prediction)
                self.confidence_label.setText("Точность: " + str(confidence))
            else:
                QMessageBox.critical(self, "Ошибка", "Не удалось получить")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
