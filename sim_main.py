import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QStackedWidget
from home_page import HomePage
from upload_page import UploadPage
from simulate_page import SimulatePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Super Simulator")
        self.setGeometry(100, 100, 1500, 1000)

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)

        self.home_page = HomePage(self)
        self.upload_page = UploadPage(self)
        self.simulate_page = SimulatePage(self)

        self.stack.addWidget(self.home_page)
        self.stack.addWidget(self.upload_page)
        self.stack.addWidget(self.simulate_page)

        self.stack.setCurrentWidget(self.home_page)

    def switch_to_upload(self):
        self.stack.setCurrentWidget(self.upload_page)

    def switch_to_simulate(self):
        self.stack.setCurrentWidget(self.simulate_page)

    def switch_to_home(self):
        self.stack.setCurrentWidget(self.home_page)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
