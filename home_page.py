from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

# initialising the home page widget
class HomePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # creates a vertical layout holding all widgets
        layout = QVBoxLayout()

        # sets name of simulator and its alignment and style
        label = QLabel("Super Simulator")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 50px; font-weight: bold;")

        # horizontal layout for buttons
        button_layout = QHBoxLayout()

        # creates upload button and sets size and style
        upload_btn = QPushButton("Upload MRI")
        upload_btn.setFixedSize(250, 100)
        upload_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        upload_btn.clicked.connect(self.parent.switch_to_upload)

        # creates simulator button and sets size and style
        simulator_btn = QPushButton("MRI Simulator")
        simulator_btn.setFixedSize(250, 100)  # Set a larger size for the button
        simulator_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        simulator_btn.clicked.connect(self.parent.switch_to_simulate)

        # adds both buttons next to each other
        button_layout.addWidget(upload_btn)
        button_layout.addWidget(simulator_btn)
        button_layout.setSpacing(60)
        button_layout.setAlignment(Qt.AlignCenter)

        # add the title and buttons
        layout.addWidget(label)
        layout.addLayout(button_layout)
        layout.setSpacing(150)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)
