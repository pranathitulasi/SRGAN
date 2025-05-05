from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

class HomePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        layout = QVBoxLayout()

        # name of simulator
        label = QLabel("Super Simulator")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 50px; font-weight: bold;")

        # layout for buttons
        button_layout = QHBoxLayout()

        # upload button
        upload_btn = QPushButton("Upload MRI")
        upload_btn.setFixedSize(250, 100)
        upload_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        upload_btn.clicked.connect(self.parent.switch_to_upload)

        # simulate button
        simulate_btn = QPushButton("MRI Simulator")
        simulate_btn.setFixedSize(250, 100)  # Set a larger size for the button
        simulate_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        simulate_btn.clicked.connect(self.parent.switch_to_simulate)

        # adds both buttons next to each other
        button_layout.addWidget(upload_btn)
        button_layout.addWidget(simulate_btn)
        button_layout.setSpacing(60)
        button_layout.setAlignment(Qt.AlignCenter)

        # add the title and buttons
        layout.addWidget(label)
        layout.addLayout(button_layout)
        layout.setSpacing(150)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)
