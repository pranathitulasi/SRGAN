from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt

class HomePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        layout = QVBoxLayout()

        label = QLabel("Super Simulator")
        label.setAlignment(Qt.AlignCenter)
        label.setStyleSheet("font-size: 50px; font-weight: bold;")

        # Create a horizontal layout for buttons
        button_layout = QHBoxLayout()

        # Upload button
        upload_btn = QPushButton("Upload MRI")
        upload_btn.setFixedSize(250, 100)  # Set a larger size for the button
        upload_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        upload_btn.clicked.connect(self.parent.switch_to_upload)

        # Simulate button
        simulate_btn = QPushButton("MRI Simulator")
        simulate_btn.setFixedSize(250, 100)  # Set a larger size for the button
        simulate_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        simulate_btn.clicked.connect(self.parent.switch_to_simulate)

        # Add buttons to the horizontal layout
        button_layout.addWidget(upload_btn)
        button_layout.addWidget(simulate_btn)
        button_layout.setSpacing(60)  # Space between buttons
        button_layout.setAlignment(Qt.AlignCenter)  # Center the buttons

        # Add the label and button layout to the main vertical layout
        layout.addWidget(label)
        layout.addLayout(button_layout)
        layout.setSpacing(150)
        layout.setAlignment(Qt.AlignCenter)

        self.setLayout(layout)
