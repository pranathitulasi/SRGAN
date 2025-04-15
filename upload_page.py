import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QFileDialog, QHBoxLayout, QSpacerItem, QSizePolicy
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
from PIL import Image
from srgan_runner import run_srgan

class UploadPage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        # Main layout
        main_layout = QVBoxLayout()

        # Top bar: Home button
        top_bar = QHBoxLayout()
        home_btn = QPushButton("Home")
        home_btn.setFixedSize(100, 55)  # Bigger button
        home_btn.setStyleSheet("font-size: 20px; font-weight: bold;")
        home_btn.clicked.connect(lambda: self.parent.switch_to_home())
        top_bar.addWidget(home_btn)
        top_bar.addStretch()
        main_layout.addLayout(top_bar)

        # Image display layout (side by side)
        self.image_layout = QHBoxLayout()

        # Original Image column (hidden initially)
        self.original_label = QLabel()
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setFixedSize(512, 512)  # Large image display

        # Super-Resolved Image column (hidden initially)
        self.sr_label = QLabel()
        self.sr_label.setAlignment(Qt.AlignCenter)
        self.sr_label.setFixedSize(512, 512)  # Large image display

        self.original_text = QLabel("Original")
        self.original_text.setAlignment(Qt.AlignCenter)
        self.original_text.setStyleSheet("font-size: 25px; font-weight: bold;")  # Larger text
        self.original_text.setVisible(False)

        self.sr_text = QLabel("Super-Resolved")
        self.sr_text.setAlignment(Qt.AlignCenter)
        self.sr_text.setStyleSheet("font-size: 25px; font-weight: bold;")  # Larger text
        self.sr_text.setVisible(False)

        orig_col = QVBoxLayout()
        orig_col.addWidget(self.original_label, alignment=Qt.AlignCenter)
        orig_col.addWidget(self.original_text, alignment=Qt.AlignCenter)

        sr_col = QVBoxLayout()
        sr_col.addWidget(self.sr_label, alignment=Qt.AlignCenter)
        sr_col.addWidget(self.sr_text, alignment=Qt.AlignCenter)

        self.image_layout.addLayout(orig_col)
        self.image_layout.addLayout(sr_col)

        self.image_layout.setContentsMargins(0, 40, 0, 0)

        main_layout.addLayout(self.image_layout)

        # Spacer to add space between content and buttons
        main_layout.addItem(QSpacerItem(0, 20, QSizePolicy.Minimum, QSizePolicy.Expanding))

        # Upload and Save buttons side by side at bottom
        button_layout = QHBoxLayout()

        # Upload button (bigger button)
        self.upload_btn = QPushButton("Upload PNG")
        self.upload_btn.setFixedSize(200, 90)  # Larger button
        self.upload_btn.clicked.connect(self.upload_image)
        self.upload_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        button_layout.addWidget(self.upload_btn)

        # Save button (initially hidden, bigger button)
        self.save_btn = QPushButton("Save SR Image")
        self.save_btn.setFixedSize(200, 90)  # Larger button
        self.save_btn.setStyleSheet("font-size: 25px; font-weight: bold;")
        self.save_btn.setVisible(False)  # Only shows after upload
        self.save_btn.clicked.connect(self.save_image)
        button_layout.addWidget(self.save_btn)

        main_layout.addLayout(button_layout)

        # Set the layout for the page
        self.setLayout(main_layout)

    def upload_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open PNG Image", "", "Image Files (*.png)")
        if not file_path:
            return

        # Load the image
        image = Image.open(file_path).convert('L')
        image_np = np.array(image)

        # Run SRGAN
        sr_image = run_srgan(image_np)

        # Display the images
        orig_pixmap = self.numpy_to_pixmap(image_np)
        sr_pixmap = self.numpy_to_pixmap(sr_image)

        self.original_label.setPixmap(orig_pixmap)
        self.sr_label.setPixmap(sr_pixmap)

        # Show labels after image upload
        self.original_text.setVisible(True)
        self.sr_text.setVisible(True)

        # Save the SR image for later saving
        self.last_sr_image = sr_image  # Store numpy array for saving
        self.save_btn.setVisible(True)  # Show the "Save" button

    def numpy_to_pixmap(self, array):
        # Convert numpy array to QPixmap
        if array.ndim == 2:
            h, w = array.shape
            bytes_per_line = w
            image = QImage(array.data, w, h, bytes_per_line, QImage.Format_Grayscale8)
            return QPixmap.fromImage(image).scaled(512, 512, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        else:
            return QPixmap()

    def save_image(self):
        # Save the SR image to disk
        if hasattr(self, 'last_sr_image'):
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)"
            )
            if file_path:
                # Convert numpy array to Image and save
                img = Image.fromarray(np.clip(self.last_sr_image, 0, 255).astype(np.uint8))
                img.save(file_path)

# ui needs to be more sophisticated, all images need to be same size
