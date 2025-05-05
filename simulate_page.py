import os
import re
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QPushButton, QFileDialog, QGridLayout
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QImage
from PIL import Image
from srgan_runner import run_srgan

class SimulatePage(QWidget):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.slice_folder = r"C:\Users\ptula\OneDrive - University of Greenwich\Documents\Year3\COMP1682_FYP\gitsrgan\Dataset_modified\output_downscaled"
        self.slices = self.load_slices()
        self.last_axis_selected = 'x'

        # keeps track of current selected slice
        self.current_sr_image = None
        self.slice_thickness = "0.25mm"
        self.fov = "220mm x 220mm"

        self.setLayout(self.create_ui())
        self.initialize_middle_slice()

    # function for loading the images from the 3 axes
    def load_slices(self):
        pattern = re.compile(r'.*-(x|y|z)-(\d+)\.png')
        slices = {'x': [], 'y': [], 'z': []}
        for fname in os.listdir(self.slice_folder):
            match = pattern.match(fname)
            if match:
                axis, index = match.groups()
                slices[axis].append((int(index), fname))

        for axis in slices:
            slices[axis] = sorted(slices[axis], key=lambda x: x[0])
        return slices

    def create_ui(self):
        main_layout = QVBoxLayout()

        # home Button
        home_btn = QPushButton("Home")
        home_btn.setFixedSize(100, 55)
        home_btn.setStyleSheet("font-size: 18px; font-weight: bold;")
        home_btn.clicked.connect(lambda: self.parent.switch_to_home())
        main_layout.addWidget(home_btn, alignment=Qt.AlignLeft)

        grid_wrapper = QHBoxLayout()
        grid_wrapper.setAlignment(Qt.AlignHCenter)

        grid = QGridLayout()
        self.slice_sliders = {}
        self.slice_labels = {}
        self.slice_indices = {}

        # initialises 3 panels, slice numbers and scroll bar
        for col, (axis, label) in enumerate(zip(['x', 'y', 'z'], ['Sagittal (X)', 'Coronal (Y)', 'Axial (Z)'])):
            panel_layout = QVBoxLayout()
            panel_layout.setAlignment(Qt.AlignCenter)

            img_label = QLabel()
            img_label.setFixedSize(300, 300)
            img_label.setAlignment(Qt.AlignCenter)
            img_label.setScaledContents(True)

            axis_label = QLabel(label)
            axis_label.setAlignment(Qt.AlignHCenter)
            axis_label.setStyleSheet("font-weight: bold; font-size: 20px;")

            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(0)
            slider.setMaximum(len(self.slices[axis]) - 1)
            slider.valueChanged.connect(lambda _, ax=axis: self.on_slider_change(ax))

            index_label = QLabel("Slice: 0")
            index_label.setAlignment(Qt.AlignHCenter)
            index_label.setStyleSheet("font-size: 18px;")

            self.slice_labels[axis] = img_label
            self.slice_sliders[axis] = slider
            self.slice_indices[axis] = index_label

            panel_layout.addWidget(img_label)
            panel_layout.addSpacing(8)
            panel_layout.addWidget(axis_label)
            panel_layout.addWidget(index_label)
            panel_layout.addWidget(slider)

            grid.addLayout(panel_layout, 0, col)

        grid_wrapper.addLayout(grid)
        main_layout.addLayout(grid_wrapper)

        output_layout = QHBoxLayout()
        output_layout.setAlignment(Qt.AlignCenter)

        # outputs SR image
        self.sr_label = QLabel()
        self.sr_label.setFixedSize(400, 400)
        self.sr_label.setAlignment(Qt.AlignCenter)
        self.sr_label.setScaledContents(True)
        output_layout.addWidget(self.sr_label)

        # info panel
        info_panel = QVBoxLayout()
        info_panel.setSpacing(10)

        self.run_btn = QPushButton("Select Slice")
        self.run_btn.setFixedSize(200, 55)
        self.run_btn.setStyleSheet("font-size: 18px; font-weight: bold;")
        self.run_btn.clicked.connect(self.run_srgan_slice)
        info_panel.addWidget(self.run_btn)

        self.info_label = QLabel()
        self.info_label.setStyleSheet("font-size: 20px;")
        self.info_label.setContentsMargins(0, -20, 0, 0)
        self.info_label.setWordWrap(True)
        info_panel.addWidget(self.info_label)

        output_layout.addSpacing(20)
        output_layout.addLayout(info_panel)

        main_layout.addLayout(output_layout)

        # save button
        self.save_btn = QPushButton("Save Image")
        self.save_btn.setFixedSize(150, 55)
        self.save_btn.setStyleSheet("font-size: 18px; font-weight: bold; padding: 8px 20px;")
        self.save_btn.clicked.connect(self.save_current_sr_image)
        main_layout.addSpacing(15)
        main_layout.addWidget(self.save_btn, alignment=Qt.AlignHCenter)

        return main_layout

    def on_slider_change(self, axis):
        self.last_axis_selected = axis
        self.update_preview(axis)

    def update_preview(self, axis):
        idx = self.slice_sliders[axis].value()
        index, filename = self.slices[axis][idx]
        full_path = os.path.join(self.slice_folder, filename)

        img = Image.open(full_path).convert('L')
        img_np = np.array(img, dtype=np.uint8)

        self.slice_indices[axis].setText(f"Slice: {index}")
        pixmap = self.numpy_to_pixmap(img_np, 300)
        self.slice_labels[axis].setPixmap(pixmap)

    def run_srgan_slice(self):
        axis = self.last_axis_selected
        idx = self.slice_sliders[axis].value()
        slice_index, filename = self.slices[axis][idx]
        path = os.path.join(self.slice_folder, filename)

        img = Image.open(path).convert('L')
        img_np = np.array(img, dtype=np.uint8)

        sr_img = run_srgan(img_np)
        self.current_sr_image = Image.fromarray(sr_img)

        pixmap = self.numpy_to_pixmap(sr_img, 440)
        self.sr_label.setPixmap(pixmap)

        self.info_label.setText(
            f"<b>Axis:</b> {axis.upper()}<br>"
            f"<b>Slice Number:</b> {slice_index}<br>"
            f"<b>Original Size:</b> {img_np.shape[1]} x {img_np.shape[0]} px<br>"
            f"<b>Upscaled Size:</b> {sr_img.shape[1]} x {sr_img.shape[0]} px<br>"
            f"<b>Slice Thickness:</b> {self.slice_thickness}<br>"
            f"<b>Field of View (FOV):</b> {self.fov}"
        )

    def numpy_to_pixmap(self, array, size=300):
        h, w = array.shape
        qimg = QImage(array.data, w, h, w, QImage.Format_Grayscale8)
        return QPixmap.fromImage(qimg).scaled(size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def initialize_middle_slice(self):
        for axis in ['x', 'y', 'z']:
            middle_index = len(self.slices[axis]) // 2
            self.slice_sliders[axis].setValue(middle_index)
            self.update_preview(axis)

    def save_current_sr_image(self):
        if self.current_sr_image:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if path:
                self.current_sr_image.save(path)


