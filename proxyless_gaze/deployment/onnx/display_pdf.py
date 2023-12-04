'''
Written almost entirely by ChatGPT 3.5: https://chat.openai.com/share/4b60691d-3bb1-41ec-8e3b-54b00c976377
'''
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QFileDialog, QLabel, QProgressBar
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt, QTimer
import fitz  # PyMuPDF

class PDFViewer(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PDF Viewer")
        self.setGeometry(100, 100, 800, 600)

        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)

        self.layout = QVBoxLayout(self.central_widget)

        self.button_open = QPushButton("Open PDF", self)
        self.button_open.clicked.connect(self.open_pdf)
        self.layout.addWidget(self.button_open)

        self.viewer_widget = PDFViewerWidget(self)
        self.layout.addWidget(self.viewer_widget)

        self.button_previous = QPushButton("Previous Page", self)
        self.button_previous.clicked.connect(self.viewer_widget.previous_page)
        self.layout.addWidget(self.button_previous)

        self.button_next = QPushButton("Next Page", self)
        self.button_next.clicked.connect(self.viewer_widget.next_page)
        self.layout.addWidget(self.button_next)

        self.button_fit_to_screen = QPushButton("Fit to Screen", self)
        self.button_fit_to_screen.clicked.connect(self.viewer_widget.fit_to_screen)
        self.layout.addWidget(self.button_fit_to_screen)

        self.button_toggle_fullscreen = QPushButton("Toggle Fullscreen", self)
        self.button_toggle_fullscreen.clicked.connect(self.toggle_fullscreen)
        self.layout.addWidget(self.button_toggle_fullscreen)

        self.fullscreen = False

    def open_pdf(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Open PDF File", "", "PDF Files (*.pdf);;All Files (*)", options=options)

        if file_name:
            self.viewer_widget.load_pdf(file_name)
            self.fullscreen = True
            self.showFullScreen()
            self.viewer_widget.fit_to_screen()

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.showNormal()
        else:
            self.showFullScreen()

        self.fullscreen = not self.fullscreen


class PDFViewerWidget(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.layout = QVBoxLayout(self)

        self.doc = None
        self.current_page = 0

        self.label = QLabel(self)
        self.layout.addWidget(self.label)

        self.loading_bar = QProgressBar(self)
        self.layout.addWidget(self.loading_bar)

    def load_pdf(self, file_path):
        self.doc = fitz.open(file_path)
        self.current_page = 0
        self.update_display()

    def update_display(self):
        # if hasattr(self, 'label'):
        #     self.layout.removeWidget(self.label)
        if self.doc:
            page = self.doc[self.current_page]
            pixmap = page.get_pixmap()
            qt_image = QPixmap.fromImage(QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride, QImage.Format_RGB888))

            self.label.setPixmap(qt_image)

    def next_page(self):
        if self.doc and self.current_page < len(self.doc) - 1:
            self.current_page += 1
            self.update_display()

    def previous_page(self):
        if self.doc and self.current_page > 0:
            self.current_page -= 1
            self.update_display()

    def fit_to_screen(self):
        if self.doc:
            page = self.doc[self.current_page]
            pixmap = page.get_pixmap()
            qt_image = QPixmap.fromImage(QImage(pixmap.samples, pixmap.width, pixmap.height, pixmap.stride, QImage.Format_RGB888))

            label_size = self.label.size()

            scaled_pixmap = qt_image.scaled(label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.label.setPixmap(scaled_pixmap)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PDFViewer()
    window.show()
    sys.exit(app.exec_())
