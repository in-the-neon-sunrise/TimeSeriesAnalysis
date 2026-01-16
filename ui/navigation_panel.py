from PySide6.QtWidgets import QWidget, QVBoxLayout, QPushButton
from PySide6.QtCore import Signal

class NavigationPanel(QWidget):
    data_clicked = Signal()
    preprocessing_clicked = Signal()
    features_clicked = Signal()
    segmentation_clicked = Signal()
    clustering_clicked = Signal()
    markov_clicked = Signal()
    report_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QVBoxLayout(self)

        self.data_btn = QPushButton("Импорт данных")
        self.preprocessing_btn = QPushButton("Предобработка")
        self.features_btn = QPushButton("Признаки")
        self.segmentation_btn = QPushButton("Сегментация")
        self.clustering_btn = QPushButton("Кластеризация")
        self.markov_btn = QPushButton("Цепи Маркова")
        self.report_btn = QPushButton("Отчет")

        self.data_btn.clicked.connect(self.data_clicked.emit)
        self.preprocessing_btn.clicked.connect(self.preprocessing_clicked.emit)
        self.features_btn.clicked.connect(self.features_clicked.emit)
        self.segmentation_btn.clicked.connect(self.segmentation_clicked.emit)
        self.clustering_btn.clicked.connect(self.clustering_clicked.emit)
        self.markov_btn.clicked.connect(self.markov_clicked.emit)
        self.report_btn.clicked.connect(self.report_clicked.emit)

        layout.addWidget(self.data_btn)
        layout.addWidget(self.preprocessing_btn)
        layout.addWidget(self.features_btn)
        layout.addWidget(self.segmentation_btn)
        layout.addWidget(self.clustering_btn)
        layout.addWidget(self.markov_btn)
        layout.addWidget(self.report_btn)
        layout.addStretch()