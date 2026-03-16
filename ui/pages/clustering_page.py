from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QScrollArea, QWidget, QTableWidget, QTableWidgetItem
)

from PySide6.QtCore import QTimer
import numpy as np

from ui.pages.base_page import BasePage


class ClusteringPage(BasePage):

    def __init__(self, data_vm):
        super().__init__()

        self.vm = data_vm

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        main_layout = QVBoxLayout(content)
        main_layout.setSpacing(14)

        title = QLabel("Кластеризация сегментов")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)

        body_layout = QHBoxLayout()
        main_layout.addLayout(body_layout)

        # левая панель настроек
        controls = QVBoxLayout()

        controls.addWidget(self._method_group())
        controls.addWidget(self._metric_group())
        controls.addWidget(self._params_group())

        self.run_btn = QPushButton("Запустить кластеризацию")
        self.run_btn.clicked.connect(self.run_clustering)
        controls.addWidget(self.run_btn)

        self.export_btn = QPushButton("Экспорт результатов")
        controls.addWidget(self.export_btn)

        controls.addStretch()

        body_layout.addLayout(controls, 1)

        # правая панель результатов
        results_layout = QVBoxLayout()

        results_layout.addWidget(QLabel("Метрики качества"))

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels(["Metric", "Value"])
        results_layout.addWidget(self.metrics_table)

        results_layout.addWidget(QLabel("Статистика кластеров"))

        self.cluster_table = QTableWidget()
        self.cluster_table.setColumnCount(3)
        self.cluster_table.setHorizontalHeaderLabels([
            "Cluster ID",
            "Objects",
            "Share (%)"
        ])
        results_layout.addWidget(self.cluster_table)

        results_layout.addWidget(QLabel("Распределение сегментов"))

        self.segment_table = QTableWidget()
        self.segment_table.setColumnCount(2)
        self.segment_table.setHorizontalHeaderLabels([
            "Segment",
            "Cluster"
        ])
        results_layout.addWidget(self.segment_table)

        body_layout.addLayout(results_layout, 2)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

    # ---------------- METHOD ----------------

    def _method_group(self):

        group = QGroupBox("Метод кластеризации")

        layout = QVBoxLayout()

        self.method_box = QComboBox()
        self.method_box.addItems([
            "K-means",
            "DBSCAN"
        ])

        layout.addWidget(self.method_box)

        group.setLayout(layout)

        return group

    # ---------------- METRIC ----------------

    def _metric_group(self):

        group = QGroupBox("Метрика расстояния")

        layout = QVBoxLayout()

        self.metric_box = QComboBox()
        self.metric_box.addItems([
            "Euclidean",
            "Manhattan",
            "Cosine"
        ])

        layout.addWidget(self.metric_box)

        group.setLayout(layout)

        return group

    # ---------------- PARAMS ----------------

    def _params_group(self):

        group = QGroupBox("Параметры")

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Количество кластеров"))

        self.k_clusters = QSpinBox()
        self.k_clusters.setRange(2, 15)
        self.k_clusters.setValue(4)

        layout.addWidget(self.k_clusters)

        layout.addWidget(QLabel("Epsilon (для DBSCAN)"))

        self.eps = QDoubleSpinBox()
        self.eps.setRange(0.01, 5.0)
        self.eps.setValue(0.5)

        layout.addWidget(self.eps)

        layout.addWidget(QLabel("Min samples"))

        self.min_samples = QSpinBox()
        self.min_samples.setRange(2, 50)
        self.min_samples.setValue(5)

        layout.addWidget(self.min_samples)

        group.setLayout(layout)

        return group

    # ---------------- RUN ----------------

    def run_clustering(self):

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Выполняется кластеризация...")

        QTimer.singleShot(1500, self.generate_results)

    # ---------------- GENERATE RESULTS ----------------

    def generate_results(self):

        k = self.k_clusters.value()
        total_objects = np.random.randint(150, 300)

        counts = np.random.multinomial(total_objects, [1/k]*k)

        self.populate_metrics()
        self.populate_cluster_table(counts, total_objects)
        self.populate_segment_table(total_objects, k)

        self.run_btn.setEnabled(True)
        self.run_btn.setText("Запустить кластеризацию")

    # ---------------- METRICS ----------------

    def populate_metrics(self):

        metrics = [
            ("Silhouette score", np.random.uniform(0.25, 0.75)),
            ("Davies–Bouldin index", np.random.uniform(0.6, 1.8)),
            ("Calinski–Harabasz index", np.random.uniform(80, 300))
        ]

        self.metrics_table.setRowCount(len(metrics))

        for i, (name, value) in enumerate(metrics):

            self.metrics_table.setItem(
                i, 0, QTableWidgetItem(name)
            )

            self.metrics_table.setItem(
                i, 1, QTableWidgetItem(f"{value:.3f}")
            )

    # ---------------- CLUSTER TABLE ----------------

    def populate_cluster_table(self, counts, total):

        self.cluster_table.setRowCount(len(counts))

        for i, c in enumerate(counts):

            share = 100 * c / total

            self.cluster_table.setItem(
                i, 0, QTableWidgetItem(str(i))
            )

            self.cluster_table.setItem(
                i, 1, QTableWidgetItem(str(c))
            )

            self.cluster_table.setItem(
                i, 2, QTableWidgetItem(f"{share:.1f}")
            )

    # ---------------- SEGMENT TABLE ----------------

    def populate_segment_table(self, total_objects, k):

        rows = min(20, total_objects)

        self.segment_table.setRowCount(rows)

        for i in range(rows):

            cluster = np.random.randint(0, k)

            self.segment_table.setItem(
                i, 0, QTableWidgetItem(f"Segment {i}")
            )

            self.segment_table.setItem(
                i, 1, QTableWidgetItem(str(cluster))
            )