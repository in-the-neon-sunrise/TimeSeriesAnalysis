from PySide6.QtWidgets import (
    QLabel, QVBoxLayout, QHBoxLayout, QGroupBox,
    QPushButton, QComboBox, QSpinBox,
    QScrollArea, QWidget, QTableWidget, QTableWidgetItem
)

from PySide6.QtCore import QTimer
import numpy as np

from ui.pages.base_page import BasePage


class MarkovPage(BasePage):

    def __init__(self, data_vm):
        super().__init__()

        self.vm = data_vm

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        main_layout = QVBoxLayout(content)
        main_layout.setSpacing(14)

        title = QLabel("Анализ состояний (Марковские цепи)")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        main_layout.addWidget(title)

        body_layout = QHBoxLayout()
        main_layout.addLayout(body_layout)

        # ---------------- НАСТРОЙКИ ----------------

        controls = QVBoxLayout()

        controls.addWidget(self._order_group())
        controls.addWidget(self._params_group())

        self.run_btn = QPushButton("Построить модель")
        self.run_btn.clicked.connect(self.run_model)
        controls.addWidget(self.run_btn)

        self.export_btn = QPushButton("Экспорт результатов")
        controls.addWidget(self.export_btn)

        controls.addStretch()

        body_layout.addLayout(controls, 1)

        # ---------------- РЕЗУЛЬТАТЫ ----------------

        results_layout = QVBoxLayout()

        results_layout.addWidget(QLabel("Матрица переходов"))

        self.transition_table = QTableWidget()
        results_layout.addWidget(self.transition_table)

        results_layout.addWidget(QLabel("Стационарное распределение"))

        self.stationary_table = QTableWidget()
        self.stationary_table.setColumnCount(2)
        self.stationary_table.setHorizontalHeaderLabels([
            "State",
            "Probability"
        ])
        results_layout.addWidget(self.stationary_table)

        results_layout.addWidget(QLabel("Основные характеристики"))

        self.metrics_table = QTableWidget()
        self.metrics_table.setColumnCount(2)
        self.metrics_table.setHorizontalHeaderLabels([
            "Metric",
            "Value"
        ])
        results_layout.addWidget(self.metrics_table)

        body_layout.addLayout(results_layout, 2)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

    # ---------------- ORDER ----------------

    def _order_group(self):

        group = QGroupBox("Параметры модели")

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Порядок цепи"))

        self.order_box = QComboBox()
        self.order_box.addItems(["1", "2"])

        layout.addWidget(self.order_box)

        group.setLayout(layout)

        return group

    # ---------------- PARAMS ----------------

    def _params_group(self):

        group = QGroupBox("Дополнительные параметры")

        layout = QVBoxLayout()

        layout.addWidget(QLabel("Минимальное число переходов"))

        self.min_transitions = QSpinBox()
        self.min_transitions.setRange(1, 50)
        self.min_transitions.setValue(5)

        layout.addWidget(self.min_transitions)

        group.setLayout(layout)

        return group

    # ---------------- RUN MODEL ----------------

    def run_model(self):

        self.run_btn.setEnabled(False)
        self.run_btn.setText("Оценка модели...")

        QTimer.singleShot(1500, self.generate_results)

    # ---------------- GENERATE RESULTS ----------------

    def generate_results(self):

        states = np.random.randint(3, 6)

        matrix = np.random.rand(states, states)
        matrix = matrix / matrix.sum(axis=1, keepdims=True)

        self.populate_transition_matrix(matrix)
        self.populate_stationary(matrix)
        self.populate_metrics(states)

        self.run_btn.setEnabled(True)
        self.run_btn.setText("Построить модель")

    # ---------------- TRANSITION MATRIX ----------------

    def populate_transition_matrix(self, matrix):

        n = matrix.shape[0]

        self.transition_table.setRowCount(n)
        self.transition_table.setColumnCount(n)

        headers = [f"S{i}" for i in range(n)]

        self.transition_table.setHorizontalHeaderLabels(headers)
        self.transition_table.setVerticalHeaderLabels(headers)

        for i in range(n):
            for j in range(n):

                value = matrix[i, j]

                self.transition_table.setItem(
                    i,
                    j,
                    QTableWidgetItem(f"{value:.3f}")
                )

    # ---------------- STATIONARY ----------------

    def populate_stationary(self, matrix):

        eigvals, eigvecs = np.linalg.eig(matrix.T)

        stat = np.real(eigvecs[:, np.isclose(eigvals, 1)])
        stat = stat[:, 0]
        stat = stat / stat.sum()

        n = len(stat)

        self.stationary_table.setRowCount(n)

        for i in range(n):

            self.stationary_table.setItem(
                i, 0, QTableWidgetItem(f"S{i}")
            )

            self.stationary_table.setItem(
                i, 1, QTableWidgetItem(f"{stat[i]:.3f}")
            )

    # ---------------- METRICS ----------------

    def populate_metrics(self, states):

        metrics = [
            ("Number of states", states),
            ("Observed transitions", np.random.randint(150, 400)),
            ("Entropy rate", np.random.uniform(0.5, 1.5)),
            ("Average state duration", np.random.uniform(3, 12))
        ]

        self.metrics_table.setRowCount(len(metrics))

        for i, (name, value) in enumerate(metrics):

            if isinstance(value, float):
                value = f"{value:.3f}"

            self.metrics_table.setItem(
                i, 0, QTableWidgetItem(name)
            )

            self.metrics_table.setItem(
                i, 1, QTableWidgetItem(str(value))
            )