from PySide6.QtWidgets import (
    QLabel,
    QVBoxLayout,
    QGroupBox,
    QRadioButton,
    QComboBox,
    QPushButton,
    QSpinBox,
    QDoubleSpinBox,
    QScrollArea,
    QWidget,
    QTableWidget,
    QTableWidgetItem,
    QMessageBox,
)

from ui.pages.base_page import BasePage
from services.preprocessing_service import PreprocessingService


class PreprocessingPage(BasePage):
    def __init__(self, data_vm):
        super().__init__()

        self.df = None

        self.vm = data_vm
        self.vm.data_loaded.connect(self.on_data_loaded)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)

        content = QWidget()
        scroll.setWidget(content)

        layout = QVBoxLayout(content)
        layout.setSpacing(14)

        title = QLabel("Предварительная обработка данных")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)

        self.column_selector = QComboBox()
        self.column_selector.setEnabled(False)
        self.column_selector.currentIndexChanged.connect(self.update_preview)

        layout.addWidget(QLabel("Анализируемый столбец"))
        layout.addWidget(self.column_selector)

        self.missing_group = self._missing_group()
        layout.addWidget(self.missing_group)

        self.smoothing_group = self._smoothing_group()
        layout.addWidget(self.smoothing_group)

        self.scale_group = self._scaling_group()
        layout.addWidget(self.scale_group)

        self.stats_group = QGroupBox("Сравнение статистик до и после")
        stats_layout = QVBoxLayout(self.stats_group)
        self.stats_table = QTableWidget(0, 3)
        self.stats_table.setHorizontalHeaderLabels(["metric", "before", "after"])
        self.stats_table.verticalHeader().setVisible(False)
        stats_layout.addWidget(self.stats_table)
        layout.addWidget(self.stats_group)

        self.preview_group = QGroupBox("Предпросмотр первых 10 значений")
        preview_layout = QVBoxLayout(self.preview_group)
        self.preview_table = QTableWidget(0, 3)
        self.preview_table.setHorizontalHeaderLabels(["index", "original", "processed"])
        self.preview_table.verticalHeader().setVisible(False)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(self.preview_group)

        self.warning_group = QGroupBox("Предупреждения")
        warning_layout = QVBoxLayout(self.warning_group)
        self.warning_label = QLabel("Нет")
        self.warning_label.setWordWrap(True)
        warning_layout.addWidget(self.warning_label)
        layout.addWidget(self.warning_group)

        self.apply_btn = QPushButton("Применить предобработку к столбцу")
        self.apply_btn.clicked.connect(self.apply)
        layout.addWidget(self.apply_btn)

        root_layout = QVBoxLayout(self)
        root_layout.addWidget(scroll)

        self.update_preview()

    def _scaling_group(self):
        box = QGroupBox("Масштабирование")
        layout = QVBoxLayout(box)

        self.scale_none = QRadioButton("Не применять")
        self.scale_minmax = QRadioButton("Min–Max")
        self.scale_z = QRadioButton("Z-score")
        self.scale_robust = QRadioButton("Robust scaling")

        self.scale_none.setChecked(True)

        for btn in [
            self.scale_none,
            self.scale_minmax,
            self.scale_z,
            self.scale_robust
        ]:
            btn.toggled.connect(self.update_preview)
            layout.addWidget(btn)

        return box

    def _missing_group(self):
        box = QGroupBox("Обработка пропусков")
        layout = QVBoxLayout(box)

        self.miss_none = QRadioButton("Не обрабатывать")
        self.miss_drop = QRadioButton("Удалить строки")
        self.miss_mean = QRadioButton("Заполнить mean")
        self.miss_median = QRadioButton("Заполнить median")
        self.miss_interp = QRadioButton("Интерполяция")

        self.miss_none.setChecked(True)

        for btn in [
            self.miss_none,
            self.miss_drop,
            self.miss_mean,
            self.miss_median,
            self.miss_interp
        ]:
            btn.toggled.connect(self.update_preview)
            layout.addWidget(btn)

        return box

    def _smoothing_group(self):
        box = QGroupBox("Сглаживание")
        layout = QVBoxLayout(box)

        self.smooth_none = QRadioButton("Не применять")
        self.smooth_ma = QRadioButton("Скользящее среднее")
        self.smooth_median = QRadioButton("Медианный фильтр")
        self.smooth_ewm = QRadioButton("Экспоненциальное сглаживание")

        self.smooth_none.setChecked(True)

        self.window_spin = QSpinBox()
        self.window_spin.setRange(2, 100)
        self.window_spin.setValue(5)
        self.window_spin.valueChanged.connect(self.update_preview)

        self.alpha_spin = QDoubleSpinBox()
        self.alpha_spin.setRange(0.01, 1.0)
        self.alpha_spin.setSingleStep(0.05)
        self.alpha_spin.setValue(0.3)
        self.window_spin.valueChanged.connect(self.update_preview)

        for btn in [
            self.smooth_none,
            self.smooth_ma,
            self.smooth_median,
            self.smooth_ewm
        ]:
            btn.toggled.connect(self.update_preview)
            layout.addWidget(btn)

        layout.addWidget(QLabel("Окно сглаживания:"))
        layout.addWidget(self.window_spin)

        layout.addWidget(QLabel("Alpha (для EWM):"))
        layout.addWidget(self.alpha_spin)

        return box

    def _get_options(self):
        missing = "none"
        if self.miss_drop.isChecked():
            missing = "drop"
        elif self.miss_mean.isChecked():
            missing = "mean"
        elif self.miss_median.isChecked():
            missing = "median"
        elif self.miss_interp.isChecked():
            missing = "interpolate"

        smoothing = "none"
        if self.smooth_ma.isChecked():
            smoothing = "moving_average"
        elif self.smooth_median.isChecked():
            smoothing = "median"
        elif self.smooth_ewm.isChecked():
            smoothing = "ewm"

        scaling = "none"
        if self.scale_minmax.isChecked():
            scaling = "minmax"
        elif self.scale_z.isChecked():
            scaling = "zscore"
        elif self.scale_robust.isChecked():
            scaling = "robust"

        return {
            "missing": missing,
            "smoothing": smoothing,
            "scaling": scaling,
            "window": self.window_spin.value(),
            "alpha": self.alpha_spin.value(),
        }

    @staticmethod
    def _format_value(value):
        if value is None:
            return "—"
        if isinstance(value, str):
            return value
        try:
            if value != value:
                return "—"
        except Exception:
            pass
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def _set_warning(self, text: str):
        self.warning_label.setText(text or "Нет")

    def _clear_tables(self):
        self.stats_table.setRowCount(0)
        self.preview_table.setRowCount(0)

    def _update_stats_table(self, before: dict, after: dict):
        metrics = ["rows", "missing", "mean", "median", "std", "min", "max", "q25", "q75"]
        self.stats_table.setRowCount(len(metrics))

        for i, metric in enumerate(metrics):
            self.stats_table.setItem(i, 0, QTableWidgetItem(metric))
            self.stats_table.setItem(i, 1, QTableWidgetItem(self._format_value(before.get(metric))))
            self.stats_table.setItem(i, 2, QTableWidgetItem(self._format_value(after.get(metric))))

        self.stats_table.resizeColumnsToContents()

    def _update_preview_table(self, preview_df):
        self.preview_table.setRowCount(len(preview_df))

        for row_pos, (_, row) in enumerate(preview_df.iterrows()):
            self.preview_table.setItem(row_pos, 0, QTableWidgetItem(self._format_value(row["index"])))
            self.preview_table.setItem(row_pos, 1, QTableWidgetItem(self._format_value(row["original"])))
            self.preview_table.setItem(row_pos, 2, QTableWidgetItem(self._format_value(row["processed"])))

        self.preview_table.resizeColumnsToContents()

    def update_preview(self):
        self._clear_tables()

        if self.df is None:
            self._set_warning("Данные не загружены.")
            return

        if self.column_selector.count() == 0:
            self._set_warning("Нет числовых столбцов для предобработки.")
            return

        col = self.column_selector.currentText()
        if not col or col not in self.df.columns:
            self._set_warning("Выбранный столбец отсутствует в данных.")
            return

        options = self._get_options()
        original = self.df[col]
        warning_messages = []

        if options["smoothing"] in {"moving_average", "median"} and options["window"] > len(original):
            warning_messages.append(
                f"Окно сглаживания ({options['window']}) больше длины ряда ({len(original)})."
            )

        try:
            processed = PreprocessingService.apply_pipeline(
                original,
                missing_method=options["missing"],
                smoothing_method=options["smoothing"],
                scaling_method=options["scaling"],
                window=options["window"],
                alpha=options["alpha"],
            )
        except Exception as exc:
            self._set_warning(f"Ошибка предпросмотра: {exc}")
            return

        if processed.empty:
            warning_messages.append("После предобработки не осталось строк.")

        if processed.replace([float("inf"), float("-inf")], float("nan")).dropna().empty:
            warning_messages.append("После предобработки нет валидных числовых значений.")

        before_summary = PreprocessingService.series_summary(original)
        after_summary = PreprocessingService.series_summary(processed)
        preview_df = PreprocessingService.build_preview(original, processed, n=10)

        self._update_stats_table(before_summary, after_summary)
        self._update_preview_table(preview_df)
        self._set_warning("\n".join(warning_messages) if warning_messages else "Нет")

    def apply(self):
        if self.df is None:
            QMessageBox.warning(self, "Предобработка", "Данные не загружены.")
            return

        col = self.column_selector.currentText()
        if not col or col not in self.df.columns:
            QMessageBox.warning(self, "Предобработка", "Выбранный столбец отсутствует.")
            return

        options = self._get_options()
        new_df = self.df.copy()

        try:
            if options["missing"] == "drop":
                new_df = new_df.loc[new_df[col].notna()].copy()
                missing_method = "none"
            else:
                missing_method = options["missing"]

            processed = PreprocessingService.apply_pipeline(
                new_df[col],
                missing_method=missing_method,
                smoothing_method=options["smoothing"],
                scaling_method=options["scaling"],
                window=options["window"],
                alpha=options["alpha"],
            )

            if processed.empty and len(new_df) > 0:
                QMessageBox.warning(self, "Предобработка", "После обработки не осталось данных в выбранном столбце.")
                return

            new_df[col] = processed
            self.df = new_df

            if hasattr(self.vm, "project") and hasattr(self.vm.project, "set_processed_data"):
                self.vm.project.set_processed_data(
                    self.df,
                    params={
                        "column": col,
                        "missing": options["missing"],
                        "smoothing": options["smoothing"],
                        "scaling": options["scaling"],
                        "window": options["window"],
                        "alpha": options["alpha"],
                    },
                )

            self.update_preview()
            QMessageBox.information(self, "Предобработка", "Предобработка успешно применена.")
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка", f"Не удалось применить предобработку: {exc}")

    def on_data_loaded(self, df):
        self.df = df
        numeric_cols = df.select_dtypes(include="number").columns
        self.column_selector.clear()

        if len(numeric_cols) == 0:
            self.column_selector.setEnabled(False)
            self._clear_tables()
            self._set_warning("Нет числовых столбцов для предобработки.")
            return

        self.column_selector.addItems(numeric_cols)
        self.column_selector.setEnabled(True)
        self.update_preview()

    def init_ui_for_data(self):
        if self.df is None:
            return

        numeric_cols = self.df.select_dtypes(include="number").columns
        self.column_selector.clear()
        self.column_selector.addItems(numeric_cols)
        self.update_preview()