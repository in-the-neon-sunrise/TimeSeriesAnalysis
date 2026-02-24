from PySide6.QtWidgets import QLabel, QVBoxLayout, QPushButton
from ui.pages.base_page import BasePage
from services.report_service import ReportService
from PySide6.QtWidgets import QFileDialog, QMessageBox

class ReportPage(BasePage):
    def __init__(self, project):
        super().__init__()
        self.project = project
        self.report_btn = QPushButton("Создать отчет")
        self.report_btn.clicked.connect(self.on_report_clicked)

        layout = QVBoxLayout(self)
        title = QLabel("Отчет")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")

        layout.addWidget(title)
        layout.addWidget(self.report_btn)
        layout.addStretch()

    def on_report_clicked(self):
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить отчёт",
            "report.pdf",
            "PDF (*.pdf)"
        )
        if not file_path:
            return

        ReportService.generate_test_report(file_path)
        QMessageBox.information(
            self,
            "Готово",
            "PDF-отчёт успешно сгенерирован"
        )
