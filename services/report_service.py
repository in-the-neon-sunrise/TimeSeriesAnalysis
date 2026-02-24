import os
import random
import tempfile
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.pdfbase.ttfonts import TTFont

class ReportService:
    @staticmethod
    def generate_test_report(
        output_path: str,
        data_rows_count: int | None = None
    ):
        base_dir = os.path.dirname(os.path.dirname(__file__))

        font_path = os.path.join(base_dir, "assets", "DejaVuSans.ttf")
        pdfmetrics.registerFont(TTFont("DejaVu", font_path))

        styles = getSampleStyleSheet()

        styles["Normal"].fontName = "DejaVu"
        styles["Title"].fontName = "DejaVu"

        styles.add(
            ParagraphStyle(
                name="Header",
                fontName="DejaVu",
                fontSize=14,
                spaceAfter=10
            )
        )

        x = np.linspace(0, 10, 100)
        y = np.cos(x) + np.random.normal(scale=0.3, size=len(x))

        plot_path = os.path.join(tempfile.gettempdir(), "test_plot.png")

        plt.figure()
        plt.plot(x, y)
        plt.title("Случайный график")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()

        doc = SimpleDocTemplate(output_path, pagesize=A4)
        story = []

        story.append(Paragraph(
            "Отчёт анализа временного ряда",
            styles["Title"]
        ))
        story.append(Spacer(1, 12))

        story.append(Paragraph(
            f"Дата генерации: {datetime.now().strftime('%d.%m.%Y %H:%M')}",
            styles["Normal"]
        ))
        story.append(Spacer(1, 12))

        if data_rows_count is not None:
            story.append(Paragraph(
                "Информация о входных данных",
                styles["Header"]
            ))
            story.append(Paragraph(
                f"Количество строк во входном наборе данных: <b>{data_rows_count}</b>",
                styles["Normal"]
            ))
            story.append(Spacer(1, 12))

        story.append(Paragraph(
            "Штирлиц долго смотрел в одну точку. Потом в другую. Двоеточие — догадался Штирлиц.",
            styles["Normal"]
        ))
        story.append(Spacer(1, 20))

        image_path = os.path.join(base_dir, "assets", "cat_image.jpg" )

        if os.path.exists(image_path):
            story.append(Paragraph(
                "Изображение:",
                styles["Header"]
            ))
            story.append(Spacer(1, 8))
            story.append(Image(image_path, width=200, height=200))
            story.append(Spacer(1, 20))

        story.append(Paragraph(
            "График:",
            styles["Header"]
        ))
        story.append(Spacer(1, 8))
        story.append(Image(plot_path, width=400, height=250))

        doc.build(story)

        return output_path