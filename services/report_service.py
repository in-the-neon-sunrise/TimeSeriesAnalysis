from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from services.data_statistics_service import DataStatisticsService


@dataclass
class ReportGenerationOptions:
    title: str = "Отчёт по анализу временного ряда"
    author: str = ""
    include_data_overview: bool = True
    include_primary_analysis: bool = True
    include_preprocessing: bool = True
    include_feature_extraction: bool = True
    include_segmentation: bool = True
    include_clustering: bool = True
    include_markov_modeling: bool = True
    include_plots: bool = True
    include_tables: bool = True
    include_summary: bool = True

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ReportGenerationOptions":
        values = {k: payload.get(k, getattr(cls, k)) for k in cls.__dataclass_fields__}
        return cls(**values)


class ReportService:
    FONT_NAME = "DejaVu"

    def __init__(self, project_service):
        self.project = project_service
        self._styles = None

    def generate_report(self, output_path: str, options: ReportGenerationOptions) -> str:
        if not output_path:
            raise ValueError("Не выбран путь для сохранения отчёта.")

        output = Path(output_path)
        if output.suffix.lower() != ".pdf":
            output = output.with_suffix(".pdf")

        self._register_fonts()
        self._styles = self._build_styles()

        output.parent.mkdir(parents=True, exist_ok=True)

        temp_dir = Path(tempfile.mkdtemp(prefix="report_assets_"))
        doc = SimpleDocTemplate(str(output), pagesize=A4, title=options.title)
        story: List[Any] = []

        self._build_title_section(story, options, output)

        if options.include_data_overview:
            self._build_data_section(story, options, temp_dir)
        if options.include_primary_analysis:
            self._build_primary_analysis_section(story, options, temp_dir)
        if options.include_preprocessing:
            self._build_preprocessing_section(story)
        if options.include_feature_extraction:
            self._build_features_section(story, options, temp_dir)
        if options.include_segmentation:
            self._build_segmentation_section(story, options, temp_dir)
        if options.include_clustering:
            self._build_clustering_section(story, options, temp_dir)
        if options.include_markov_modeling:
            self._build_markov_section(story, options, temp_dir)
        if options.include_summary:
            self._build_summary_section(story, output)

        doc.build(story)
        self._cleanup_temp_images(temp_dir)

        generated_at = datetime.now().isoformat(timespec="seconds")
        self.project.parameters["report"] = {
            "last_report_path": str(output),
            "generated_at": generated_at,
            "options": options.__dict__,
        }
        if hasattr(self.project, "last_report_path"):
            self.project.last_report_path = str(output)
        if hasattr(self.project, "last_report_generated_at"):
            self.project.last_report_generated_at = generated_at

        return str(output)

    def _register_fonts(self):
        base_dir = Path(__file__).resolve().parent.parent
        font_path = base_dir / "assets" / "DejaVuSans.ttf"
        if not font_path.exists():
            raise FileNotFoundError(
                "Не найден шрифт для кириллицы: assets/DejaVuSans.ttf. "
                "Добавьте TTF-файл в assets/fonts или assets."
            )
        pdfmetrics.registerFont(TTFont(self.FONT_NAME, str(font_path)))

    def _build_styles(self):
        styles = getSampleStyleSheet()
        styles["Normal"].fontName = self.FONT_NAME
        styles["Title"].fontName = self.FONT_NAME
        styles.add(ParagraphStyle(name="Section", fontName=self.FONT_NAME, fontSize=14, spaceAfter=8, spaceBefore=12))
        styles.add(ParagraphStyle(name="Small", fontName=self.FONT_NAME, fontSize=9, leading=12))
        return styles

    def _build_title_section(self, story: List[Any], options: ReportGenerationOptions, output_path: Path):
        project_name = self.project.file_path or "Без названия"
        story.append(Paragraph("Приложение для автоматизированного анализа временных рядов", self._styles["Title"]))
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>{options.title}</b>", self._styles["Normal"]))
        story.append(Paragraph(f"Дата формирования: {datetime.now().strftime('%d.%m.%Y %H:%M')}", self._styles["Normal"]))
        story.append(Paragraph(f"Проект: {project_name}", self._styles["Normal"]))
        if options.author:
            story.append(Paragraph(f"Автор: {options.author}", self._styles["Normal"]))
        story.append(Paragraph(f"Отчёт сформирован автоматически и сохранён в: {output_path}", self._styles["Small"]))
        story.append(Spacer(1, 12))

    def _build_data_section(self, story: List[Any], options: ReportGenerationOptions, temp_dir: Path):
        story.append(Paragraph("1. Информация о данных", self._styles["Section"]))
        df = self.project.raw_data
        if df is None or df.empty:
            story.append(Paragraph("Этап не выполнен: исходные данные не загружены.", self._styles["Normal"]))
            return

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        missing_total = int(df.isna().sum().sum())
        time_col = self._guess_time_column(df)
        story.extend(
            [
                Paragraph(f"Количество строк: {len(df)}", self._styles["Normal"]),
                Paragraph(f"Количество столбцов: {df.shape[1]}", self._styles["Normal"]),
                Paragraph(f"Числовых столбцов: {len(numeric_cols)}", self._styles["Normal"]),
                Paragraph(f"Пропусков всего: {missing_total}", self._styles["Normal"]),
                Paragraph(f"Временная колонка: {time_col or 'не определена'}", self._styles["Normal"]),
                Paragraph(f"Список столбцов: {', '.join(map(str, df.columns.tolist()))}", self._styles["Small"]),
            ]
        )

        if options.include_tables:
            dtypes_df = pd.DataFrame({"column": df.columns, "dtype": df.dtypes.astype(str)})
            self._add_dataframe_table(story, dtypes_df, "Типы данных", max_rows=20)

        if options.include_plots:
            plot_path = self._plot_series_preview(df, temp_dir)
            if plot_path:
                self._add_plot_image(story, plot_path, "Превью временного ряда")

    def _build_primary_analysis_section(self, story: List[Any], options: ReportGenerationOptions, temp_dir: Path):
        story.append(Paragraph("2. Первичный анализ", self._styles["Section"]))
        df = self.project.raw_data
        if df is None or df.empty:
            story.append(Paragraph("Этап не выполнен.", self._styles["Normal"]))
            return

        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if not numeric_cols:
            story.append(Paragraph("Нет числовых столбцов для первичного анализа.", self._styles["Normal"]))
            return

        selected_col = numeric_cols[0]
        series = df[selected_col].dropna()
        stats = DataStatisticsService.descriptive_statistics(series)
        adf = DataStatisticsService.stationarity_adf(series)
        outliers = DataStatisticsService.detect_outliers_iqr(series)

        for k, v in stats.items():
            story.append(Paragraph(f"{k}: {v:.5f}" if isinstance(v, (int, float, np.floating)) else f"{k}: {v}", self._styles["Normal"]))

        story.append(Paragraph(f"ADF p-value: {adf['p_value']:.5f} | стационарность: {'ДА' if adf['is_stationary'] else 'НЕТ'}", self._styles["Normal"]))
        story.append(Paragraph(f"Выбросы (IQR): {int(outliers['count'])}", self._styles["Normal"]))
        story.append(Paragraph(f"Пропуски (доля): {df[selected_col].isna().mean():.2%}", self._styles["Normal"]))

        if options.include_tables:
            desc_df = df[numeric_cols].describe().transpose().reset_index().rename(columns={"index": "column"})
            self._add_dataframe_table(story, desc_df, "Descriptive statistics", max_rows=20)

        if options.include_plots:
            acf_path = self._plot_acf(series, temp_dir)
            if acf_path:
                self._add_plot_image(story, acf_path, f"ACF ({selected_col})")

    def _build_preprocessing_section(self, story: List[Any]):
        story.append(Paragraph("3. Preprocessing", self._styles["Section"]))
        processed = self.project.processed_data
        params = self.project.parameters.get("preprocessing", {})

        if processed is None or processed.empty:
            story.append(Paragraph("Этап не выполнен.", self._styles["Normal"]))
            return

        story.append(Paragraph(f"Размер обработанных данных: {processed.shape[0]} x {processed.shape[1]}", self._styles["Normal"]))
        if params:
            for key, value in params.items():
                story.append(Paragraph(f"{key}: {value}", self._styles["Small"]))
        else:
            story.append(Paragraph("Параметры preprocessing не сохранены.", self._styles["Small"]))

    def _build_features_section(self, story: List[Any], options: ReportGenerationOptions, temp_dir: Path):
        story.append(Paragraph("4. Feature extraction", self._styles["Section"]))
        features = self.project.features
        params = self.project.parameters.get("features", {})

        if features is None or features.empty:
            story.append(Paragraph("Этап не выполнен.", self._styles["Normal"]))
            return

        story.append(Paragraph(f"Размер feature matrix: {features.shape[0]} x {features.shape[1]}", self._styles["Normal"]))
        if params:
            story.append(Paragraph(f"Исходные столбцы: {params.get('selected_columns', [])}", self._styles["Small"]))
            story.append(Paragraph(f"Sliding window: window={params.get('window_size')} step={params.get('step_size')}", self._styles["Small"]))
            story.append(Paragraph(f"Признаки: {params.get('selected_features', [])}", self._styles["Small"]))

        if options.include_tables:
            self._add_dataframe_table(story, features, "Preview признаков", max_rows=10, max_cols=8)

        if options.include_plots:
            corr_path = self._plot_correlation(features, temp_dir)
            if corr_path:
                self._add_plot_image(story, corr_path, "Корреляционная матрица")

    def _build_segmentation_section(self, story: List[Any], options: ReportGenerationOptions, temp_dir: Path):
        story.append(Paragraph("5. Segmentation (SDA)", self._styles["Section"]))
        segments = self.project.segments
        seg_params = self.project.parameters.get("segmentation", {})
        seg_result = self.project.parameters.get("segmentation_result", {})

        if segments is None or segments.empty:
            story.append(Paragraph("Этап не выполнен.", self._styles["Normal"]))
            return

        summary = seg_result.get("summary", {})
        best = seg_result.get("best_result", {})
        story.append(Paragraph(f"Количество состояний/сегментов: {summary.get('n_segments', segments['stage_id'].nunique() if 'stage_id' in segments.columns else 'n/a')}", self._styles["Normal"]))
        story.append(Paragraph(f"Параметры запуска: {seg_params}", self._styles["Small"]))
        if best:
            story.append(Paragraph(f"Лучший результат (фрагмент): {best}", self._styles["Small"]))

        if options.include_tables:
            table_df = self._segments_table_from_data(segments)
            self._add_dataframe_table(story, table_df, "Таблица сегментов", max_rows=20)

        if options.include_plots:
            plot_path = self._plot_segmentation(segments, temp_dir)
            if plot_path:
                self._add_plot_image(story, plot_path, "Границы сегментации")

    def _build_clustering_section(self, story: List[Any], options: ReportGenerationOptions, temp_dir: Path):
        story.append(Paragraph("6. Clustering", self._styles["Section"]))
        clusters = self.project.clusters
        result = self.project.clustering_result or {}
        params = self.project.parameters.get("clustering", {})

        if clusters is None or clusters.empty:
            story.append(Paragraph("Этап не выполнен.", self._styles["Normal"]))
            return

        summary = result.get("summary", {})
        metrics = result.get("metrics", {})
        story.append(Paragraph(f"Метод: {result.get('method', params.get('method', 'unknown'))}", self._styles["Normal"]))
        story.append(Paragraph(f"Параметры: {result.get('params', params.get('params', {}))}", self._styles["Small"]))
        story.append(Paragraph(f"Выбранные признаки: {result.get('selected_columns', params.get('selected_columns', []))}", self._styles["Small"]))
        story.append(Paragraph(f"Число кластеров: {summary.get('number_of_clusters', clusters['cluster_id'].nunique() if 'cluster_id' in clusters.columns else 'n/a')}", self._styles["Normal"]))
        story.append(Paragraph(f"Размеры кластеров: {summary.get('cluster_sizes', {})}", self._styles["Small"]))
        story.append(Paragraph(f"Silhouette: {metrics.get('silhouette')}", self._styles["Normal"]))
        story.append(Paragraph(f"Davies-Bouldin: {metrics.get('davies_bouldin')}", self._styles["Normal"]))
        story.append(Paragraph(f"Calinski-Harabasz: {metrics.get('calinski_harabasz')}", self._styles["Normal"]))

        if options.include_tables:
            preview_cols = [c for c in ["segment_id", "start_idx", "end_idx", "cluster_id"] if c in clusters.columns]
            table_df = clusters[preview_cols] if preview_cols else clusters
            self._add_dataframe_table(story, table_df, "Сегменты с cluster_id", max_rows=20)

        if options.include_plots:
            path = self._plot_clustering(clusters, temp_dir)
            if path:
                self._add_plot_image(story, path, "Scatter кластеров")

    def _build_markov_section(self, story: List[Any], options: ReportGenerationOptions, temp_dir: Path):
        story.append(Paragraph("7. Markov modeling", self._styles["Section"]))
        markov_result = self.project.markov_result or {}
        params = self.project.parameters.get("markov", {})

        transition_counts = markov_result.get("transition_counts") if isinstance(markov_result, dict) else None
        transition_probs = markov_result.get("transition_probabilities") if isinstance(markov_result, dict) else None
        summary = markov_result.get("summary", {}) if isinstance(markov_result, dict) else {}

        if transition_probs is None and self.project.markov_matrix is None:
            story.append(Paragraph("Этап не выполнен.", self._styles["Normal"]))
            return

        probs_df = transition_probs if isinstance(transition_probs, pd.DataFrame) else self.project.markov_matrix
        if not isinstance(probs_df, pd.DataFrame) or probs_df.empty:
            story.append(Paragraph("Матрица переходов пуста.", self._styles["Normal"]))
            return

        story.append(Paragraph(f"Порядок цепи: {params.get('order', markov_result.get('order', 1))}", self._styles["Normal"]))
        story.append(Paragraph(f"Количество состояний: {len(probs_df.columns)}", self._styles["Normal"]))
        story.append(Paragraph(f"Количество переходов: {summary.get('observed_transitions', 'n/a')}", self._styles["Normal"]))
        story.append(Paragraph(f"Entropy: {summary.get('weighted_entropy', 'n/a')}", self._styles["Normal"]))

        top_transitions = markov_result.get("transitions_long_table") if isinstance(markov_result, dict) else None
        if isinstance(top_transitions, pd.DataFrame) and not top_transitions.empty:
            head = top_transitions.head(5)
            txt = "; ".join(
                f"{row['history_state']}→{row['next_state']} ({row['probability']:.3f})"
                for _, row in head.iterrows()
            )
            story.append(Paragraph(f"Наиболее вероятные переходы: {txt}", self._styles["Small"]))

        if options.include_tables:
            if isinstance(transition_counts, pd.DataFrame) and not transition_counts.empty:
                self._add_dataframe_table(story, transition_counts.reset_index(), "Transition counts", max_rows=12, max_cols=8)
            self._add_dataframe_table(story, probs_df.reset_index(), "Transition probabilities", max_rows=12, max_cols=8)

        if options.include_plots:
            heatmap = self._plot_markov_heatmap(probs_df, temp_dir)
            if heatmap:
                self._add_plot_image(story, heatmap, "Heatmap матрицы переходов")

    def _build_summary_section(self, story: List[Any], output_path: Path):
        story.append(Paragraph("8. Заключение", self._styles["Section"]))
        completed = {
            "data": self.project.raw_data is not None,
            "preprocessing": self.project.processed_data is not None,
            "features": self.project.features is not None,
            "segmentation": self.project.segments is not None,
            "clustering": self.project.clusters is not None,
            "markov": self.project.markov_result is not None or self.project.markov_matrix is not None,
        }
        done = [k for k, v in completed.items() if v]
        story.append(Paragraph(f"Выполненные этапы: {', '.join(done) if done else 'нет'}", self._styles["Normal"]))

        if self.project.raw_data is not None:
            story.append(Paragraph(f"Обработано наблюдений: {len(self.project.raw_data)}", self._styles["Normal"]))
        if self.project.segments is not None and "stage_id" in self.project.segments.columns:
            story.append(Paragraph(f"Получено сегментов: {self.project.segments['stage_id'].nunique()}", self._styles["Normal"]))
        if self.project.clusters is not None and "cluster_id" in self.project.clusters.columns:
            story.append(Paragraph(f"Получено кластеров: {self.project.clusters['cluster_id'].nunique()}", self._styles["Normal"]))

        story.append(Paragraph(f"Отчёт сохранён: {output_path}", self._styles["Small"]))

    def _add_dataframe_table(
        self,
        story: List[Any],
        df: pd.DataFrame,
        title: str,
        max_rows: int = 10,
        max_cols: int = 6,
    ):
        if df is None or df.empty:
            story.append(Paragraph(f"{title}: данных нет.", self._styles["Small"]))
            return

        story.append(Paragraph(title, self._styles["Normal"]))
        preview = df.iloc[:max_rows, :max_cols].copy()
        if len(df.columns) > max_cols:
            preview["..."] = ""

        data = [list(preview.columns)]
        for row in preview.itertuples(index=False):
            formatted = []
            for val in row:
                if isinstance(val, float):
                    formatted.append(f"{val:.5f}")
                else:
                    formatted.append(str(val))
            data.append(formatted)

        table = Table(data, repeatRows=1)
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
                    ("FONTNAME", (0, 0), (-1, -1), self.FONT_NAME),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("GRID", (0, 0), (-1, -1), 0.25, colors.grey),
                ]
            )
        )
        story.append(table)
        if len(df) > max_rows:
            story.append(Paragraph(f"Показаны первые {max_rows} строк из {len(df)}.", self._styles["Small"]))
        story.append(Spacer(1, 10))

    def _add_plot_image(self, story: List[Any], image_path: str, title: str):
        if not os.path.exists(image_path):
            return
        story.append(Paragraph(title, self._styles["Normal"]))
        story.append(Image(image_path, width=460, height=220))
        story.append(Spacer(1, 10))

    @staticmethod
    def _guess_time_column(df: pd.DataFrame) -> Optional[str]:
        candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
        if candidates:
            return candidates[0]
        return None

    @staticmethod
    def _cleanup_temp_images(temp_dir: Path):
        if not temp_dir.exists():
            return
        for file in temp_dir.glob("*.png"):
            try:
                file.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            temp_dir.rmdir()
        except OSError:
            pass

    def _plot_series_preview(self, df: pd.DataFrame, temp_dir: Path) -> Optional[str]:
        numeric = df.select_dtypes(include="number")
        if numeric.empty:
            return None
        series = numeric.iloc[:, 0].dropna()
        if series.empty:
            return None

        plt.figure(figsize=(8, 3))
        plt.plot(series.values)
        plt.title("Time series preview")
        plt.tight_layout()
        path = temp_dir / "series_preview.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return str(path)

    def _plot_acf(self, series: pd.Series, temp_dir: Path) -> Optional[str]:
        if series.empty or len(series) < 5:
            return None
        acf_values = DataStatisticsService.autocorrelation(series, lags=min(40, max(1, len(series) // 2)))
        plt.figure(figsize=(8, 3))
        plt.stem(acf_values)
        plt.title("Autocorrelation function")
        plt.tight_layout()
        path = temp_dir / "acf.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return str(path)

    def _plot_correlation(self, features: pd.DataFrame, temp_dir: Path) -> Optional[str]:
        numeric = features.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return None

        corr = numeric.iloc[:, :20].corr()
        plt.figure(figsize=(8, 4))
        plt.imshow(corr, aspect="auto", cmap="viridis")
        plt.colorbar()
        plt.title("Feature correlation")
        plt.tight_layout()
        path = temp_dir / "feature_corr.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return str(path)

    def _plot_segmentation(self, segments: pd.DataFrame, temp_dir: Path) -> Optional[str]:
        if "stage_id" not in segments.columns:
            return None
        numeric_cols = segments.select_dtypes(include="number").columns.tolist()
        feature_cols = [c for c in numeric_cols if c not in {"stage_id", "source_index", "is_boundary"}]
        if not feature_cols:
            return None

        y = segments[feature_cols[0]].values
        plt.figure(figsize=(8, 3))
        plt.plot(y, label=feature_cols[0])
        boundaries = segments.index[segments.get("is_boundary", False) == True].tolist() if "is_boundary" in segments.columns else []
        for idx in boundaries:
            plt.axvline(idx, color="red", alpha=0.4)
        plt.title("Segmentation boundaries")
        plt.tight_layout()
        path = temp_dir / "segmentation.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return str(path)

    def _plot_clustering(self, clusters: pd.DataFrame, temp_dir: Path) -> Optional[str]:
        if "cluster_id" not in clusters.columns:
            return None
        numeric = clusters.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            return None

        x = numeric.iloc[:, 0]
        y = numeric.iloc[:, 1]
        labels = clusters["cluster_id"]

        plt.figure(figsize=(7, 4))
        plt.scatter(x, y, c=labels, cmap="tab10", s=28)
        plt.title("Cluster scatter")
        plt.tight_layout()
        path = temp_dir / "clusters.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return str(path)

    def _plot_markov_heatmap(self, probs_df: pd.DataFrame, temp_dir: Path) -> Optional[str]:
        if probs_df.empty:
            return None

        matrix = probs_df.to_numpy(dtype=float)
        plt.figure(figsize=(7, 4))
        plt.imshow(matrix, aspect="auto", cmap="magma")
        plt.colorbar(label="P")
        plt.title("Markov transition probabilities")
        plt.xlabel("next_state")
        plt.ylabel("history_state")
        plt.tight_layout()
        path = temp_dir / "markov_heatmap.png"
        plt.savefig(path, dpi=120)
        plt.close()
        return str(path)

    @staticmethod
    def _segments_table_from_data(segments: pd.DataFrame) -> pd.DataFrame:
        if "stage_id" not in segments.columns:
            return segments.head(20)
        return (
            segments.groupby("stage_id", sort=True)
            .agg(start_idx=("source_index", "min") if "source_index" in segments.columns else ("stage_id", "size"),
                 end_idx=("source_index", "max") if "source_index" in segments.columns else ("stage_id", "size"),
                 size=("stage_id", "size"))
            .reset_index()
        )