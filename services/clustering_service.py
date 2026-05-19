from __future__ import annotations

from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from core.clustering.clustering_models import ClusteringResult


class ClusteringService:
    SUPPORTED_METHODS = {"kmeans", "dbscan"}

    EXCLUDED_FEATURE_COLUMNS = {
        "segment_id",
        "start_idx",
        "end_idx",
        "source_start_idx",
        "source_end_idx",
        "start_time",
        "end_time",
        "duration",
        "cluster_id",
        "cluster",
        "label",
        "state",
        "source_chunk_id",
        "score",
        "is_boundary",
    }

    DEFAULT_SUFFIX_PRIORITY = ("mean", "std", "length")

    def run_clustering(
        self,
        segments_df: pd.DataFrame,
        method: str,
        selected_columns: List[str],
        params: Dict[str, Any],
        progress_callback=None,
        is_cancelled=None,
    ) -> ClusteringResult:
        self._check_cancel(is_cancelled)

        if segments_df is None or segments_df.empty:
            raise ValueError("Нет данных сегментации. Сначала выполните этап segmentation.")
        if "segment_id" not in segments_df.columns:
            raise ValueError("Для кластеризации требуется таблица сегментов (ожидается столбец segment_id).")
        if len(segments_df) < 2:
            raise ValueError("Для кластеризации нужно минимум 2 сегмента.")

        method_norm = (method or "").strip().lower()
        if method_norm not in self.SUPPORTED_METHODS:
            raise ValueError(f"Неподдерживаемый метод кластеризации: {method}")

        feature_df, selected_columns = self._prepare_feature_matrix(segments_df, selected_columns)
        X = self._scale_if_needed(feature_df.to_numpy(dtype=float), bool(params.get("scale", True)))

        if progress_callback:
            progress_callback.emit(25, "Данные для кластеризации подготовлены")
        self._check_cancel(is_cancelled)

        if method_norm == "kmeans":
            labels, distance_metric, normalized_params = self._run_kmeans(X, params)
        else:
            labels, distance_metric, normalized_params = self._run_dbscan(X, params)

        clustered_segments = segments_df.loc[feature_df.index].copy()
        clustered_segments["cluster_id"] = labels.astype(int)
        clustered_segments = self._sort_for_markov(clustered_segments)

        metrics = self._calculate_metrics(X, labels)
        warnings = self._build_warnings(labels, metrics, len(clustered_segments), method_norm)

        summary = self._build_summary(
            method=method_norm,
            labels=labels,
            selected_columns=selected_columns,
            params=normalized_params,
            metrics=metrics,
            total_segments=len(clustered_segments),
            warnings=warnings,
        )

        if progress_callback:
            progress_callback.emit(100, "Формирование результатов завершено")

        return ClusteringResult(
            method=method_norm,
            params=normalized_params,
            selected_columns=selected_columns,
            labels=labels.astype(int).tolist(),
            clustered_segments=clustered_segments,
            metrics=metrics,
            summary=summary,
            distance_metric=distance_metric,
            source_info={
                "input_table": "segments",
                "number_of_segments": len(clustered_segments),
            },
        )

    def evaluate_kmeans_range(
        self,
        segments_df: pd.DataFrame,
        selected_columns: List[str],
        params: Dict[str, Any],
        k_min: int = 2,
        k_max: int = 10,
    ) -> pd.DataFrame:
        if segments_df is None or segments_df.empty:
            raise ValueError("Нет данных сегментов для подбора k.")

        feature_df, selected_columns = self._prepare_feature_matrix(segments_df, selected_columns)
        X = self._scale_if_needed(feature_df.to_numpy(dtype=float), bool(params.get("scale", True)))

        n_samples = len(X)
        k_min = max(2, int(k_min))
        k_max = min(int(k_max), n_samples - 1)

        if k_max < k_min:
            raise ValueError(
                "Некорректный диапазон k: нужно минимум 3 валидных сегмента, чтобы посчитать метрики для k >= 2."
            )

        rows: List[Dict[str, Any]] = []
        random_state = int(params.get("random_state", 42))
        n_init = int(params.get("n_init", 10))

        for k in range(k_min, k_max + 1):
            labels = KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit_predict(X)
            metrics = self._calculate_metrics(X, labels)

            rows.append(
                {
                    "k": int(k),
                    "silhouette": metrics.get("silhouette"),
                    "calinski_harabasz": metrics.get("calinski_harabasz"),
                    "davies_bouldin": metrics.get("davies_bouldin"),
                    "inertia": float(KMeans(n_clusters=k, random_state=random_state, n_init=n_init).fit(X).inertia_),
                }
            )

        return pd.DataFrame(rows)

    @staticmethod
    def choose_best_k(evaluation_df: pd.DataFrame, metric: str = "silhouette") -> Optional[int]:
        if evaluation_df is None or evaluation_df.empty or metric not in evaluation_df.columns:
            return None

        valid = evaluation_df.dropna(subset=[metric])
        if valid.empty:
            return None

        if metric == "davies_bouldin":
            return int(valid.sort_values(metric, ascending=True).iloc[0]["k"])

        return int(valid.sort_values(metric, ascending=False).iloc[0]["k"])

    @staticmethod
    def _check_cancel(is_cancelled):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Задача отменена")

    def _prepare_feature_matrix(
        self,
        segments_df: pd.DataFrame,
        selected_columns: List[str],
    ) -> Tuple[pd.DataFrame, List[str]]:
        numeric_df = segments_df.select_dtypes(include="number")

        if numeric_df.empty:
            raise ValueError("В таблице сегментов нет числовых признаков для кластеризации.")

        if selected_columns:
            valid_columns = [c for c in selected_columns if c in numeric_df.columns]
        else:
            valid_columns = self.default_feature_columns(segments_df)

        valid_columns = [
            c for c in valid_columns
            if c in numeric_df.columns and c.lower() not in self.EXCLUDED_FEATURE_COLUMNS
        ]

        if not valid_columns:
            raise ValueError("Выбранные признаки недоступны или не являются числовыми.")

        feature_df = numeric_df[valid_columns].replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna(axis=0)

        constant_cols = [c for c in feature_df.columns if feature_df[c].nunique(dropna=True) <= 1]
        if constant_cols:
            feature_df = feature_df.drop(columns=constant_cols)
            valid_columns = [c for c in valid_columns if c not in constant_cols]

        if feature_df.empty:
            raise ValueError("После удаления NaN/inf и константных признаков не осталось данных для кластеризации.")
        if len(feature_df) < 2:
            raise ValueError("Слишком мало валидных сегментов после очистки признаков.")
        if not valid_columns:
            raise ValueError("Все выбранные признаки оказались константными.")

        return feature_df, valid_columns

    def default_feature_columns(self, segments_df: pd.DataFrame) -> List[str]:
        numeric_cols = [
            c for c in segments_df.select_dtypes(include="number").columns
            if c.lower() not in self.EXCLUDED_FEATURE_COLUMNS
        ]

        recommended: List[str] = []
        for col in numeric_cols:
            base, suffix = self.split_segment_feature_name(col)
            if col == "length" or suffix in {"mean", "std"}:
                recommended.append(col)

        return recommended or numeric_cols

    @staticmethod
    def split_segment_feature_name(column: str) -> Tuple[str, str]:
        if column == "length":
            return "segment", "length"

        known_suffixes = ("mean", "std", "min", "max", "median", "var", "variance", "skew", "kurtosis", "energy", "rms")
        for suffix in known_suffixes:
            marker = f"_{suffix}"
            if column.endswith(marker):
                return column[: -len(marker)], suffix

        return column, "value"

    @staticmethod
    def _scale_if_needed(X: np.ndarray, scale: bool) -> np.ndarray:
        return StandardScaler().fit_transform(X) if scale else X

    @staticmethod
    def _run_kmeans(X: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        n_clusters = int(params.get("n_clusters", 3))
        random_state = int(params.get("random_state", 42))
        n_init = int(params.get("n_init", 10))

        if n_clusters < 2:
            raise ValueError("Для KMeans параметр n_clusters должен быть >= 2.")
        if n_clusters > len(X):
            raise ValueError("Количество кластеров не может быть больше числа сегментов.")

        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit_predict(X)
        return (
            labels.astype(int),
            "euclidean",
            {
                "n_clusters": n_clusters,
                "random_state": random_state,
                "n_init": n_init,
                "scale": bool(params.get("scale", True)),
            },
        )

    @staticmethod
    def _run_dbscan(X: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        eps = float(params.get("eps", 0.5))
        min_samples = int(params.get("min_samples", 5))
        metric = str(params.get("metric", "euclidean"))

        if eps <= 0:
            raise ValueError("Для DBSCAN параметр eps должен быть > 0.")
        if min_samples < 1:
            raise ValueError("Для DBSCAN параметр min_samples должен быть >= 1.")

        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)
        return (
            labels.astype(int),
            metric,
            {
                "eps": eps,
                "min_samples": min_samples,
                "metric": metric,
                "scale": bool(params.get("scale", True)),
            },
        )

    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float | None]:
        mask = labels != -1
        X_eval = X[mask]
        y_eval = labels[mask]

        n_clusters = len(set(y_eval.tolist())) if len(y_eval) else 0
        if n_clusters < 2 or len(X_eval) < 2:
            return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}

        silhouette = None
        if n_clusters < len(X_eval):
            silhouette = float(silhouette_score(X_eval, y_eval))

        return {
            "silhouette": silhouette,
            "davies_bouldin": float(davies_bouldin_score(X_eval, y_eval)),
            "calinski_harabasz": float(calinski_harabasz_score(X_eval, y_eval)),
        }

    def build_pca_projection(self, clustered_segments: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        if len(clustered_segments) < 2 or len(feature_columns) < 2:
            return pd.DataFrame(columns=["PC1", "PC2", "cluster_id", "segment_id"])

        X = clustered_segments[feature_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        if len(X) < 2:
            return pd.DataFrame(columns=["PC1", "PC2", "cluster_id", "segment_id"])

        Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))
        pcs = PCA(n_components=2).fit_transform(Xs)

        out = pd.DataFrame(
            {
                "PC1": pcs[:, 0],
                "PC2": pcs[:, 1],
                "cluster_id": clustered_segments.loc[X.index, "cluster_id"].to_numpy(),
                "segment_id": clustered_segments.loc[X.index, "segment_id"].to_numpy()
                if "segment_id" in clustered_segments.columns
                else X.index.to_numpy(),
            }
        )
        return out.reset_index(drop=True)

    def build_feature_importance(
        self,
        clustered_segments: pd.DataFrame,
        feature_columns: List[str],
        top_n: int = 10,
        exclude_noise: bool = False,
    ) -> pd.DataFrame:
        df = clustered_segments.copy()
        if exclude_noise and "cluster_id" in df.columns:
            df = df[df["cluster_id"] != -1]

        if df.empty or "cluster_id" not in df.columns or len(set(df["cluster_id"].tolist())) < 2:
            return pd.DataFrame(columns=["feature", "score"])

        X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
        valid = X.dropna(axis=0).index
        X = X.loc[valid]
        y = df.loc[valid, "cluster_id"].to_numpy()

        if len(set(y.tolist())) < 2 or len(X) < 2:
            return pd.DataFrame(columns=["feature", "score"])

        scores, _ = f_classif(X.to_numpy(dtype=float), y)
        res = pd.DataFrame(
            {
                "feature": list(X.columns),
                "score": np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0),
            }
        )
        return res.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    @staticmethod
    def cluster_size_table(clustered_segments: pd.DataFrame) -> pd.DataFrame:
        if clustered_segments is None or clustered_segments.empty or "cluster_id" not in clustered_segments.columns:
            return pd.DataFrame(columns=["cluster_id", "count", "share"])

        counts = clustered_segments["cluster_id"].value_counts(dropna=False).sort_index()
        total = len(clustered_segments)
        return pd.DataFrame(
            {
                "cluster_id": counts.index.astype(int),
                "count": counts.values.astype(int),
                "share": [float(v / total) for v in counts.values],
            }
        )

    @staticmethod
    def _build_summary(
        method: str,
        labels: np.ndarray,
        selected_columns: List[str],
        params: Dict[str, Any],
        metrics: Dict[str, float | None],
        total_segments: int,
        warnings: List[str],
    ) -> Dict[str, Any]:
        series = pd.Series(labels, name="cluster_id")
        cluster_sizes = series.value_counts(dropna=False).sort_index().to_dict()
        unique_clusters = sorted(set(labels.tolist()))
        n_noise = int(cluster_sizes.get(-1, 0))

        return {
            "method": method,
            "number_of_segments": int(total_segments),
            "number_of_clusters": int(len([c for c in unique_clusters if c != -1])),
            "number_of_noise_points": n_noise,
            "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()},
            "selected_columns": list(selected_columns),
            "parameters": dict(params),
            "metrics": dict(metrics),
            "warnings": list(warnings),
        }

    @staticmethod
    def _build_warnings(
        labels: np.ndarray,
        metrics: Dict[str, float | None],
        total_segments: int,
        method: str,
    ) -> List[str]:
        warnings: List[str] = []
        unique = sorted(set(labels.tolist()))
        clusters = [c for c in unique if c != -1]
        n_noise = int((labels == -1).sum())

        if total_segments < 6:
            warnings.append("Сегментов очень мало. Метрики кластеризации могут быть нестабильными.")

        if method == "dbscan" and n_noise == total_segments:
            warnings.append("DBSCAN отнес все сегменты к шуму. Попробуйте увеличить eps или уменьшить min_samples.")

        if len(clusters) <= 1:
            warnings.append("Получился один кластер. Для Марковской модели переходов этого обычно недостаточно.")

        counts = pd.Series(labels).value_counts()
        if total_segments and not counts.empty and counts.max() / total_segments > 0.8 and len(clusters) > 1:
            warnings.append("Один кластер содержит более 80% сегментов. Возможно, признаки плохо разделяют состояния.")

        silh = metrics.get("silhouette")
        if silh is not None and silh < 0.05:
            warnings.append("Silhouette близок к нулю. Кластеры слабо отделены друг от друга.")

        return warnings

    @staticmethod
    def _sort_for_markov(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["start_idx", "start_time", "segment_id"]:
            if col in df.columns:
                return df.sort_values(col, kind="stable").reset_index(drop=True)
        return df.reset_index(drop=True)
