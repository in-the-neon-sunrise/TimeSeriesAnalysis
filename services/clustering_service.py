from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import StandardScaler

from core.clustering.clustering_models import ClusteringResult


class ClusteringService:
    SUPPORTED_METHODS = {"kmeans", "dbscan"}
    EXCLUDED_FEATURE_COLUMNS = {
        "segment_id", "start_idx", "end_idx", "start_time", "end_time", "duration", "cluster_id", "cluster",
        "label", "state", "source_chunk_id", "score",
    }

    def run_clustering(self, segments_df: pd.DataFrame, method: str, selected_columns: List[str],
                       params: Dict[str, Any], progress_callback=None, is_cancelled=None, ) -> ClusteringResult:
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
        X = self._scale_if_needed(feature_df.to_numpy(dtype=float), params.get("scale", False))
        if progress_callback:
            progress_callback.emit(25, "Данные для кластеризации подготовлены")


        if method_norm == "kmeans":
            labels, distance_metric, normalized_params = self._run_kmeans(X, params)
        else:
            labels, distance_metric, normalized_params = self._run_dbscan(X, params)

        clustered_segments = segments_df.loc[feature_df.index].copy()

        clustered_segments["cluster_id"] = labels
        clustered_segments = self._sort_for_markov(clustered_segments)

        metrics = self._calculate_metrics(X, labels)
        summary = self._build_summary(method_norm, labels, selected_columns, normalized_params, metrics, len(clustered_segments))
        if progress_callback:
            progress_callback.emit(100, "Формирование результатов завершено")

        return ClusteringResult(method=method_norm, params=normalized_params, selected_columns=selected_columns,
                                labels=labels.tolist(), clustered_segments=clustered_segments, metrics=metrics,
                                summary=summary,
                                distance_metric=distance_metric, source_info={"input_table": "segments",
                                                                              "number_of_segments": len(
                                                                                  clustered_segments)}, )

    @staticmethod
    def _check_cancel(is_cancelled):
        if is_cancelled and is_cancelled():
            raise RuntimeError("Задача отменена")

    def _prepare_feature_matrix(self, segments_df: pd.DataFrame, selected_columns: List[str]) -> Tuple[pd.DataFrame, List[str]]:
        numeric_df = segments_df.select_dtypes(include="number")
        if numeric_df.empty:
            raise ValueError("В таблице сегментов нет числовых признаков для кластеризации.")

        if selected_columns:
            valid_columns = [c for c in selected_columns if c in numeric_df.columns]
        else:
            valid_columns = list(numeric_df.columns)

        if not valid_columns:
            raise ValueError("Выбранные признаки недоступны или не являются числовыми.")

        feature_df = numeric_df[valid_columns].replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.dropna(axis=0)
        if feature_df.empty:
            raise ValueError("После удаления NaN/inf не осталось валидных строк для кластеризации.")

        if len(feature_df) < 2:
            raise ValueError("Слишком мало валидных сегментов после очистки признаков.")

        return feature_df, valid_columns

    @staticmethod
    def _scale_if_needed(X: np.ndarray, scale: bool) -> np.ndarray:
        return StandardScaler().fit_transform(X) if scale else X

    @staticmethod
    def _run_kmeans(X: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        n_clusters = int(params.get("n_clusters", 3)); random_state = int(params.get("random_state", 42)); n_init = int(params.get("n_init", 10))
        if n_clusters < 2: raise ValueError("Для KMeans параметр n_clusters должен быть >= 2.")
        if n_clusters > len(X): raise ValueError("Количество кластеров не может быть больше числа сегментов.")
        labels = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init).fit_predict(X)
        return labels.astype(int), "euclidean", {"n_clusters": n_clusters, "random_state": random_state, "n_init": n_init, "scale": bool(params.get("scale", False)), "use_length": bool(params.get("use_length", False))}

    @staticmethod
    def _run_dbscan(X: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, str, Dict[str, Any]]:
        eps = float(params.get("eps", 0.5));
        min_samples = int(params.get("min_samples", 5));
        metric = str(params.get("metric", "euclidean"))
        if eps <= 0: raise ValueError("Для DBSCAN параметр eps должен быть > 0.")
        if min_samples < 1: raise ValueError("Для DBSCAN параметр min_samples должен быть >= 1.")
        labels = DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit_predict(X)
        return labels.astype(int), metric, {"eps": eps, "min_samples": min_samples, "metric": metric,
                                            "scale": bool(params.get("scale", False)),
                                            "use_length": bool(params.get("use_length", False))}

    def _calculate_metrics(self, X: np.ndarray, labels: np.ndarray) -> Dict[str, float | None]:
        mask = labels != -1
        X_eval = X[mask]; y_eval = labels[mask]
        n_clusters = len(set(y_eval.tolist())) if len(y_eval) else 0
        if n_clusters < 2 or len(X_eval) < 2:
            return {"silhouette": None, "davies_bouldin": None, "calinski_harabasz": None}
        silhouette = None if n_clusters >= len(X_eval) else float(silhouette_score(X_eval, y_eval))
        return {"silhouette": silhouette, "davies_bouldin": float(davies_bouldin_score(X_eval, y_eval)), "calinski_harabasz": float(calinski_harabasz_score(X_eval, y_eval))}

    def build_pca_projection(self, clustered_segments: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        if len(clustered_segments) < 2 or len(feature_columns) < 2:
            return pd.DataFrame(columns=["PC1", "PC2", "cluster_id"])
        X = clustered_segments[feature_columns].replace([np.inf, -np.inf], np.nan).dropna(axis=0)
        if len(X) < 2:
            return pd.DataFrame(columns=["PC1", "PC2", "cluster_id"])
        Xs = StandardScaler().fit_transform(X.to_numpy(dtype=float))
        pcs = PCA(n_components=2).fit_transform(Xs)
        out = pd.DataFrame({"PC1": pcs[:, 0], "PC2": pcs[:, 1], "cluster_id": clustered_segments.loc[X.index, "cluster_id"].to_numpy()})
        return out.reset_index(drop=True)

    def build_feature_importance(self, clustered_segments: pd.DataFrame, feature_columns: List[str], top_n: int = 10, exclude_noise: bool = False) -> pd.DataFrame:
        df = clustered_segments.copy()
        if exclude_noise:
            df = df[df["cluster_id"] != -1]
        if df.empty or len(set(df["cluster_id"].tolist())) < 2:
            return pd.DataFrame(columns=["feature", "score"])
        X = df[feature_columns].replace([np.inf, -np.inf], np.nan)
        valid = X.dropna(axis=0).index
        X = X.loc[valid]
        y = df.loc[valid, "cluster_id"].to_numpy()
        if len(set(y.tolist())) < 2 or len(X) < 2:
            return pd.DataFrame(columns=["feature", "score"])
        scores, _ = f_classif(X.to_numpy(dtype=float), y)
        res = pd.DataFrame({"feature": feature_columns, "score": np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)})
        return res.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)

    @staticmethod
    def _build_summary(method: str, labels: np.ndarray, selected_columns: List[str], params: Dict[str, Any], metrics: Dict[str, float | None], total_segments: int,) -> Dict[str, Any]:
        series = pd.Series(labels, name="cluster_id")
        cluster_sizes = series.value_counts(dropna=False).sort_index().to_dict()
        unique_clusters = sorted(set(labels.tolist()))
        n_noise = int(cluster_sizes.get(-1, 0))
        return {"method": method, "number_of_segments": int(total_segments), "number_of_clusters": int(len([c for c in unique_clusters if c != -1])), "number_of_noise_points": n_noise, "cluster_sizes": {str(k): int(v) for k, v in cluster_sizes.items()}, "selected_columns": list(selected_columns), "parameters": dict(params), "metrics": dict(metrics)}

    @staticmethod
    def _sort_for_markov(df: pd.DataFrame) -> pd.DataFrame:
        for col in ["start_idx", "start_time", "segment_id"]:
            if col in df.columns:
                return df.sort_values(col, kind="stable").reset_index(drop=True)
        return df.reset_index(drop=True)
