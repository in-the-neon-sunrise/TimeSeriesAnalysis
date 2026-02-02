import sqlite3
import pandas as pd
import pickle
from core.data_models.project_state import ProjectState

class ProjectRepository:

    def _connect(self, path: str):
        return sqlite3.connect(path)

    def init_db(self, path: str):
        conn = self._connect(path)
        cur = conn.cursor()

        cur.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value BLOB)")
        cur.execute("CREATE TABLE IF NOT EXISTS dataframes (name TEXT PRIMARY KEY, data BLOB)")

        conn.commit()
        conn.close()

    def _save_dataframe(self, conn, name: str, df: pd.DataFrame):
        blob = pickle.dumps(df)
        conn.execute("REPLACE INTO dataframes (name, data) VALUES (?, ?)", (name, blob))


    def _load_dataframe(self, conn, name: str) -> pd.DataFrame | None:
        cur = conn.execute(
            "SELECT data FROM dataframes WHERE name = ?", (name,)
        )
        row = cur.fetchone()
        if row is None:
            return None
        return pickle.loads(row[0])

    def save_project(self, state: ProjectState, path: str):
        self.init_db(path)
        conn = self._connect(path)

        # DataFrames
        mapping = {
            "raw_data": state.raw_data,
            "preprocessed_data": state.preprocessed_data,
            "features": state.features,
            "segments": state.segments,
            "clusters": state.clusters,
            "markov_matrix": state.markov_matrix,
        }

        for name, df in mapping.items():
            if df is not None:
                self._save_dataframe(conn, name, df)

        # параметры
        conn.execute(
            "REPLACE INTO metadata (key, value) VALUES (?, ?)",
            ("params", pickle.dumps(state.params))
        )

        conn.commit()
        conn.close()

    def load_project(self, path: str) -> ProjectState:
        conn = self._connect(path)

        state = ProjectState()
        state.file_path = path

        for name in [
            "raw_data",
            "preprocessed_data",
            "features",
            "segments",
            "clusters",
            "markov_matrix",
        ]:
            setattr(state, name, self._load_dataframe(conn, name))

        cur = conn.execute(
            "SELECT value FROM metadata WHERE key = 'params'"
        )
        row = cur.fetchone()
        if row:
            state.params = pickle.loads(row[0])

        state.is_dirty = False
        conn.close()
        return state