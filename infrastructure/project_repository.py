import sqlite3
import json
import pandas as pd


class ProjectRepository:

    def save(self, project, file_path: str):
        conn = sqlite3.connect(file_path)

        # Сохраняем таблицы
        if project.raw_data is not None:
            project.raw_data.to_sql("raw_data", conn, if_exists="replace", index=False)

        if project.processed_data is not None:
            project.processed_data.to_sql("processed_data", conn, if_exists="replace", index=False)

        if project.features is not None:
            project.features.to_sql("features", conn, if_exists="replace", index=False)

        if project.segments is not None:
            project.segments.to_sql("segments", conn, if_exists="replace", index=False)

        if project.clusters is not None:
            project.clusters.to_sql("clusters", conn, if_exists="replace", index=False)

        if project.markov_matrix is not None:
            project.markov_matrix.to_sql("markov_matrix", conn, if_exists="replace", index=True, index_label="history_state")

        if project.markov_result:
            counts = project.markov_result.get("transition_counts")
            if isinstance(counts, pd.DataFrame):
                counts.to_sql("markov_transition_counts", conn, if_exists="replace", index=True, index_label="history_state")

            probs = project.markov_result.get("transition_probabilities")
            if isinstance(probs, pd.DataFrame):
                probs.to_sql("markov_transition_probabilities", conn, if_exists="replace", index=True, index_label="history_state")

            transitions = project.markov_result.get("transitions_long_table")
            if isinstance(transitions, pd.DataFrame):
                transitions.to_sql("markov_transitions_long", conn, if_exists="replace", index=False)

        # Сохраняем параметры как JSON
        params_json = json.dumps(project.parameters)

        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT
            )
        """
        )

        conn.execute("DELETE FROM metadata")

        conn.execute(
            "INSERT INTO metadata (key, value) VALUES (?, ?)",
            ("parameters", params_json)
        )

        conn.commit()
        conn.close()

    def load(self, file_path: str, project):
        conn = sqlite3.connect(file_path)

        tables = self._get_tables(conn)

        if "raw_data" in tables:
            project.raw_data = pd.read_sql("SELECT * FROM raw_data", conn)

        if "processed_data" in tables:
            project.processed_data = pd.read_sql("SELECT * FROM processed_data", conn)

        if "features" in tables:
            project.features = pd.read_sql("SELECT * FROM features", conn)

        if "segments" in tables:
            project.segments = pd.read_sql("SELECT * FROM segments", conn)

        if "clusters" in tables:
            project.clusters = pd.read_sql("SELECT * FROM clusters", conn)

        if "markov_matrix" in tables:
            markov_matrix = pd.read_sql("SELECT * FROM markov_matrix", conn)
            if "history_state" in markov_matrix.columns:
                markov_matrix = markov_matrix.set_index("history_state")
            project.markov_matrix = markov_matrix

        self._load_markov_result(conn, tables, project)

        # Загружаем параметры
        if "metadata" in tables:
            cursor = conn.execute("SELECT value FROM metadata WHERE key='parameters'")
            row = cursor.fetchone()
            if row:
                project.parameters = json.loads(row[0])

        conn.close()

    def _load_markov_result(self, conn, tables, project):
        markov_result = {}

        if "markov_transition_counts" in tables:
            counts = pd.read_sql("SELECT * FROM markov_transition_counts", conn)
            if "history_state" in counts.columns:
                counts = counts.set_index("history_state")
            markov_result["transition_counts"] = counts

        if "markov_transition_probabilities" in tables:
            probs = pd.read_sql("SELECT * FROM markov_transition_probabilities", conn)
            if "history_state" in probs.columns:
                probs = probs.set_index("history_state")
            markov_result["transition_probabilities"] = probs

        if "markov_transitions_long" in tables:
            markov_result["transitions_long_table"] = pd.read_sql("SELECT * FROM markov_transitions_long", conn)

        if markov_result:
            project.markov_result = markov_result

    def _get_tables(self, conn):
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table';"
        )
        return [row[0] for row in cursor.fetchall()]
