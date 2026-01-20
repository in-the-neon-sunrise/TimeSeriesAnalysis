import json
import pandas as pd
from ui.models.project import Project


def save_project(project: Project, file_path: str):
    print("SAVE PROJECT CALLED:", file_path)
    data = {
        "csv_path": project.csv_path,
        "dataframe": project.dataframe.to_dict(orient="list")
        if project.dataframe is not None else None
    }

    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def load_project(file_path: str) -> Project:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    df = None
    if data["dataframe"] is not None:
        df = pd.DataFrame(data["dataframe"])

    return Project(
        csv_path=data["csv_path"],
        dataframe=df
    )