import pandas as pd
import numpy as np

class DataLoader:
    """
    Módulo para cargar y preparar el dataset base.
    """
    @staticmethod
    def load_and_prepare(data_path: str, nrows: int = 5000) -> pd.DataFrame:
        df_raw = pd.read_csv(data_path, nrows=nrows)
        df_base = df_raw[df_raw["role"] == "customer"].copy()
        df_base["text_length"] = df_base["text"].fillna("").str.len()
        df_base["response_time_mins"] = np.random.exponential(15, len(df_base))
        df_base["outcome"] = df_base["outcome"].fillna("Pending")
        return df_base
