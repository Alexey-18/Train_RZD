from pathlib import Path
import pandas as pd

def load_all_csv(root_dir: str, pattern: str='*.csv') -> pd.DataFrame:
    """
    Считывает все CSV из папки и подпапок, склеивает в один DataFrame
    """
    files = list(Path(root_dir).rglob(pattern))
    if not files:
        raise FileNotFoundError(f'No CSV in {root_dir}')
    dfs = [pd.read_csv(f) for f in files]
    return pd.concat(dfs, ignore_index=True)