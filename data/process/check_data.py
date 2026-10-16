import pandas as pd
from pathlib import Path
file_path = "E:/code/stock-agent/data/process/dataset_sent17.parquet"
p = Path(file_path)
data = pd.read_parquet(p)
print(data)