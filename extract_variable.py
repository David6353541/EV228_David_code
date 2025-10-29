import pandas as pd

def variable(file_path, variable_name):
   
    data = pd.read_csv(file_path)
    if variable_name not in data.columns:
        raise ValueError(f"'{variable_name}' not found in {file_path}. Available columns: {list(data.columns)}")
    return data[variable_name]