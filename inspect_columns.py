import os
import pandas as pd

path = r'c:\Users\BTC\Music\Anomaly _Testing_Code\coloumn list for model (feature set ready till now ).xlsx'
print('exists', os.path.exists(path))
if os.path.exists(path):
    xl = pd.ExcelFile(path)
    print('sheets', xl.sheet_names)
    for sheet in xl.sheet_names:
        print('\nSHEET', sheet)
        df = pd.read_excel(path, sheet_name=sheet)
        print(df.head(60).to_string(index=False))
        print('---')
