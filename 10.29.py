import os
import pandas as pd
import matplotlib.pyplot as plt
from extract_variable import variable

os.chdir(r'C:\\Users\\zheng\\Desktop\\EV228')
KRDU_temp=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\KRDU_temp_188708-202508.csv")
KRDU_temp_years=KRDU_temp['YEAR']
print(KRDU_temp_years)
KRDU_temp_JAN=KRDU_temp['JAN']
print(KRDU_temp_JAN)

KRDU_temp_years=variable("C:\\Users\\zheng\\Desktop\\EV228\\KRDU_temp_188708-202508.csv", 'YEAR')
print(KRDU_temp_years)

KRDU_temp_JAN=variable("C:\\Users\\zheng\\Desktop\\EV228\\KRDU_temp_188708-202508.csv", 'JAN')
print(KRDU_temp_JAN)

