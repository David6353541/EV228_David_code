import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import linregress

DF=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\KRDU_temp_188708-202508.csv")
print(DF.columns)

subset=DF 

slope,intercept,r,p,se=linregress(subset['YEAR'],subset['metANN'])

print("Slope:",slope)
print("Intercept:",intercept)
print("Correlation coefficient (r):",r)
print("P-value:",p)
print("Standard error:",se)

plt.figure(figsize=(10,6))
plt.scatter(subset['YEAR'],subset['metANN'],color='green',label='collected data',alpha=0.7)
plt.plot(subset['YEAR'],intercept+slope*subset['YEAR'],color='red',linestyle='--',linewidth=2,label=f'Linear fit (slope={slope:.4f})')
plt.xlabel('Year')
plt.ylabel('Annual Mean Temperature (metANN)')
plt.title('Linear Regression of Temperature vs Year')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()