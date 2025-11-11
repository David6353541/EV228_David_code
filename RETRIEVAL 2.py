import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df=pd.read_csv("C:\\Users\\zheng\\Desktop\\EV228\\ASM00094998_temp_194804-202508.csv")

print(f"Data spans from {df['YEAR'].min()}to{df['YEAR'].max()}")
print(f"Total years:{df['YEAR'].max()-df['YEAR'].min()+1} years")
print(f"Number of records: {len(df)} years")
df_clean=df.copy()
df_clean=df_clean.replace(999.90, np.nan)
monthly_cols=['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
seasonal_cols=['D-J-F', 'M-A-M', 'J-J-A', 'S-O-N', 'metANN']

all_temp_data=[]
for col in monthly_cols+seasonal_cols:
    if col in df_clean.columns:
        all_temp_data.extend(df_clean[col].dropna().values)
all_temp_data=np.array(all_temp_data)

print("\nStatistical analysis:")
print(f"Mean temperature:{np.nanmean(all_temp_data):.2f}°C")
print(f"Standard deviation:{np.nanstd(all_temp_data):.2f}°C")
print(f"Maximum temperature:{np.nanmax(all_temp_data):.2f}°C")
print(f"Minimum temperature:{np.nanmin(all_temp_data):.2f}°C")


plt.figure(figsize=(12,8))
plt.subplot(2,1,1)
plt.plot(df_clean['YEAR'],df_clean['metANN'],'b-',linewidth=2,label='Annual Mean')
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.title('Armagh Observatory Annual Temperature 1948-2025',fontsize=14,fontweight='bold')
plt.grid(True,alpha=0.3)
plt.legend()

plt.subplot(2,1,2)
plt.plot(df_clean['YEAR'],df_clean['D-J-F'],'r-',label='Winter (D-J-F)',alpha=0.7)
plt.plot(df_clean['YEAR'],df_clean['M-A-M'],'g-',label='Spring (M-A-M)',alpha=0.7)
plt.plot(df_clean['YEAR'],df_clean['J-J-A'],'b-',label='Summer (J-J-A)',alpha=0.7)
plt.plot(df_clean['YEAR'],df_clean['S-O-N'],'orange',label='Autumn (S-O-N)',alpha=0.7)
plt.xlabel('Year')
plt.ylabel('Temperature (°C)')
plt.title('Seasonal Temperatures')
plt.grid(True,alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

