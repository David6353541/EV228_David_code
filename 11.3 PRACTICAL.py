import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr


file_path="C:\\Users\\zheng\\Desktop\\EV228\\SGM00061600_temp_189201-202508.csv"
PART2=pd.read_csv(file_path)

print(PART2)
print(PART2.columns)

def descriptive_stats(df):

    df_cleaned=df.mask(df==999.9)
    stats={
        'mean': df_cleaned.mean(numeric_only=True),
        'median': df_cleaned.median(numeric_only=True),
        'min': df_cleaned.min(numeric_only=True),
        'max': df_cleaned.max(numeric_only=True),
        'std': df_cleaned.std(numeric_only=True)}
    return stats, df_cleaned


stats, cleaned_data = descriptive_stats(PART2)
print(stats)

clean_data=cleaned_data.dropna(subset=['metANN'])


plt.figure(figsize=(10, 8))
plt.plot(clean_data['YEAR'], clean_data['metANN'], color='blue', linewidth=1.3)
plt.title("metANN changing trend")
plt.xlabel("Year")
plt.xlim(1892,2025)
plt.ylabel("Temperature (°C)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig("C:\\Users\\zheng\\Desktop\\EV228\\PART2.png", dpi=300, bbox_inches='tight', pad_inches=0)
plt.show()



file_path="C:\\Users\\zheng\\Desktop\\EV228\\era5_10mwind_1980-1989.nc"
PART3=xr.open_dataset(file_path)

print(PART3)
print("\nVariables:", list(PART3.data_vars))
print("\nCoordinates:", list(PART3.coords))
print("\nDimensions:", PART3.dims)

wind_var=list(PART3.data_vars)[0]
wind_mean=PART3[wind_var].mean(dim='valid_time', skipna=True)

plt.figure(figsize=(10, 6))
wind_mean.plot(cmap='coolwarm', cbar_kwargs={'label': '10m Wind Speed/Si10'})
plt.title("Time-Averaged Wind Speed", fontsize=14)
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.tight_layout()
plt.savefig("C:\\Users\\zheng\\Desktop\\EV228\\PART3_ERA5_10mwind.png", dpi=300, bbox_inches='tight', pad_inches=0.1)
plt.show()












