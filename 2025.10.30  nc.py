import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from extract_variable import variable


os.chdir(r'C:\\Users\\zheng\\Desktop\\EV228')

data=xr.open_dataset("C:\\Users\\zheng\\Desktop\\EV228\\7be50dd7ac9e9756276848e8e7883d1b.nc")

variables=data.data_vars
print(variables)
print(data.coords)

for coord in data.coords:
    print(f"{coord}:{data.coords[coord].attrs.get('units','No units specified')}")

avg=data.mean(dim='valid_time')
avg_df=avg.to_dataframe().reset_index()

avg_pivot=avg_df.pivot(index='latitude',columns='longitude',values='d2m')
print(avg_pivot)

plt.figure(figsize=(10,8))
contour=plt.contourf(avg_pivot.columns,avg_pivot.index,avg_pivot.values,cmap='viridis')

cbar=plt.colorbar(contour)
cbar.set_label('Air Temperature (°C)')

plt.title('Averaged Air Temperature Over Time')
plt.xlabel('Longitude')
plt.ylabel('Latitude')

plt.show()
