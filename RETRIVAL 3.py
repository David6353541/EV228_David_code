import os
import pandas as pd
import matplotlib.pyplot as plt
import xarray as xr
import numpy as np
import matplotlib.dates as mdates

def descriptive_stats(file_path,dims=None):
    data=pd.read_csv(file_path)
    stats={
        'mean': data.mean(numeric_only=True),
        'median': data.median(numeric_only=True), 
        'min': data.min(numeric_only=True),
        'max': data.max(numeric_only=True),
        'std': data.std(numeric_only=True)}
    return stats,data

os.chdir(r'C:\Users\zheng\Desktop\EV228')

DSBR=pd.read_csv("Bishop-Rock.csv")

stats,df=descriptive_stats("Bishop-Rock.csv")
df['TIMESTAMP']=pd.to_datetime(df['TIMESTAMP'])
df_clean = df.replace([999.9,999.90,-999.9],np.nan)

start_date=df_clean['TIMESTAMP'].min()
end_date=df_clean['TIMESTAMP'].max()
print(f"Time period: {start_date.date()} to {end_date.date()}")

for var in ['T_HMP','T_HMP_2', 'RH', 'WS_AVG']:
    if var in stats['mean']:
        print(f"{var}: Mean={stats['mean'][var]:.1f}, Std={stats['std'][var]:.1f}, "
              f"Min={stats['min'][var]:.1f}, Max={stats['max'][var]:.1f}")

plt.figure(figsize=(12, 8))
plt.subplot(3,1,1)
plt.plot(df_clean['TIMESTAMP'],df_clean['T_HMP'], 'b-')
plt.ylabel('Temp (°C)')
plt.title('Temperature - All Data')
plt.grid(True,alpha=0.3)

plt.subplot(3,1,2)
plt.plot(df_clean['TIMESTAMP'],df_clean['RH'], 'g-')
plt.ylabel('RH (%)')
plt.grid(True,alpha=0.3)

plt.subplot(3,1,3)
plt.plot(df_clean['TIMESTAMP'],df_clean['WS_AVG'], 'r-')
plt.ylabel('Wind (m/s)')
plt.xlabel('Time')
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

subset=df_clean.head(60)
fig,ax=plt.subplots(3,1,figsize=(12,8))

ax[0].plot(subset['TIMESTAMP'],subset['T_HMP'],'bo-',markersize=3)
ax[0].set_ylabel('Temp (°C)')
ax[0].set_title('First 60 Hours')
ax[0].xaxis.set_major_locator(plt.MaxNLocator(6))
ax[0].tick_params(axis='x',rotation=45)

ax[1].plot(subset['TIMESTAMP'],subset['RH'],'go-',markersize=3)
ax[1].set_ylabel('RH (%)')
ax[1].xaxis.set_major_locator(plt.MaxNLocator(6))
ax[1].tick_params(axis='x',rotation=45)

ax[2].plot(subset['TIMESTAMP'],subset['WS_AVG'],'ro-',markersize=3)
ax[2].set_ylabel('Wind (m/s)')
ax[2].set_xlabel('Time')
ax[2].xaxis.set_major_locator(plt.MaxNLocator(6))
ax[2].tick_params(axis='x',rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(df_clean['TIMESTAMP'],df_clean['T_HMP'], 'b-')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.gca().xaxis.set_major_locator(mdates.DayLocator(interval=4))
plt.title('Temperature with Readable X-Axis')
plt.ylabel('Temp (°C)')
plt.tick_params(axis='x',rotation=45)
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

x=np.arange(len(df_clean))
y=df_clean['T_HMP'].dropna()
z=np.polyfit(x[:len(y)],y,1)
trend=np.poly1d(z)

plt.figure(figsize=(12,4))
plt.plot(df_clean['TIMESTAMP'],df_clean['T_HMP'],'b-',label='Temperature')
plt.plot(df_clean['TIMESTAMP'].iloc[:len(y)], trend(x[:len(y)]),'r-',linewidth=2,label='Trendline')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
plt.tick_params(axis='x',rotation=45)
plt.ylabel('Temp (°C)')
plt.legend()
plt.grid(True,alpha=0.3)
plt.tight_layout()
plt.show()

corr=df_clean['T_HMP'].corr(df_clean['T_HMP_2'])
print(f"T_HMP/T_HMP_2 correlation: {corr:.6f}")

