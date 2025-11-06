# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 22:04:17 2025

@author: zheng
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
import xarray as xr
from scipy.stats import linregress
import cartopy.feature as cfeature
import cartopy.crs as ccrs



'''set the output directory'''
output_dir=r"C:\Users\zheng\Desktop\EV228"
os.chdir(output_dir)


'''import the data'''
os.chdir(r'C:\\Users\\zheng\\Desktop\\EV228')

EVS1=pd.read_csv(r"C:\\Users\\zheng\\Desktop\\EV228\\CHM00054511.csv") #Beijing
EVS2=pd.read_csv(r"C:\\Users\\zheng\Desktop\\EV228\\CHM00053698.csv") #Shijiazhuang
EVS3=pd.read_csv(r"C:\\Users\\zheng\\Desktop\\EV228\\CHM00054423.csv") #Chengde


print(EVS1.head())
print(EVS1.columns)
print(EVS2.head())
print(EVS2.columns)
print(EVS3.head())
print(EVS3.columns)
'''check the data'''
pd.set_option('display.max_rows', 20)       
pd.set_option('display.max_columns', None)    
pd.set_option('display.width', None)          
pd.set_option('display.max_colwidth', None)   


print("\nEVS1 Preview:")
print(EVS1.head())
print("\nEVS1 Info (with NaN kept):")
print(EVS1.info())
print("\nEVS2 Preview:")
print(EVS2.head())
print("\nEVS2 Info (with NaN kept):")
print(EVS2.info())
print("\nEVS3 Preview:")
print(EVS3.head())
print("\nEVS3 Info (with NaN kept):")
print(EVS3.info())

EVS1['DATE']=pd.to_datetime(EVS1['DATE'])
EVS2['DATE']=pd.to_datetime(EVS2['DATE'])
EVS3['DATE']=pd.to_datetime(EVS3['DATE'])


'''Monthly average PRCP changing trend Beijing'''
df1=EVS1[EVS1['DATE'] >= '1951-01-01'].copy()   
df1.set_index('DATE', inplace=True)
monthly_prcp=df1['PRCP'].resample('M').mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_prcp.index, monthly_prcp.values, color='blue', linewidth=1.5)
plt.title("Monthly Average Precipitation Trend")
plt.xlabel("Year-Month")
plt.ylabel("Precipitation (mm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

'''Monthly average PRCP changing trend Shijiazhuang'''
df2=EVS2[EVS2['DATE'] >= '1951-01-01'].copy()   
df2.set_index('DATE', inplace=True)
monthly_prcp=df2['PRCP'].resample('M').mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_prcp.index, monthly_prcp.values, color='blue', linewidth=1.5)
plt.title("Monthly Average Precipitation Trend")
plt.xlabel("Year-Month")
plt.ylabel("Precipitation (mm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

'''Monthly average PRCP changing trend Chengde'''
df3=EVS3[EVS3['DATE'] >= '1951-01-01'].copy()   
df3.set_index('DATE', inplace=True)
monthly_prcp=df3['PRCP'].resample('M').mean()

plt.figure(figsize=(12, 6))
plt.plot(monthly_prcp.index, monthly_prcp.values, color='blue', linewidth=1.5)
plt.title("Monthly Average Precipitation Trend")
plt.xlabel("Year-Month")
plt.ylabel("Precipitation (mm)")
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()



'''PRCP trend in 2025'''
dfA=EVS1[(EVS1['DATE'] >= '2025-01-01') & (EVS1['DATE'] <= '2025-9-01')].copy()
dfA['RollingMean']=dfA['PRCP'].rolling(window=30, min_periods=1).mean()
dfA['RollingStd']=dfA['PRCP'].rolling(window=30, min_periods=1).std()

dfB=EVS2[(EVS2['DATE'] >= '2025-01-01') & (EVS2['DATE'] <= '2025-9-01')].copy()
dfB['RollingMean']=dfB['PRCP'].rolling(window=30, min_periods=1).mean()
dfB['RollingStd']=dfB['PRCP'].rolling(window=30, min_periods=1).std()

dfC=EVS3[(EVS3['DATE'] >= '2025-01-01') & (EVS3['DATE'] <= '2025-9-01')].copy()
dfC['RollingMean']=dfC['PRCP'].rolling(window=30, min_periods=1).mean()
dfC['RollingStd']=dfC['PRCP'].rolling(window=30, min_periods=1).std()


'''find anomaly threshold (mean ± 2*std)'''
upper_threshold=dfA['RollingMean'] + 2 * dfA['RollingStd']
lower_threshold=dfA['RollingMean'] - 2 * dfA['RollingStd']

upper_threshold1=dfB['RollingMean'] + 2 * dfB['RollingStd']
lower_threshold1=dfB['RollingMean'] - 2 * dfB['RollingStd']

upper_threshold2=dfC['RollingMean'] + 2 * dfC['RollingStd']
lower_threshold2=dfC['RollingMean'] - 2 * dfC['RollingStd']

dfA['Upper_Threshold']=dfA['RollingMean'] + 2 * dfA['RollingStd']
dfA['Lower_Threshold']=dfA['RollingMean'] - 2 * dfA['RollingStd']

dfB['Upper_Threshold']=dfB['RollingMean'] + 2 * dfB['RollingStd']
dfB['Lower_Threshold']=dfB['RollingMean'] - 2 * dfB['RollingStd']

dfC['Upper_Threshold']=dfC['RollingMean'] + 2 * dfC['RollingStd']
dfC['Lower_Threshold']=dfC['RollingMean'] - 2 * dfC['RollingStd']

'''Identify anomalies'''
anomalies_A=dfA[(dfA['PRCP'] > dfA['Upper_Threshold']) | (dfA['PRCP'] < dfA['Lower_Threshold'])]
print(f"Detected {len(anomalies_A)} anomalies in Beijing")

anomalies_B=dfB[(dfB['PRCP'] > dfB['Upper_Threshold']) | (dfB['PRCP'] < dfB['Lower_Threshold'])]
print(f"Detected {len(anomalies_B)} anomalies in Shijiazhuang")

anomalies_C=dfC[(dfC['PRCP'] > dfC['Upper_Threshold']) | (dfC['PRCP'] < dfC['Lower_Threshold'])]
print(f"Detected {len(anomalies_C)} anomalies in Chengde")

'''Plot Beijing 2025 PRCP data'''
plt.figure(figsize=(12, 6))
plt.plot(dfA['DATE'], dfA['PRCP'], color='green', linewidth=1.1, label='Daily PRCP')
plt.plot(dfA['DATE'], dfA['RollingMean'], color='orange', linewidth=1.3, label='30-day Mean')
plt.fill_between(dfA['DATE'], dfA['Lower_Threshold'], dfA['Upper_Threshold'], color='yellow', alpha=0.2, label='Normal Range')
plt.scatter(anomalies_A['DATE'], anomalies_A['PRCP'], color='red', s=20, label='Anomalies')
plt.title("Precipitation Anomalies 2025 (1)")
plt.xlabel("Date")
plt.ylabel("Precipitation(mm)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


'''Plot Shijiazhuang 2025 PRCP data'''
plt.figure(figsize=(12, 6))
plt.plot(dfB['DATE'], dfB['PRCP'], color='green', linewidth=1.1, label='Daily PRCP')
plt.plot(dfB['DATE'], dfB['RollingMean'], color='orange', linewidth=1.3, label='30-day Mean')
plt.fill_between(dfB['DATE'], dfB['Lower_Threshold'], dfB['Upper_Threshold'], color='yellow', alpha=0.2, label='Normal Range')
plt.scatter(anomalies_B['DATE'], anomalies_B['PRCP'], color='red', s=20, label='Anomalies')
plt.title("Precipitation Anomalies 2025 (2)")
plt.xlabel("Date")
plt.ylabel("Precipitation(mm)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


'''Plot Chengde 2025 PRCP data'''
plt.figure(figsize=(12, 6))
plt.plot(dfC['DATE'], dfC['PRCP'], color='green', linewidth=1.1, label='Daily PRCP')
plt.plot(dfC['DATE'], dfC['RollingMean'], color='orange', linewidth=1.3, label='30-day Mean')
plt.fill_between(dfC['DATE'], dfC['Lower_Threshold'], dfC['Upper_Threshold'], color='yellow', alpha=0.2, label='Normal Range')
plt.scatter(anomalies_C['DATE'], anomalies_C['PRCP'], color='red', s=20, label='Anomalies')
plt.title("Precipitation Anomalies 2025 (3)")
plt.xlabel("Date")
plt.ylabel("Precipitation(mm)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()


'''在地图上显示各自的位置    Show them in the map(not good as Google map)'''

cities={"Beijing": (116.4074, 39.9042),
    "Shijiazhuang": (114.5149, 38.0428),
    "Chengde": (117.9634, 40.9527)}

'''创建地图'''
plt.figure(figsize=(8, 8))
ax=plt.axes(projection=ccrs.PlateCarree())

'''设置显示范围为北方地区'''
ax.set_extent([110, 122, 35, 43], crs=ccrs.PlateCarree())

'''增加地理特征'''
ax.add_feature(cfeature.BORDERS, linewidth=0.7)
ax.add_feature(cfeature.COASTLINE, linewidth=0.7)
ax.add_feature(cfeature.LAKES, alpha=0.5)
ax.add_feature(cfeature.RIVERS, alpha=0.5)
ax.add_feature(cfeature.LAND, facecolor='lightgray')
ax.add_feature(cfeature.OCEAN, facecolor='aliceblue')

'''绘制城市点并标注名称'''
for city, (lon, lat) in cities.items():
    ax.plot(lon, lat, 'ro', markersize=6, transform=ccrs.PlateCarree())
    ax.text(lon + 0.3, lat + 0.2, city, fontsize=10, weight='bold', transform=ccrs.PlateCarree())

gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', alpha=0.6, linestyle='--')
gl.top_labels=False    
gl.right_labels=False   
gl.xlabel_style={'size': 9}
gl.ylabel_style={'size': 9}

'''Show it on map'''
plt.title("Locations of Three Stations in Northern China", fontsize=13)
plt.tight_layout()
plt.show()






'''Function: 计算每年总降水与峰值月份'''
def calc_rain_features(df, city_name):
    # 日期处理
    df['DATE']=pd.to_datetime(df['DATE'], errors='coerce')
    df = df.dropna(subset=['DATE'])  # 删除日期为空的行
    df['Year']=df['DATE'].dt.year
    df['Month']=df['DATE'].dt.month

    # 年总降水量（跳过缺失值）
    annual_prcp=df.groupby('Year')['PRCP'].sum(min_count=1).reset_index(name='Total_PRCP')

    # 计算月平均降水
    monthly_mean=df.groupby(['Year', 'Month'])['PRCP'].mean().reset_index()

    # 移除整年全为 NaN 的年份
    valid_years=monthly_mean.groupby('Year')['PRCP'].apply(lambda x: x.notna().any())
    monthly_mean=monthly_mean[monthly_mean['Year'].isin(valid_years[valid_years].index)]

    # 计算每年降水峰值月份
    idx=monthly_mean.groupby('Year')['PRCP'].idxmax().dropna()
    rain_peak=monthly_mean.loc[idx]
    rain_peak=rain_peak[['Year', 'Month']].rename(columns={'Month': 'Peak_Month'})

    # 合并结果
    result=pd.merge(annual_prcp, rain_peak, on='Year', how='inner')
    result['City'] = city_name
    return result


'''三个城市数据'''
df_beijing=calc_rain_features(EVS1, 'Beijing')
df_shijiazhuang=calc_rain_features(EVS2, 'Shijiazhuang')
df_chengde=calc_rain_features(EVS3, 'Chengde')

'''合并'''
combined=pd.concat([df_beijing, df_shijiazhuang, df_chengde], ignore_index=True)


'''年总降水量趋势图 Annual Average precipitation Trend(use in the poster)'''

plt.figure(figsize=(10,6))
for city, color in zip(['Beijing', 'Shijiazhuang', 'Chengde'], ['green','red','blue']):
    subset = combined[combined['City']==city]
    
    plt.plot(
        subset['Year'], subset['Total_PRCP'], 
        marker='o', 
        markersize=4,        
        alpha=0.8,         
        linewidth=2,        
        label=city, 
        color=color)
    
    slope, intercept, r, p, se=linregress(subset['Year'], subset['Total_PRCP'])
    plt.plot(
        subset['Year'], 
        intercept+slope * subset['Year'], 
        color=color, 
        linestyle='--', 
        linewidth=2)


plt.title("Annual Total Precipitation Trend (1951–2025)", fontsize=18)
plt.xlabel("Year", fontsize=14)
plt.ylabel("Total Precipitation (mm)", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()







'''峰值月份变化趋势图     暂不使用/not decide to use'''
plt.figure(figsize=(10,6))
for city, color in zip(['Beijing', 'Shijiazhuang', 'Chengde'], ['green','red','blue']):
    subset=combined[combined['City']==city]
    plt.plot(subset['Year'], subset['Peak_Month'], marker='o', label=city, color=color)
    
    slope, intercept, r, p, se=linregress(subset['Year'], subset['Peak_Month'])
    plt.plot(subset['Year'], intercept+slope*subset['Year'], color=color, linestyle='--')

plt.title("Month of Maximum Rainfall Shift (1951–2025)")
plt.xlabel("Year")
plt.ylabel("Month (1–12)")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
