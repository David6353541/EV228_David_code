# -*- coding: utf-8 -*-
"""
Created on Sun Nov  9 19:06:52 2025

@author: zheng
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
from scipy.stats import linregress
import os

def diagnose_precipitation_data(dataarray, region_name):
    print(f"\n--- {region_name} Data Diagnosis ---")
    print(f"Data shape: {dataarray.shape}")
    print(f"Time range: {dataarray.valid_time.values[0]} to {dataarray.valid_time.values[-1]}")
    print(f"Data type: {dataarray.dtype}")
    print(f"Units: {dataarray.attrs.get('units', 'Not specified')}")
    print(f"Long name: {dataarray.attrs.get('long_name', 'Not specified')}")
    
    data_min = float(dataarray.min().values)
    data_max = float(dataarray.max().values)
    data_mean = float(dataarray.mean().values)
    

    #CHECK THE UNIT USED IN DATASET
    print(f"Raw values - Min: {data_min:.8f}, Max: {data_max:.8f}, Mean: {data_mean:.8f}")
    
    if data_mean < 1e-8:
        print("Interpretation: Very small values - likely m/s (precipitation rate)")
        return 'm/s'
    elif data_mean < 0.01: 
        print("Interpretation: Moderate values - likely monthly total in meters")
        return 'm_monthly'
    elif data_mean > 0.1:
        print("Interpretation: Large values - check if already in mm")
        return 'check_units'
    else:
        return 'unknown'

def convert_era5_monthly_precipitation(monthly_data, region_name):
    print(f"\nConverting {region_name} monthly precipitation data...")
    
    original_mean = float(monthly_data.mean().values)
    print(f"Original mean value: {original_mean:.8f}")
    print(f"Original units: {monthly_data.attrs.get('units', 'Not specified')}")
    
    if original_mean < 0.01:  
        print("Using conversion: monthly mean in m/day → mm/month")
        days_in_month = 30.4375  
        monthly_total_mm = monthly_data * days_in_month * 1000
        conversion_type = "m/day_to_mm/month"
    
    elif original_mean < 1.0:  
        print("Using conversion: monthly total in meters → mm/month")
        monthly_total_mm = monthly_data * 1000
        conversion_type = "m_to_mm"
    
    else:
        print("Using direct scaling approach")
        monthly_total_mm = monthly_data 
        conversion_type = "direct"
    
    print(f"Conversion method: {conversion_type}")
    print(f"Converted mean: {monthly_total_mm.mean().values:.1f} mm/month")
    
    monthly_total_mm.attrs['units'] = 'mm/month'
    monthly_total_mm.attrs['long_name'] = 'Monthly total precipitation'
    monthly_total_mm.attrs['conversion_method'] = conversion_type
    return monthly_total_mm

def calculate_spatial_mean(data_array):
    return data_array.mean(dim=["latitude", "longitude"])

def calculate_trend(years, values):
    slope, intercept, r_value, p_value, std_err = linregress(years, values)
    trend_line = intercept + slope * years
    return slope, intercept, p_value, r_value**2, trend_line

def calculate_moving_average(data, window=12):
    return data.rolling(valid_time=window, center=True).mean()

def calculate_decadal_averages(annual_data, start_year=1960, end_year=2020):
    decades = range(start_year, end_year, 10)
    decadal_means = []
    for year in decades:
        decade_data = annual_data.sel(valid_time=slice(f"{year}-01-01", f"{year+9}-12-31"))
        decadal_means.append(decade_data.mean().values)
    return decades, decadal_means

def analyze_seasonal_trends(monthly_data):
    """Seasonal trend"""
    monthly_clim = monthly_data.groupby('valid_time.month').mean()
    seasons = {'Winter': [12, 1, 2], 'Spring': [3, 4, 5], 'Summer': [6, 7, 8], 'Autumn': [9, 10, 11]}
    seasonal_trends = {}
    for season_name, months in seasons.items():
        seasonal_mask = monthly_data['valid_time.month'].isin(months)
        seasonal_data = monthly_data.where(seasonal_mask, drop=True)
        seasonal_annual = seasonal_data.resample(valid_time='YE').sum()
        
        if len(seasonal_annual) > 2:
            years = seasonal_annual["valid_time.year"].values
            slope, _, p_value, _, _ = calculate_trend(years, seasonal_annual.values)
            seasonal_trends[season_name] = slope
    
    return monthly_clim, seasonal_trends

# Set the path and import the file
PATH_NORTH = "C:\\Users\\zheng\\Desktop\\EV228\\9975c00545ab96ba4dd840ad53ea2934.nc"
PATH_CENTRAL = "C:\\Users\\zheng\\Desktop\\EV228\\1943052617f20d9e8ad93b99d7e4c322.nc"
print("Loading ERA5 Monthly Averaged Reanalysis Data...")
print("=" * 60)

ds_north = xr.open_dataset(PATH_NORTH)
ds_central = xr.open_dataset(PATH_CENTRAL)

var_north = list(ds_north.data_vars)[0]
var_central = list(ds_central.data_vars)[0]
print(f"Variables found: North - {var_north}, Central - {var_central}")

print(f"\nNorthern data attributes:")
for key, value in ds_north[var_north].attrs.items():
    print(f" {key}: {value}")
print(f"\nCentral data attributes:")
for key, value in ds_central[var_central].attrs.items():
    print(f" {key}: {value}")

print("\nProcessing regional data and time range...")
north_region = ds_north.sel(latitude=slice(42, 36), longitude=slice(106, 118), valid_time=slice("1960-01-01", "2025-12-31"))
central_region = ds_central.sel(latitude=slice(36, 27), longitude=slice(105, 118), valid_time=slice("1960-01-01", "2025-12-31"))

print("\n" + "=" * 60)
north_units = diagnose_precipitation_data(north_region[var_north], "Northern Region")
central_units = diagnose_precipitation_data(central_region[var_central], "Central Region")

print("\n" + "=" * 60)
precip_north_monthly = convert_era5_monthly_precipitation(north_region[var_north], "Northern Region")
precip_central_monthly = convert_era5_monthly_precipitation(central_region[var_central], "Central Region")
precip_north_ts = calculate_spatial_mean(precip_north_monthly)
precip_central_ts = calculate_spatial_mean(precip_central_monthly)

print(f"\n" + "=" * 60)
print(f"Northern China monthly mean: {precip_north_ts.mean().values:.1f} mm/month")
print(f"Central China monthly mean: {precip_central_ts.mean().values:.1f} mm/month")
print(f"Northern China annual estimate: {precip_north_ts.mean().values * 12:.0f} mm/year")
print(f"Central China annual estimate: {precip_central_ts.mean().values * 12:.0f} mm/year")

#print out normal ranges as reference
print(f"\nTypical expected ranges:")
print(f"Northern China: 200-800 mm/year (≈15-65 mm/month)")
print(f"Central China: 600-1500 mm/year (≈50-125 mm/month)")

north_monthly_mean = precip_north_ts.mean().values
central_monthly_mean = precip_central_ts.mean().values

#check the data
if north_monthly_mean > 200 or central_monthly_mean > 200: 
    print(f"\nApplying correction factor - values appear too high")
    correction_factor = 0.001
    precip_north_ts = precip_north_ts * correction_factor
    precip_central_ts = precip_central_ts * correction_factor
    print(f"Corrected Northern China: {precip_north_ts.mean().values:.1f} mm/month")
    print(f"Corrected Central China: {precip_central_ts.mean().values:.1f} mm/month")

precip_north_annual = precip_north_ts.resample(valid_time='YE').sum()
precip_central_annual = precip_central_ts.resample(valid_time='YE').sum()

print(f"\nAnnual Precipitation:")
print(f"Northern China: {precip_north_annual.mean().values:.0f} mm/year")
print(f"Central China: {precip_central_annual.mean().values:.0f} mm/year")

years = precip_north_annual["valid_time.year"].values
slope_n, intercept_n, p_n, r2_n, trend_n = calculate_trend(years, precip_north_annual.values)
slope_c, intercept_c, p_c, r2_c, trend_c = calculate_trend(years, precip_central_annual.values)

print(f"Northern Region: {slope_n:.3f} mm/year (p={p_n:.4f}, R²={r2_n:.4f})")
print(f"Central Region: {slope_c:.3f} mm/year (p={p_c:.4f}, R²={r2_c:.4f})")



# FIGURE 1
plt.style.use('default')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

ax1.plot(precip_north_ts.valid_time, precip_north_ts, 
         color='#0072B2', linewidth=0.8, alpha=0.7, label='Northern China')
ax1.plot(precip_central_ts.valid_time, precip_central_ts, 
         color='#E69F00', linewidth=0.8, alpha=0.7, label='Central China')
ax1.set_ylabel('Precipitation (mm/month)', fontsize=12)
ax1.set_title('Monthly Precipitation (1960–2025, ERA5)', fontsize=14, fontweight='bold')
ax1.grid(True, linestyle='--', alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xticks(pd.date_range(start='1960-01-01', end='2025-01-01', freq='10YS'))
ax1.set_xticklabels(['1960', '1970', '1980', '1990', '2000', '2010', '2020'], fontsize=10)
ax1.set_xlim(pd.Timestamp('1960-01-01'), pd.Timestamp('2025-12-31'))
ax1.text(0.02, 0.95, 'Northern China (42-36°N, 106-118°E)', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
ax1.text(0.02, 0.85, 'Central China (36-27°N, 105-118°E)', 
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

north_smooth = calculate_moving_average(precip_north_ts)
central_smooth = calculate_moving_average(precip_central_ts)

years_smooth = pd.to_datetime(north_smooth.valid_time.values).year
slope_n_smooth, intercept_n_smooth, p_n_smooth, r2_n_smooth, trend_n_smooth = calculate_trend(
    years_smooth[~np.isnan(north_smooth.values)], 
    north_smooth.values[~np.isnan(north_smooth.values)]
)
slope_c_smooth, intercept_c_smooth, p_c_smooth, r2_c_smooth, trend_c_smooth = calculate_trend(
    years_smooth[~np.isnan(central_smooth.values)], 
    central_smooth.values[~np.isnan(central_smooth.values)]
)

ax2.plot(north_smooth.valid_time, north_smooth, 
         color='#0072B2', linewidth=2, label='Northern China (12-month moving avg)')
ax2.plot(central_smooth.valid_time, central_smooth, 
         color='#E69F00', linewidth=2, label='Central China (12-month moving avg)')

trend_line_n_smooth = intercept_n_smooth + slope_n_smooth * years_smooth
trend_line_c_smooth = intercept_c_smooth + slope_c_smooth * years_smooth

ax2.plot(north_smooth.valid_time, trend_line_n_smooth, 
         color='#0072B2', linestyle='--', linewidth=1.5, alpha=0.8, 
         label=f'N. China trend: {slope_n_smooth:.3f} mm/yr')
ax2.plot(central_smooth.valid_time, trend_line_c_smooth, 
         color='#E69F00', linestyle='--', linewidth=1.5, alpha=0.8,
         label=f'C. China trend: {slope_c_smooth:.3f} mm/yr')

ax2.set_ylabel('Precipitation (mm/month)', fontsize=12)
ax2.set_title('Smoothed Precipitation (12-month moving average)', fontsize=14, fontweight='bold')
ax2.grid(True, linestyle='--', alpha=0.3)
ax2.legend(loc='upper right', fontsize=10)
ax2.set_xticks(pd.date_range(start='1960-01-01', end='2025-01-01', freq='10YS'))
ax2.set_xticklabels(['1960', '1970', '1980', '1990', '2000', '2010', '2020'], fontsize=10)
ax2.set_xlim(pd.Timestamp('1960-01-01'), pd.Timestamp('2025-12-31'))

plt.tight_layout()
plt.savefig("C:\\Users\\zheng\\Desktop\\EV228\\ERA5_Monthly_Precipitation_CORRECTED.png", 
            dpi=600, bbox_inches='tight', facecolor='white')
plt.show()

print("\n" + "=" * 60)
print("STATISTICAL ANALYSIS RESULTS")
print("=" * 60)

print(f"\nNorthern Region (42-36°N, 106-118°E):")
print(f"Mean Annual Precipitation: {precip_north_annual.mean().values:.0f} ± {precip_north_annual.std().values:.0f} mm")
print(f"Long-term Trend: {slope_n:.3f} mm/year ({slope_n*10:.2f} mm/decade)")
print(f"Trend Significance: p = {p_n:.4f}")
print(f"Wettest Year: {precip_north_annual.max().values:.0f} mm ({years[np.argmax(precip_north_annual.values)]})")
print(f"Driest Year: {precip_north_annual.min().values:.0f} mm ({years[np.argmin(precip_north_annual.values)]})")

print(f"\nCentral Region (36-27°N, 105-118°E):")
print(f"Mean Annual Precipitation: {precip_central_annual.mean().values:.0f} ± {precip_central_annual.std().values:.0f} mm")
print(f"Long-term Trend: {slope_c:.3f} mm/year ({slope_c*10:.2f} mm/decade)")
print(f"Trend Significance: p = {p_c:.4f}")
print(f"Wettest Year: {precip_central_annual.max().values:.0f} mm ({years[np.argmax(precip_central_annual.values)]})")
print(f"Driest Year: {precip_central_annual.min().values:.0f} mm ({years[np.argmin(precip_central_annual.values)]})")

relative_change_north = (slope_n * (2025-1960) / precip_north_annual.mean().values * 100)
relative_change_central = (slope_c * (2025-1960) / precip_central_annual.mean().values * 100)

print(f"\nRelative Changes (1960-2025):")
print(f"Northern Region: {relative_change_north:+.2f}%")
print(f"Central Region: {relative_change_central:+.2f}%")

difference = precip_central_annual.mean().values - precip_north_annual.mean().values
ratio = precip_north_annual.mean().values / precip_central_annual.mean().values

print(f"\nInter-regional Differences:")
print(f"Central - Northern Mean Difference: {difference:.0f} mm/year")
print(f"Northern/Central Ratio: {ratio:.3f}")

print(f"\nRESEARCH HYPOTHESIS EVALUATION:")
if slope_n > slope_c:
    if slope_n > 0 and slope_c > 0:
        print("Both regions show increasing trends, Northern region increases faster")
        print("Supports precipitation belt northward shift hypothesis")
    elif slope_n > slope_c:
        print("Northern region shows relatively less decrease")
        print("Partially supports northward shift hypothesis")
else:
    print("Trend pattern does not support northward shift hypothesis")




print("\nSeasonal precipitation trend analysis")
north_monthly, north_seasonal = analyze_seasonal_trends(precip_north_ts)
central_monthly, central_seasonal = analyze_seasonal_trends(precip_central_ts)

# Correlation
print(f"\nCompair the seasonal trend (Northern/Central part):")
support_count = 0
for season in north_seasonal.keys():
    trend_diff = north_seasonal[season] - central_seasonal[season]
    supports_hypothesis = trend_diff > 0
    if supports_hypothesis:
        support_count += 1
    
    print(f"{season}: Northern part{ north_seasonal[season]:.3f} vs Central part{central_seasonal[season]:.3f} mm/Year | difference: {trend_diff:.3f} | {'Support' if supports_hypothesis else 'Nonsupport'}")

# Check the hypotheses and print out the result
print(f"\nEvaluation of Research Hypotheses:")
print(f"{support_count}/4 seasons support the hypothesis: The potential northern shift of prcp belt")
if support_count >= 3:
    print("→ Strongly support the hypothesis")
elif support_count >= 2:
    print("→ Partially support the hypothesis") 
else:
    print("→ The hypothesis is not supported by result")

# identify the significant value and season
main_season = max(north_seasonal.items(), key=lambda x: abs(x[1]))[0]
print(f"Dominant season: {main_season} (The trend is the most significant)")

# FIGURE 2
plt.figure(figsize=(10, 5))
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
plt.plot(months, north_monthly, 'o-', color='#0072B2', linewidth=2, label='Northern China')
plt.plot(months, central_monthly, 'o-', color='#E69F00', linewidth=2, label='Central China')
plt.ylabel('Precipitation (mm/month)')
plt.title('Monthly Precipitation Climatology')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
