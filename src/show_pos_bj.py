import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aqs = pd.read_csv('../input/Beijing_AirQuality_Stations_cn.csv')
aqs.columns = ['grid_name', 'Longitude', 'Latitude']
grid = pd.read_csv('../input/Beijing_grid_weather_station.csv')
grid.columns = ['grid_name', 'Latitude', 'Longitude']

plt.scatter(aqs.Latitude, aqs.Longitude)
plt.scatter(grid.Latitude, grid.Longitude)

plt.show()