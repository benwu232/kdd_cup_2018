import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

aqs_ld = pd.read_csv('../input/London_AirQuality_Stations.csv')
grid_ld = pd.read_csv('../input/London_grid_weather_station.csv')
grid_ld.columns = ['grid_name', 'Latitude', 'Longitude']

plt.scatter(aqs_ld.Longitude, aqs_ld.Latitude)
plt.scatter(grid_ld.Longitude, grid_ld.Latitude)

plt.show()