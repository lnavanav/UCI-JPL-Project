#!/usr/bin/env python
# coding: utf-8

# In[1]:


# From "data_read.py"
from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma

def read_data_wind(filename): # note the change of function name (just for proper naming)
    f = Dataset(filename)  # open the first file to check the dimension of U and V
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    time = f.variables['time'] 
    date = num2date(time[:], time.units)
    u = f.variables['U'][:]
    v = f.variables['V'][:]
    speed = f.variables['speed'][:]
    sample_size = f.variables['sample_size'][:] # note the addition of this line to get the sample_size masked array
    return u, v, speed, lon, lat, date, sample_size

# From "plot_data.py"
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
from mpl_toolkits.basemap import Basemap, maskoceans

def plot_global_contour_map(data, lon, lat, levels, figure_file, title = 'title', cmap = cm.jet):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'eck4', lon_0=0, llcrnrlat = lat.min(), urcrnrlat = lat.max(), 
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    max = m.contourf(x, y, data, levels = levels, extend='both', cmap = cmap)
    cbar_ax = fig.add_axes([0.92, 0.3,0.01, 0.4])
    cb=plt.colorbar(max,cax=cbar_ax)
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()
    
def plot_global_vector_map(uwind, vwind, lon, lat, figure_file, yskip=10, xskip=20):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'eck4', lon_0=0, llcrnrlat = lat.min(), urcrnrlat = lat.max(),
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    N = ma.mean(np.sqrt(uwind[::yskip, ::xskip]**2+vwind[::yskip, ::xskip]**2))
    max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], uwind[::yskip, ::xskip]/N, vwind[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()


# In[2]:


misr_u, misr_v, misr_speed, misr_lon, misr_lat, misr_time, misr_sample_size = read_data_wind('MISR_CMV_MAR2000-FEB2018_monthly.nc')


# In[3]:


# First, I will plot the georeferenced average sample_size across the whole dataset.

misr_sample_size_average_1p5km = ma.mean(misr_sample_size[:,0,:,:], axis = 0)
misr_sample_size_average_2p5km = ma.mean(misr_sample_size[:,1,:,:], axis = 0)
misr_sample_size_average_4p0km = ma.mean(misr_sample_size[:,2,:,:], axis = 0)


# In[25]:


plot_global_contour_map(misr_sample_size_average_1p5km, misr_lon, misr_lat, np.arange(62)+1, 'MISR_CMV_average_sample_size_1p5km',
                        "MISR_CMV_average_sample_size_1p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_2p5km, misr_lon, misr_lat, np.arange(62)+1, 'MISR_CMV_average_sample_size_2p5km',
                        "MISR_CMV_average_sample_size_2p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_4p0km, misr_lon, misr_lat, np.arange(62)+1, 'MISR_CMV_average_sample_size_4p0km',
                        "MISR_CMV_average_sample_size_4p0km", cmap = cm.brg)

# Then, I will try to see what happens when sample_size is averaged on a seasonal basis (Summer and Winter only)
# Then, I will try to derive some explanations of Alex's results based only off of sample_size numerics

misr_months = np.array([i.month for i in misr_time])

misr_t_index_winter = np.where((misr_months >= 12) | (misr_months <= 2))[0]
misr_t_index_spring = np.where(np.logical_and(misr_months >= 3, misr_months <= 5))[0]
misr_t_index_summer = np.where(np.logical_and(misr_months >= 6, misr_months <= 8))[0]
misr_t_index_fall = np.where(np.logical_and(misr_months >= 9, misr_months <= 11))[0]
# In[11]:


misr_sample_size_average_winter_1p5km = ma.mean(misr_sample_size[misr_t_index_winter,0,:,:], axis = 0)
misr_sample_size_average_winter_2p5km = ma.mean(misr_sample_size[misr_t_index_winter,1,:,:], axis = 0)
misr_sample_size_average_winter_4p0km = ma.mean(misr_sample_size[misr_t_index_winter,2,:,:], axis = 0)

misr_sample_size_average_spring_1p5km = ma.mean(misr_sample_size[misr_t_index_spring,0,:,:], axis = 0)
misr_sample_size_average_spring_2p5km = ma.mean(misr_sample_size[misr_t_index_spring,1,:,:], axis = 0)
misr_sample_size_average_spring_4p0km = ma.mean(misr_sample_size[misr_t_index_spring,2,:,:], axis = 0)

misr_sample_size_average_summer_1p5km = ma.mean(misr_sample_size[misr_t_index_summer,0,:,:], axis = 0)
misr_sample_size_average_summer_2p5km = ma.mean(misr_sample_size[misr_t_index_summer,1,:,:], axis = 0)
misr_sample_size_average_summer_4p0km = ma.mean(misr_sample_size[misr_t_index_summer,2,:,:], axis = 0)

misr_sample_size_average_fall_1p5km = ma.mean(misr_sample_size[misr_t_index_fall,0,:,:], axis = 0)
misr_sample_size_average_fall_2p5km = ma.mean(misr_sample_size[misr_t_index_fall,1,:,:], axis = 0)
misr_sample_size_average_fall_4p0km = ma.mean(misr_sample_size[misr_t_index_fall,2,:,:], axis = 0)


# In[24]:


plot_global_contour_map(misr_sample_size_average_winter_1p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_winter_1p5km', 
                        "MISR_CMV_average_sample_size_winter_1p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_winter_2p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_winter_2p5km', 
                        "MISR_CMV_average_sample_size_winter_2p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_winter_4p0km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_winter_4p0km', 
                        "MISR_CMV_average_sample_size_winter_4p0km", cmap = cm.brg)


# In[23]:


plot_global_contour_map(misr_sample_size_average_spring_1p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_spring_1p5km', 
                        "MISR_CMV_average_sample_size_spring_1p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_spring_2p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_spring_2p5km', 
                        "MISR_CMV_average_sample_size_spring_2p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_spring_4p0km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_spring_4p0km', 
                        "MISR_CMV_average_sample_size_spring_4p0km", cmap = cm.brg)


# In[22]:


plot_global_contour_map(misr_sample_size_average_summer_1p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_summer_1p5km', 
                        "MISR_CMV_average_sample_size_summer_1p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_summer_2p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_summer_2p5km', 
                        "MISR_CMV_average_sample_size_summer_2p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_summer_4p0km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_summer_4p0km', 
                        "MISR_CMV_average_sample_size_summer_4p0km", cmap = cm.brg)


# In[21]:


plot_global_contour_map(misr_sample_size_average_fall_1p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_fall_1p5km', 
                        "MISR_CMV_average_sample_size_fall_1p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_fall_2p5km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_fall_2p5km', 
                        "MISR_CMV_average_sample_size_fall_2p5km", cmap = cm.brg)
plot_global_contour_map(misr_sample_size_average_fall_4p0km, misr_lon, misr_lat, 
                        np.arange(62)+1, 'MISR_CMV_average_sample_size_fall_4p0km', 
                        "MISR_CMV_average_sample_size_fall_4p0km", cmap = cm.brg)


# In[ ]:


# Now, I will show the results when oceans are masked

