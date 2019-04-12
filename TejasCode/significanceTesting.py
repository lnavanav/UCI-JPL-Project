#!/usr/bin/env python
# coding: utf-8

# In[50]:


# From "data_read.py"
from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
from mpl_toolkits.basemap import Basemap, maskoceans
import matplotlib.pyplot as plt

def read_data_wind(filename): # note the change of function name (just for proper naming)
    f = Dataset(filename)  # open the first file to check the dimension of U and V
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    time = f.variables['time'] 
    date = num2date(time[:], time.units)
    # u = f.variables['U'][:]
    # v = f.variables['V'][:]
    # speed = f.variables['speed'][:]
    sample_size = f.variables['MISR_sample_size'][:] # note the addition of this line to get the sample_size masked array
    cth_sample_size = f.variables['MISR_CTH_sample_size'][:] # note the addition of this line
    ar_shape = f.variables['AR_shape'][:]
    misr_u_at_cth = f.variables['MISR_U_at_CTH'][:]
    misr_v_at_cth = f.variables['MISR_V_at_CTH'][:]
    misr_cth = f.variables['MISR_CTH'][:]
    merra2_u = f.variables['MERRA2_U'][:]   # level: 3-5 km
    merra2_v = f.variables['MERRA2_V'][:]
    return lon, lat, date, sample_size, cth_sample_size, ar_shape, misr_u_at_cth, misr_v_at_cth, misr_cth, merra2_u, merra2_v


# Spatial Average
def calc_area_weighted_spatial_average(data, lon, lat, masking = 'mask_off', area_weight=True):
    '''Calculate area weighted average of the values in data
    :param data: two-dimensional masked array
    :type dataset: :class:`numpy.ma.core.MaskedArray`
    :returns: an area weighted mean value
    '''

    if lat.ndim == 1:
        lons, lats = np.meshgrid(lon, lat)
    else:
        lats = lat
    weights = np.cos(lats * np.pi / 180.)
    
    if masking == 'mask_ocean':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        data = masked_data
    if masking == 'mask_land':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        masked_data.mask = ~masked_data.mask # 'inverting' the mask to instead mask land
        data = masked_data
        
    if area_weight:
        spatial_average = ma.average(
            data[:], weights=weights)
    else:
        spatial_average = ma.average(data)

    return spatial_average

def calc_area_weighted_standard_deviation(data, lon, lat, masking='mask_off', area_weight=True):
    if lat.ndim == 1:
        lons, lats = np.meshgrid(lon, lat)
    else:
        lats = lat
    weights = np.cos(lats * np.pi / 180.)
    
    if masking == 'mask_ocean':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        data = masked_data
    if masking == 'mask_land':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        masked_data.mask = ~masked_data.mask # 'inverting' the mask to instead mask land
        data = masked_data
        
    squared_data = data[:]*data[:]
    if area_weight:
        spatial_average = ma.average(data[:], weights=weights)
        squared_data_spatial_average = ma.average(squared_data, weights=weights)
    else:
        spatial_average = ma.average(data)
        squared_data_spatial_average = ma.average(squared_data)
    standard_deviation = np.sqrt(squared_data_spatial_average-(spatial_average * spatial_average))
    
    return standard_deviation

# Significance Tests
## Pearson Correlation Coefficient
def pearson_correlation(data1, data2, lon, lat, masking='mask_off', area_weight=True):
    '''This function calculated the Pearson Correlation
    Coefficient between two datasets, data1 and data2. Masking
    and weights can be specified'''
    # Calculate data averages and standard deviations
    spatial_average1 = calc_area_weighted_spatial_average(data1, lon, lat, masking, area_weight)
    spatial_average2 = calc_area_weighted_spatial_average(data2, lon, lat, masking, area_weight)
    standard_deviation1 = calc_area_weighted_standard_deviation(data1, lon, lat, masking, area_weight)
    standard_deviation2 = calc_area_weighted_standard_deviation(data1, lon, lat, masking, area_weight)
    # Difference from Mean Datasets
    mean_diff_data1 = data1 - spatial_average1
    mean_diff_data2 = data2 - spatial_average2
    combined_data = mean_diff_data1 * mean_diff_data2
    average_mean_diff_product = calc_area_weighted_spatial_average(combined_data, lon, lat, masking, area_weight)
    # Calculate the Correlation
    pearson_correlation_coefficient = average_mean_diff_product / (standard_deviation1 * standard_deviation2)
    
    return pearson_correlation_coefficient

## Chi-squared Test Statistic
def chi_squared_statistic(data, modelData, lon, lat, masking='mask_off'):
    '''This function calculates the Chi-Squared Statistic
    for the goodness of fit between data and modelData. Each
    grid space is assumed to be a degree of freedom. This
    means the total number of degrees of freedom for the chi-
    squared distribution is (lon * lat)'''
    
    if masking == 'mask_ocean':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        masked_model_data = maskoceans(lons,lats,modelData)
        data = masked_data
        modelData = masked_model_data
    if masking == 'mask_land':
        masked_data = maskoceans(lons,lats,data) # masking oceans
        masked_data.mask = ~masked_data.mask # 'inverting' the mask to instead mask land
        masked_model_data = maskoceans(lons,lats,modelData)
        masked_model_data.mask = ~masked_model_data.mask
        data = masked_data
        modelData = masked_model_data
    scaled_squared_diff = ((data - modelData) * (data - modelData)) / modelData
    chi_squared = np.sum(scaled_squared_diff)
    #print('Number of degrees of freedom are',(lon * lat))
    
    return chi_squared

# Line Plotting
def line_plot(x,y,label_num):
    fig = plt.figure(figsize=(11,8))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x,y,label=label_num)
    plt.savefig('correlation'+str(label_num)+'.png')


# In[8]:


lon, lat, date, sample_size, cth_sample_size, ar_shape, misr_u_at_cth, misr_v_at_cth, misr_cth, merra2_u, merra2_v = read_data_wind('West_Coast_AR_and_Wind_18UTC_daily_NOV-APR_2000-2017.nc')


# In[9]:


misr_cth_speed = np.sqrt((misr_u_at_cth * misr_u_at_cth)+(misr_v_at_cth * misr_v_at_cth))
merra2_speed = np.sqrt((merra2_u * merra2_u)+(merra2_v * merra2_v))


# In[25]:


chi_squared_statistic(misr_cth_speed[500,:,:],merra2_speed[500,0,:,:],lon,lat,masking='mask_off')


# In[48]:


np.shape(merra2_speed)
x = np.arange(20)
x


# In[52]:


pearson_correlation(sample_size[750,0,:,:],misr_cth_speed[750,:,:],lon,lat,area_weight=False)


# In[54]:


t_max = 30
correlations = []
time = np.arange(t_max)
for i in range(t_max):
    correlations.append(pearson_correlation(merra2_speed[i,0,:,:],misr_cth_speed[i,:,:],lon,lat,area_weight=False))
line_plot(time,correlations,1)


# In[ ]:




