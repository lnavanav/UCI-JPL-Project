__init__.py
#################################################################
## Read me :D
## ############
## In collaboration with the University Crowdsourcing Initiative, we UCI
## students have been able to work in collaboration with JPL/NASA
## This is a library of code that has been created for the purpose of
## comparing windspeeds from MISR vs reanalysis programs (e.g. MERRA-2)
## Everything was done for the pupose of education.
## All information is public and not classified!
## Feel free to use as will, a thank you would be nice!
## Colaborators: Alexander Schein, Tejas Dethe, Luis Nava-Navarro
## Github: (https://github.com/lnavanav/UCI-JPL-Project)

from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as pyt
from mpl_toolkits.basemap import Basemap, maskoceans
# Testing

##################################################################################################################
############################################ DEFINITIONS #########################################################
##################################################################################################################

##################################################################################################################
## Definition: read_AR_winds
##  * Just reads file onto variables
## Arguments: filename
## Returns: lon, lat, dates, is_land, etc. - AR wind data
## Notes: Includes both regular sample size and CTH sample size
##################################################################################################################
def read_AR_winds(filename):
    f = Dataset(filename)
    lon = f.variables['lon'][:]
    lat = f.variables['lat'][:]
    time = f.variables['time']
    dates = num2date(time[:], time.units)
    is_land = f.variables['islnd'][:]
    is_coast = f.variables['iscst'][:]
    AR_shape = f.variables['AR_shape'][:]
    AR_lfloc = f.variables['AR_lfloc'][:]
    MERRA2_speed = f.variables['MERRA2_speed'][:]
    MERRA2_U = f.variables['MERRA2_U'][:]
    MERRA2_V = f.variables['MERRA2_V'][:]
    MISR_CTH = f.variables['MISR_CTH'][:]
    MISR_CTH_sample_size = f.variables['MISR_CTH_sample_size'][:]
    MISR_speed = f.variables['MISR_speed'][:]
    MISR_speed_at_CTH = f.variables['MISR_speed_at_CTH'][:]
    MISR_U = f.variables['MISR_U'][:]
    MISR_V = f.variables['MISR_V'][:]
    MISR_U_at_CTH = f.variables['MISR_U_at_CTH'][:]
    MISR_V_at_CTH = f.variables['MISR_V_at_CTH'][:]
    MISR_sample_size = f.variables['MISR_sample_size'][:]
    return lon, lat, dates, is_land, is_coast, AR_shape, AR_lfloc, MERRA2_speed, MERRA2_U, MERRA2_V, MISR_CTH, MISR_CTH_sample_size, MISR_speed, MISR_speed_at_CTH, MISR_U, MISR_V, MISR_U_at_CTH, MISR_V_at_CTH, MISR_sample_size


##################################################################################################################
## Definition: get_lfloc_dates
##  * We want to be able to know when landfall was made, not too
##    worried about where.
## Arguments: lfloc - The land fall locations
## Returns: t_list - a list of dates when landfall occured
## Notes:
##################################################################################################################
def get_lfloc_dates(lfloc):
    t_list = []
    for t in range(len(lfloc)):
        if ma.mean(lfloc[t,:]) == 1: #there was landfall on that day - don't care about location
            t_list.append(t)
    return t_list


##################################################################################################################
## Definition: get_lfloc_dates
##  * We want to be able to know when landfall was made, not too
##    worried about where.
## Arguments: lfloc - The land fall locations
## Returns: t_list - a list of dates when landfall occured
## Notes:
##################################################################################################################
def build_land_mask(is_land, is_coast):
    temp_mask = ~is_land.mask
    lat, lon = np.shape(is_land)    #is_land and is_coast overlap, so ~is_land and is_coast don't.
    for i in range(lat):            #need to build a mask that allows us to plot over the ocean and the coasts but NOT the land.
        for j in range(lon):
            if is_coast[i,j] == 1:
                temp_mask[i,j] = False    #NOTE: "false" allows more of the ocean/coast region to be plotted. If "true" then it's just is_land.mask.
    return temp_mask



##################################################################################################################
## Definition: plot_local_contour_map
##  * Plots wind speed contours over the west coast region of interest
## Arguments: data, lon, lat, levels, figureFile, title, cmap
## Returns: Nothing. 'void' definition. plt.show the plot contour of local area (Calif.)
## Notes: * Added msked boolean in argument, now you are able to selected the plot to be masked or not.
##################################################################################################################
def plot_local_contour_map(data, lon, lat, levels, figure_file='test', title='title', cmap=cm.jet, plot_mask, msked):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(),
                llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates(linewidth=.5)
    if msked == True: # If you want to mask, then just pass msked as true.
        masked_data = ma.array(data, mask=plot_mask)
    max = m.contourf(x, y, data, levels = levels, extend='both', cmap=cmap)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.01, 0.4]) #[0.15, 0.25, 0.3, 0.01])
    cbar_ax.set_xlabel('# of \n days')
    cb=plt.colorbar(max, cax=cbar_ax)  #orientation = 'horizontal',
    cb.ax.tick_params(labelsize=8)
    ax.set_title(title)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()


##################################################################################################################
## Definition: plot_local_vector_map
##  * Plots wind speed vectors over the west coast region of interest
## Arguments: data, lon, lat, levels, figureFile, title, cmap
## Returns: Nothing. 'void' definition. plt.show the plot contour of local area (Calif.)
## Notes: * Added msked boolean in argument, now you are able to selected the plot to be masked or not.
##################################################################################################################
def plot_local_vector_map(uwind, vwind, lon, lat, figure_file='filename', title='title', yskip=2, xskip=3, plot_mask, msked):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(),
                llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'l', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates(linewidth=.5)
    ## masking stuff start
    if msked == True:
        masked_u = ma.array(uwind, mask=plot_mask)
        masked_v = ma.array(vwind, mask=plot_mask)
    ## end
    N = ma.mean(np.sqrt(uwind[::yskip, ::xskip]**2+vwind[::yskip, ::xskip]**2)) #Obsolete?
    max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], uwind[::yskip, ::xskip]/N, vwind[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
    ax.set_title(title)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()


##################################################################################################################
## Definition: GetSampleSizeAndCTH
##  * Retrives the sample size per point and the cloud top height
##    of the point.
## Arguments: samplesize_data,CTH_data
## Returns: number of samples and cloud top heights
## Notes: 'Unspools' the cloud top height and sample size data into two MATCHED arrays so that they can be plotted
##        and analysed.
##        The important part is that the arrays are matched, so the same point in both arrays refer to the same
##        time, latitude, and longitude
##  * May look into more efficient way to do this.
##################################################################################################################
def GetSampleSizeAndCTH(samplesize_data, CTH_data):
    num_of_samples = []
    cloud_top_height = []
    #num_samples_and_CTH = []
    for t in range(np.shape(CTH_data)[0]):
        for i in range(np.shape(CTH_data)[1]):
            for j in range(np.shape(CTH_data)[2]):
                if samplesize_data.mask[t,i,j] == False and CTH_data.mask[t,i,j] == False: #two might be redundant
                    num_of_samples.append(samplesize_data[t,i,j])
                    cloud_top_height.append(CTH_data[t,i,j])
    #num_samples_and_CTH.append((samplesize_data[t,i,j], CTH_data[t,i,j]))
return num_of_samples, cloud_top_height #num_samples_and_CTH


##################################################################################################################
## Definition: SortSampleSizeIntoHeightBins
##  * We want to be able to retrieve the number of samples for each tier.
## Arguments: samplesize_data, CTH_data
## Returns: binned_sample_size, binned_avg_num_of_samples
## Notes: * binned_sample_size - is a 1x17 array where each entry is the total number f samples for each
##          1 km height bin: from 0-1 km to 16-17 km. Intervals are [a,b).
##        * binned_avg_num_of_samples - average number of samples for each height bin.
##################################################################################################################
def SortSampleSizeIntoHeightBins(samplesize_data, CTH_data):
    binned_sample_size = np.zeros(17) #17 height bins
    binned_num_of_observations = np.zeros(17)
    binned_avg_num_of_samples = np.zeros(17)
    #Doing bad coding practice here and hardcoding the height bins into 1 km sections
    for i in range(np.shape(samplesize_data)[0]):
        if CTH_data[i] >= 0:
            if CTH_data[i] < 1:
                binned_sample_size[0] += samplesize_data[i]
                binned_num_of_observations[0] += 1
            elif CTH_data[i] >= 1 and CTH_data[i] < 2:
                binned_sample_size[1] += samplesize_data[i]
                binned_num_of_observations[1] += 1
            elif CTH_data[i] >= 2 and CTH_data[i] < 3:
                binned_sample_size[2] += samplesize_data[i]
                binned_num_of_observations[2] += 1
            elif CTH_data[i] >= 3 and CTH_data[i] < 4:
                binned_sample_size[3] += samplesize_data[i]
                binned_num_of_observations[3] += 1
            elif CTH_data[i] >= 4 and CTH_data[i] < 5:
                binned_sample_size[4] += samplesize_data[i]
                binned_num_of_observations[4] += 1
            elif CTH_data[i] >= 5 and CTH_data[i] < 6:
                binned_sample_size[5] += samplesize_data[i]
                binned_num_of_observations[5] += 1
            elif CTH_data[i] >= 6 and CTH_data[i] < 7:
                binned_sample_size[6] += samplesize_data[i]
                binned_num_of_observations[6] += 1
            elif CTH_data[i] >= 7 and CTH_data[i] < 8:
                binned_sample_size[7] += samplesize_data[i]
                binned_num_of_observations[7] += 1
            elif CTH_data[i] >= 8 and CTH_data[i] < 9:
                binned_sample_size[8] += samplesize_data[i]
                binned_num_of_observations[8] += 1
            elif CTH_data[i] >= 9 and CTH_data[i] < 10:
                binned_sample_size[9] += samplesize_data[i]
                binned_num_of_observations[9] += 1
            elif CTH_data[i] >= 10 and CTH_data[i] < 11:
                binned_sample_size[10] += samplesize_data[i]
                binned_num_of_observations[10] += 1
            elif CTH_data[i] >= 11 and CTH_data[i] < 12:
                binned_sample_size[11] += samplesize_data[i]
                binned_num_of_observations[11] += 1
            elif CTH_data[i] >= 12 and CTH_data[i] < 13:
                binned_sample_size[12] += samplesize_data[i]
                binned_num_of_observations[12] += 1
            elif CTH_data[i] >= 13 and CTH_data[i] < 14:
                binned_sample_size[13] += samplesize_data[i]
                binned_num_of_observations[13] += 1
            elif CTH_data[i] >= 14 and CTH_data[i] < 15:
                binned_sample_size[14] += samplesize_data[i]
                binned_num_of_observations[14] += 1
            elif CTH_data[i] >= 15 and CTH_data[i] < 16:
                binned_sample_size[15] += samplesize_data[i]
                binned_num_of_observations[15] += 1
            elif CTH_data[i] >= 16 and CTH_data[i] < 17:
                binned_sample_size[16] += samplesize_data[i]
                binned_num_of_observations[16] += 1
    for j in range(np.shape(binned_sample_size)[0]):
        binned_avg_num_of_samples[j] = binned_sample_size[j]/binned_num_of_observations[j]
    return binned_sample_size, binned_avg_num_of_samples




###################################################################################################################
## SELF NOTES
##  * Maybe make the plot local vector/contour definitions into one function instead of two,
##      - This function would take in the combination of both of the arguments
##      - Will include an extra arguments that asks the user to imput the particular type of plot thet
##        would like to see. This ways it will be more effictient and the names will not be as long.
