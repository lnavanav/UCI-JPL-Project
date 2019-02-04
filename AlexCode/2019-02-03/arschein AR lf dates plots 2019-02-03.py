from netCDF4 import Dataset, num2date
import numpy as np
import numpy.ma as ma
import matplotlib as mpl
from matplotlib import cm
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap, maskoceans


########################################## FUNCTIONS ##########################################

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
	return lon, lat, dates, is_land, is_coast, AR_shape, AR_lfloc, MERRA2_speed, MERRA2_U, MERRA2_V, MISR_CTH, MISR_CTH_sample_size, MISR_speed, MISR_speed_at_CTH, MISR_U, MISR_V, MISR_U_at_CTH, MISR_V_at_CTH

#Plots wind speed contours over our west coast region of interest
def plot_local_contour_map(data, lon, lat, levels, figure_file, title='title', cmap=cm.jet): #added title
	lons, lats = np.meshgrid(lon, lat)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(), 
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
	x,y = m(lons, lats)
	m.drawcoastlines(linewidth=1)
	m.drawcountries(linewidth=.75)
	m.drawstates(linewidth=.5)
	max = m.contourf(x, y, data, levels = levels, extend='both', cmap=cmap)
	cbar_ax = fig.add_axes([0.85, 0.3, 0.01, 0.4]) #[0.15, 0.25, 0.3, 0.01])
	cbar_ax.set_xlabel('m/s')
	cb=plt.colorbar(max, cax=cbar_ax)  #orientation = 'horizontal',
	cb.ax.tick_params(labelsize=8)
	ax.set_title(title)
	fig.savefig(figure_file,dpi=600,bbox_inches='tight')
	plt.show()
	
#Plots wind speed contours over our west coast region of interest. Masks the land
def plot_local_contour_map_maskedland(data, lon, lat, plot_mask, levels, figure_file, title='title', cmap=cm.jet): #added title
	lons, lats = np.meshgrid(lon, lat)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(), 
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
	x,y = m(lons, lats)
	m.drawcoastlines(linewidth=1)
	m.drawcountries(linewidth=.75)
	m.drawstates(linewidth=.5)
	#MASKING STUFF START
	masked_data = ma.array(data, mask=plot_mask)
	#MASKING STUFF END
	max = m.contourf(x, y, masked_data, levels = levels, extend='both', cmap=cmap)
	cbar_ax = fig.add_axes([0.85, 0.3, 0.01, 0.4]) #[0.15, 0.25, 0.3, 0.01])
	cbar_ax.set_xlabel('m/s')
	cb=plt.colorbar(max, cax=cbar_ax)  #orientation = 'horizontal',
	cb.ax.tick_params(labelsize=8)
	ax.set_title(title)
	fig.savefig(figure_file,dpi=600,bbox_inches='tight')
	plt.show()	

#Plots wind vectors over our west coast region of interest	
def plot_local_vector_map(uwind, vwind, lon, lat, figure_file='filename', title='title', yskip=2, xskip=3):
	lons, lats = np.meshgrid(lon, lat)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(),
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
	x,y = m(lons, lats)
	m.drawcoastlines(linewidth=1)
	m.drawcountries(linewidth=.75)
	m.drawstates(linewidth=.5)
	N = ma.mean(np.sqrt(uwind[::yskip, ::xskip]**2+vwind[::yskip, ::xskip]**2)) #Obsolete?
	max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], uwind[::yskip, ::xskip]/N, vwind[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
	ax.set_title(title)
	fig.savefig(figure_file,dpi=600,bbox_inches='tight')
	plt.show()
	
#Plots wind vectors over our west coast region of interest. Masked land	
def plot_local_vector_map_maskedland(uwind, vwind, lon, lat, plot_mask, figure_file='filename', title='title', yskip=2, xskip=3):
    lons, lats = np.meshgrid(lon, lat)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    m = Basemap(ax=ax, projection = 'cyl', llcrnrlat = lat.min(), urcrnrlat = lat.max(),
        llcrnrlon = lon.min(), urcrnrlon = lon.max(), resolution = 'h', fix_aspect = True)
    x,y = m(lons, lats)
    m.drawcoastlines(linewidth=1)
    m.drawcountries(linewidth=.75)
    m.drawstates(linewidth=.5)
	#MASKING STUFF START
    masked_u = ma.array(uwind, mask=plot_mask)
	masked_v = ma.array(vwind, mask=plot_mask)
    #MASKING STUFF END
    N = ma.mean(np.sqrt(masked_u[::yskip, ::xskip]**2+masked_v[::yskip, ::xskip]**2)) #Obsolete?
    max = m.quiver(x[::yskip, ::xskip], y[::yskip, ::xskip], masked_u[::yskip, ::xskip]/N, masked_v[::yskip, ::xskip]/N, color='blue', pivot='middle', headwidth=3)
    ax.set_title(title)
    fig.savefig(figure_file,dpi=600,bbox_inches='tight')
    plt.show()
	
#Make the mask for plotting stuff - excludes land but includes the coast	
def build_land_mask(is_land, is_coast):
	temp_mask = ~is_land.mask
	lat, lon = np.shape(is_land)	#is_land and is_coast overlap, so ~is_land and is_coast don't.
	for i in range(lat):			#need to build a mask that allows us to plot over the ocean and the coasts but NOT the land.
		for j in range(lon):		
			if is_coast[i,j] == 1:
				temp_mask[i,j] = False	#NOTE: "false" allows more of the ocean/coast region to be plotted. If "true" then it's just is_land.mask.
	return temp_mask
	
#Get the landfall dates. Not the most efficient way, I think, but at least it works for now	
def get_lfloc_dates(lfloc):
	t_list = []
	for t in range(len(lfloc)):
		if ma.mean(lfloc[t,:]) == 1: #there was landfall on that day - don't care about location
			t_list.append(t)
	return t_list
	


########################################## VARIABLES ##########################################

##Get all data
lon, lat, dates, is_land, is_coast, AR_shape, AR_lfloc = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[0:7]
MERRA2_speed, MERRA2_U, MERRA2_V = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[7:10]
MISR_CTH, MISR_CTH_sample_size, MISR_speed, MISR_speed_at_CTH, MISR_U, MISR_V, MISR_U_at_CTH, MISR_V_at_CTH = read_AR_winds('West_Coast_AR_and_Wind_18UTC_daily_NOV2000-APR2017.nc')[10:]

land_mask_for_plotting = build_land_mask(is_land, is_coast)

##Time indicies
#Find all dates of landfall - returns list of dates
lfloc_dates = get_lfloc_dates(AR_lfloc) #should be 1,022 days of landfall

#CHANGES FOR THE DIFFERENCE: now taking the mean of the subtracted vectors rather than difference of the means. There shouldn't be a need to define specific masks - python should automatically mask the arrays as they're subtracted,
#because subtraction is a binary operation for masked arrays (http://folk.uio.no/inf3330/scripting/doc/python/NumPy/Numeric/numpy-22.html)

##Speeds and vectors

#Overall
merra_overall_speed_1p5km = ma.mean(MERRA2_speed[:, 0, :, :], axis=0)
merra_overall_speed_2p5km = ma.mean(MERRA2_speed[:, 1, :, :], axis=0)
merra_overall_speed_4km = ma.mean(MERRA2_speed[:, 2, :, :], axis=0)
merra_overall_u_1p5km = ma.mean(MERRA2_U[:, 0, :, :], axis=0)
merra_overall_v_1p5km = ma.mean(MERRA2_V[:, 0, :, :], axis=0)
merra_overall_u_2p5km = ma.mean(MERRA2_U[:, 1, :, :], axis=0)
merra_overall_v_2p5km = ma.mean(MERRA2_V[:, 1, :, :], axis=0)
merra_overall_u_4km = ma.mean(MERRA2_U[:, 2, :, :], axis=0)
merra_overall_v_4km = ma.mean(MERRA2_V[:, 2, :, :], axis=0)
misr_overall_speed_1p5km = ma.mean(MISR_speed[:, 0, :, :], axis=0)
misr_overall_speed_2p5km = ma.mean(MISR_speed[:, 1, :, :], axis=0)
misr_overall_speed_4km = ma.mean(MISR_speed[:, 2, :, :], axis=0)
misr_overall_u_1p5km = ma.mean(MISR_U[:, 0, :, :], axis=0)
misr_overall_v_1p5km = ma.mean(MISR_V[:, 0, :, :], axis=0)
misr_overall_u_2p5km = ma.mean(MISR_U[:, 1, :, :], axis=0)
misr_overall_v_2p5km = ma.mean(MISR_V[:, 1, :, :], axis=0)
misr_overall_u_4km = ma.mean(MISR_U[:, 2, :, :], axis=0)
misr_overall_v_4km = ma.mean(MISR_V[:, 2, :, :], axis=0)
overall_speed_diff_1p5km = ma.mean(MERRA2_speed[:, 0, :, :] - MISR_speed[:, 0, :, :], axis=0)
overall_speed_diff_2p5km = ma.mean(MERRA2_speed[:, 1, :, :] - MISR_speed[:, 1, :, :], axis=0)
overall_speed_diff_4km = ma.mean(MERRA2_speed[:, 2, :, :] - MISR_speed[:, 2, :, :], axis=0)
overall_u_diff_1p5km = ma.mean(MERRA2_U[:, 0, :, :] - MISR_U[:, 0, :, :], axis=0)
overall_v_diff_1p5km = ma.mean(MERRA2_V[:, 0, :, :] - MISR_V[:, 0, :, :], axis=0)
overall_u_diff_2p5km = ma.mean(MERRA2_U[:, 1, :, :] - MISR_U[:, 1, :, :], axis=0)
overall_v_diff_2p5km = ma.mean(MERRA2_V[:, 1, :, :] - MISR_V[:, 1, :, :], axis=0)
overall_u_diff_4km = ma.mean(MERRA2_U[:, 2, :, :] - MISR_U[:, 2, :, :], axis=0)
overall_v_diff_4km = ma.mean(MERRA2_V[:, 1, :, :] - MISR_V[:, 1, :, :], axis=0)

#Landfall dates only
merra_landfall_speed_1p5km = ma.mean(MERRA2_speed[lfloc_dates, 0, :, :], axis=0)
merra_landfall_speed_2p5km = ma.mean(MERRA2_speed[lfloc_dates, 1, :, :], axis=0)
merra_landfall_speed_4km = ma.mean(MERRA2_speed[lfloc_dates, 2, :, :], axis=0)
merra_landfall_u_1p5km = ma.mean(MERRA2_U[lfloc_dates, 0, :, :], axis=0)
merra_landfall_v_1p5km = ma.mean(MERRA2_V[lfloc_dates, 0, :, :], axis=0)
merra_landfall_u_2p5km = ma.mean(MERRA2_U[lfloc_dates, 1, :, :], axis=0)
merra_landfall_v_2p5km = ma.mean(MERRA2_V[lfloc_dates, 1, :, :], axis=0)
merra_landfall_u_4km = ma.mean(MERRA2_U[lfloc_dates, 2, :, :], axis=0)
merra_landfall_v_4km = ma.mean(MERRA2_V[lfloc_dates, 2, :, :], axis=0)
misr_landfall_speed_1p5km = ma.mean(MISR_speed[lfloc_dates, 0, :, :], axis=0)
misr_landfall_speed_2p5km = ma.mean(MISR_speed[lfloc_dates, 1, :, :], axis=0)
misr_landfall_speed_4km = ma.mean(MISR_speed[lfloc_dates, 2, :, :], axis=0)
misr_landfall_u_1p5km = ma.mean(MISR_U[lfloc_dates, 0, :, :], axis=0)
misr_landfall_v_1p5km = ma.mean(MISR_V[lfloc_dates, 0, :, :], axis=0)
misr_landfall_u_2p5km = ma.mean(MISR_U[lfloc_dates, 1, :, :], axis=0)
misr_landfall_v_2p5km = ma.mean(MISR_V[lfloc_dates, 1, :, :], axis=0)
misr_landfall_u_4km = ma.mean(MISR_U[lfloc_dates, 2, :, :], axis=0)
misr_landfall_v_4km = ma.mean(MISR_V[lfloc_dates, 2, :, :], axis=0)
landfall_speed_diff_1p5km = ma.mean(MERRA2_speed[lfloc_dates, 0, :, :] - MISR_speed[lfloc_dates, 0, :, :], axis=0)
landfall_speed_diff_2p5km = ma.mean(MERRA2_speed[lfloc_dates, 1, :, :] - MISR_speed[lfloc_dates, 1, :, :], axis=0)
landfall_speed_diff_4km = ma.mean(MERRA2_speed[lfloc_dates, 2, :, :] - MISR_speed[lfloc_dates, 2, :, :], axis=0)
landfall_u_diff_1p5km = ma.mean(MERRA2_U[lfloc_dates, 0, :, :] - MISR_U[lfloc_dates, 0, :, :], axis=0)
landfall_v_diff_1p5km = ma.mean(MERRA2_V[lfloc_dates, 0, :, :] - MISR_V[lfloc_dates, 0, :, :], axis=0)
landfall_u_diff_2p5km = ma.mean(MERRA2_U[lfloc_dates, 1, :, :] - MISR_U[lfloc_dates, 1, :, :], axis=0)
landfall_v_diff_2p5km = ma.mean(MERRA2_V[lfloc_dates, 1, :, :] - MISR_V[lfloc_dates, 1, :, :], axis=0)
landfall_u_diff_4km = ma.mean(MERRA2_U[lfloc_dates, 2, :, :] - MISR_U[lfloc_dates, 2, :, :], axis=0)
landfall_v_diff_4km = ma.mean(MERRA2_V[lfloc_dates, 1, :, :] - MISR_V[lfloc_dates, 1, :, :], axis=0)


########################################## PLOTS ##########################################

####Contour plots####
###Speed Differences

#NOTE: MERRA and MISR speeds are both very high (>8 m/s) at all altitudes so their plots would need a different scale. I'm lazy so those lines are just commented out for now.

##Overall
# #MERRA
# plot_local_contour_map_maskedland(merra_overall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra overall speed 1p5km', title='MERRA wind speed, overall, 1.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(merra_overall_speed_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra overall speed 2p5km', title='MERRA wind speed, overall, 2.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(merra_overall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra overall speed 1p5km', title='MERRA wind speed, overall, 1.5 km', cmap=cm.RdBu_r)
# #MISR
# plot_local_contour_map_maskedland(misr_overall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='misr overall speed 1p5km', title='MISR wind speed, overall, 1.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(misr_overall_speed_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='misr overall speed 2p5km', title='MISR wind speed, overall, 2.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(misr_overall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='misr overall speed 1p5km', title='MISR wind speed, overall, 1.5 km', cmap=cm.RdBu_r)
#Difference
plot_local_contour_map_maskedland(overall_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr overall speed difference 1p5km', title='MERRA - MISR wind speed difference, overall, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(overall_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr overall speed difference 2p5km', title='MERRA - MISR wind speed difference, overall, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(overall_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr overall speed difference 4km', title='MERRA - MISR wind speed difference, overall, 4 km', cmap=cm.RdBu_r)

##Landfall days only
#MERRA
# plot_local_contour_map_maskedland(merra_landfall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra landfall speed 1p5km', title='MERRA wind speed, days of AR landfall, 1.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(merra_landfall_speed_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra landfall speed 2p5km', title='MERRA wind speed, days of AR landfall, 2.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(merra_landfall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra landfall speed 1p5km', title='MERRA wind speed, days of AR landfall, 1.5 km', cmap=cm.RdBu_r)
# #MISR
# plot_local_contour_map_maskedland(misr_landfall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='misr landfall speed 1p5km', title='MISR wind speed, days of AR landfall, 1.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(misr_landfall_speed_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='misr landfall speed 2p5km', title='MISR wind speed, days of AR landfall, 2.5 km', cmap=cm.RdBu_r)
# plot_local_contour_map_maskedland(misr_landfall_speed_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='misr landfall speed 1p5km', title='MISR wind speed, days of AR landfall, 1.5 km', cmap=cm.RdBu_r)
#Difference
plot_local_contour_map_maskedland(landfall_speed_diff_1p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr landfall speed difference 1p5km', title='MERRA - MISR wind speed difference, days of AR landfall, 1.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(landfall_speed_diff_2p5km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr landfall speed difference 2p5km', title='MERRA - MISR wind speed difference, days of AR landfall, 2.5 km', cmap=cm.RdBu_r)
plot_local_contour_map_maskedland(landfall_speed_diff_4km, lon, lat, land_mask_for_plotting, np.arange(21)*0.5-5, figure_file='merra-misr landfall speed difference 4km', title='MERRA - MISR wind speed difference, days of AR landfall, 4 km', cmap=cm.RdBu_r)


####Vector plots####
#Note: not doing shears for this - can do later if there's any interest but tbh I don't think many of these plots will be that useful

##Overall
#MERRA
plot_local_vector_map_maskedland(merra_overall_u_1p5km, merra_overall_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors overall 1p5km', title='MERRA wind vectors, Overall, 1.5 km')
plot_local_vector_map_maskedland(merra_overall_u_2p5km, merra_overall_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors overall 2p5km', title='MERRA wind vectors, Overall, 2.5 km')
plot_local_vector_map_maskedland(merra_overall_u_4km, merra_overall_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors overall 4km', title='MERRA wind vectors, Overall, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_overall_u_1p5km, misr_overall_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors overall 1p5km', title='MISR wind vectors, Overall, 1.5 km')
plot_local_vector_map_maskedland(misr_overall_u_2p5km, misr_overall_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors overall 2p5km', title='MISR wind vectors, Overall, 2.5 km')
plot_local_vector_map_maskedland(misr_overall_u_4km, misr_overall_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors overall 4km', title='MISR wind vectors, Overall, 4 km')
#Difference
plot_local_vector_map_maskedland(overall_u_diff_1p5km, overall_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vector difference overall 1p5km', title='MERRA-MISR wind vector difference, Overall, 1.5 km')
plot_local_vector_map_maskedland(overall_u_diff_2p5km, overall_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vector difference overall 2p5km', title='MERRA-MISR wind vector difference, Overall, 2.5 km')
plot_local_vector_map_maskedland(overall_u_diff_4km, overall_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vector difference overall 4km', title='MERRA-MISR wind vector difference, Overall, 4 km')

##Landfall
#MERRA
plot_local_vector_map_maskedland(merra_landfall_u_1p5km, merra_landfall_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors landfall 1p5km', title='MERRA wind vectors, landfall, 1.5 km')
plot_local_vector_map_maskedland(merra_landfall_u_2p5km, merra_landfall_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra vectors landfall 2p5km', title='MERRA wind vectors, landfall, 2.5 km')
plot_local_vector_map_maskedland(merra_landfall_u_4km, merra_landfall_v_4km, lon, lat, land_mask_for_plotting, figure_file='merra vectors landfall 4km', title='MERRA wind vectors, landfall, 4 km')
#MISR
plot_local_vector_map_maskedland(misr_landfall_u_1p5km, misr_landfall_v_1p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors landfall 1p5km', title='MISR wind vectors, landfall, 1.5 km')
plot_local_vector_map_maskedland(misr_landfall_u_2p5km, misr_landfall_v_2p5km, lon, lat, land_mask_for_plotting, figure_file='misr vectors landfall 2p5km', title='MISR wind vectors, landfall, 2.5 km')
plot_local_vector_map_maskedland(misr_landfall_u_4km, misr_landfall_v_4km, lon, lat, land_mask_for_plotting, figure_file='misr vectors landfall 4km', title='MISR wind vectors, landfall, 4 km')
#Difference
plot_local_vector_map_maskedland(landfall_u_diff_1p5km, landfall_v_diff_1p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vector difference landfall 1p5km', title='MERRA-MISR wind vector difference, landfall, 1.5 km')
plot_local_vector_map_maskedland(landfall_u_diff_2p5km, landfall_v_diff_2p5km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vector difference landfall 2p5km', title='MERRA-MISR wind vector difference, landfall, 2.5 km')
plot_local_vector_map_maskedland(landfall_u_diff_4km, landfall_v_diff_4km, lon, lat, land_mask_for_plotting, figure_file='merra-misr vector difference landfall 4km', title='MERRA-MISR wind vector difference, landfall, 4 km')


