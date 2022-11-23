# -*- coding: utf-8 -*-
"""
Wellington Tree Analysis 
 - Grace Milner 

This script has been developed to link tree point records with tree crown 
polygon records and assess the certainty in the relationship. 

The main goal was to associate the information held within tree point records
(such as species and age) with tree crowns identified from remote imagery, 
dependent on the certainty of the relationship. Such data could then be used as 
training data in the future. Information held within tree crown records 
(such as height, obtained through remote sensing techniques) can also be used
to update the tree point records. 

Required input data is:
    - Shapefile with tree records represented by points 
    - Shapefile with tree crowns represented by polygons
    
Outputs include: 
    - Report in the form of a HTML table with data from both tree point and 
      crown records, with additional contextual imagery
    - Various options throughout script to generate excel files where desired
    - Updated version of input tree dataset, with added information obtained
      from tree crown dataset for each tree (where certainty threshold met)
    
"""
########################################################
######################## SET UP ########################
########################################################

#Loading libraries
#-----------------
import geopandas as gpd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import rasterio
from rasterio.mask import mask
import base64, io, urllib #for displaying plots in table

#from pyproj import CRS  #for checking CRS of sat img if needed


#Reading in data
#---------------
"""
# (For developing script, will just use subset of data. Uncomment if needed)
#Load bbox for script testing, change crs
test_bbox_path = PATH TO BBOX
test_bbox = gpd.read_file(test_bbox_path)
test_bbox = test_bbox.to_crs(2193) #to match other data
"""

# (All) Tree crown polygons
# Change to relevant file path for tree crown polygons
crown_path = PATH TO CROWNS
all_crowns = gpd.read_file(crown_path)
# (All) City Council tree records (replace path for tree point data)
#           (will also need to replace names of columns required for final dataframe dependant on information available
#            e.g. height, girth, age etc.)
trees_path = PATH TO TREE RECORDS
all_trees = gpd.read_file(trees_path)


# (Clipped) Tree crowns (for testing)
#crowns_df = all_crowns.clip(test_bbox)
# (Clipped) City Council trees (for testing)
#trees_df = all_trees.clip(test_bbox)

# When using all data:
crowns_df = all_crowns
trees_df = all_trees

# Satellite imagery
sat_img = rasterio.open(PATH TO IMAGERY)

#CRS.from_string(sat_img.GetProjection())   #can use to check CRS of sat image before use

#########################################################
###################### MAIN SCRIPT ######################
#########################################################

# Initial analysis
#-----------------

# CHECKING IF POINT (TREE) WITHIN POLYGON (TREE CROWN)

# First, copying polygon geom (geometry info of tree crowns) to new column so it will be retained during spatial join 
crowns_df['poly_geom'] = crowns_df.geometry

# Performing spatial join to merge polygons (crowns) with points (trees)
#       New column 'index right' indicates ID of polygon each point is within (if any)
#       and 'poly_geom' holds crown polygon geometry
trees_crowns_df = gpd.sjoin(trees_df, crowns_df, how="left")

# Making new column with True = within a polygon and False = not within a polygon
trees_crowns_df['in_crown'] = pd.notna(trees_crowns_df['index_right'])


# CALCULATING NUMBER OF POINTS WITHIN POLYGON

# Add new column w/ total number of points in associated polygon
# First create new table with counts 
# (creates dataframe with polygon ID and counts of points within it)
crown_point_counts = trees_crowns_df.groupby('index_right').size().reset_index(name='counts') 

# Then add counts as new column in the main trees/crowns dataframe
# (gives the total number of tree points within each polygon which is associated with at least one point)
trees_crowns_df['points_in_poly'] = trees_crowns_df['index_right'].map(crown_point_counts.set_index('index_right')['counts'])


# CALCULATING DISTANCE TO POLYGON EDGE (FOR POINTS WITHIN A POLYGON)

# Calculate minimum distance to polygon edge, nan if point not within a polygon
trees_crowns_df['edge_dist'] = trees_crowns_df.apply(lambda row : row['geometry'].distance(row['poly_geom'].boundary) 
                                                    if pd.notnull(row['poly_geom']) 
                                                    else np.nan, axis=1)

# CALCULATING DISTANCE TO POLYGON CENTROID (FOR POINTS WITHIN A POLYGON)

# Minimum distance to polygon centroid, nan if point not within a polygon
trees_crowns_df['centr_dist'] = trees_crowns_df.apply(lambda row : row['geometry'].distance(row['poly_geom'].centroid) 
                                                    if pd.notnull(row['poly_geom']) 
                                                    else np.nan, axis=1)

# DETERMINING POSITION WITHIN POLYGON 

# To make points comparable between tree crowns of different sizes, 
# use distances to centroid/edge to determine general position of points within each crown
#    distance to centroid minus distance to edge -->
#         positive values = closer to edge (worse, lower confidence) 
#         exactly zero = point equal distance from edge and centroid 
#         negative values = closer to centroid (better, higher confidence)

trees_crowns_df['position'] = trees_crowns_df['centr_dist'] - trees_crowns_df['edge_dist']


# CALCULATING DISTANCE TO NEAREST POLYGON (FOR POINTS NOT WITHIN A POLYGON)
# Retaining geometry information for nearest polygons

# First selecting only required columns
point_geoms_df = trees_df[['OBJECTID', 'geometry']] 

# Performing spatial join based on nearest polygon to each point and recording value of minimum distance
#       Only if distance to nearest polygon is equal to or less than 10m 
nearest_join_df = point_geoms_df.sjoin_nearest(crowns_df, distance_col='near_dist', max_distance = 10)

# Converting '0' values to nan (for points that are within a polygon)
nearest_join_df['near_dist'] = nearest_join_df['near_dist'].replace(0, np.nan)

# Joining nearest distance column to main dataframe
trees_crowns_df = trees_crowns_df.join(nearest_join_df['near_dist'])
# Joining geometry of nearest polygon to main dataframe
trees_crowns_df = trees_crowns_df.join(nearest_join_df['poly_geom'], lsuffix='_left', rsuffix='_right')
# Renaming to keep everything clear
trees_crowns_df.rename(columns={'poly_geom_left': 'in_poly', 'poly_geom_right': 'near_poly'}, inplace=True)




# Certainty Score
#----------------
# Summarising the calculated indicators into a single measure to allow direct comparison of confidence 
# and guide prioritisation for manual checks. 

# Creating function for standardising values into 0-1 range using min and max

def standardise(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

# Standardising each score into a 0-1 range and recording in new dataframe
# Also, subtracting standardised value from 1 to make larger numbers indicate more certainty
# (more intuitive to understand, and easier to work with weights)
#       Before, smaller numbers indicated more certainty in that point due to:
#           -fewer other points in polygon
#           -smaller distances from centroid
#           -smaller distances from nearest crown
#       Now, larger numbers indicates more certainty
cert_df = pd.DataFrame()
cert_df['z_PinP'] = standardise(trees_crowns_df['points_in_poly']).apply(lambda x: 1-x)
cert_df['z_position'] = standardise(trees_crowns_df['position']).apply(lambda x: 1-x)
cert_df['z_near_dis'] = standardise(trees_crowns_df['near_dist']).apply(lambda x: 1-x)

# Combining into weighted average

# Defining each weight [points in polygon, position within polygon, nearest distance (to closest polygon)]
# Here, distance from nearest polygon needs to be given less weighting otherwise it showed too much importance in the final score

weights = [0.4, 0.4, 0.2]

# Multiplying each column by respective weights, summing, and dividing by number of columns used 
cert_df['cert_score'] = (cert_df.mul(weights).sum(axis=1)) / cert_df.count(axis=1)

# Standardising so the certainty score falls between 0 and 1 / stretch the values to fall across that entire range of 0-1
cert_df['cert_score'] = standardise(cert_df['cert_score'])



# Tidying and Sub-Setting
#------------------------

#TIDY
# Filling in nan rows of species names from second botanical names column (combining to tidy)
trees_crowns_df['botanical_'] = trees_crowns_df['botanical_'].fillna(trees_crowns_df['botanica_1'])


# Generating cleaner dataframe with only required columns
# Change/add/remove column names depending on available information in input data
clean_df = trees_crowns_df[['OBJECTID',
                            'botanical_', 
                            'height', 
                            'girth', 
                            'age', 
                            'treeheight', 
                            'area', 
                            'mean_diameter', 
                            'minor_axis_length', 
                            'major_axis_length', 
                            'geometry', 
                            'in_poly', 
                            'in_crown', 
                            'points_in_poly', 
                            'position', 
                            'near_dist', 
                            'near_poly']]

clean_df['cert_score'] = cert_df['cert_score']

# Since there are two columns relating to tree height, one from original City Council tree point data
# and one measured from remote imagery, will rename the remotely-sensed height info
clean_df.rename(columns={'treeheight': 'new_height'}, inplace=True)


# Removing crown information obtained from remote imagery if confidence score is below a threshold
#     (additional height, area etc. info available, but should only be joined with tree records if sufficiently confident)
#define certainty threshold, above which you are willing to accept tree information obtained from remote imagery
cert_thresh = 0.6 

#removing information for trees where certainty score is below threshold        
clean_df.loc[clean_df['cert_score'] < cert_thresh, ['new_height', 
                                                    'area', 
                                                    'mean_diameter', 
                                                    'minor_axis_length', 
                                                    'major_axis_length']] = np.nan

#Updating original data
#----------------------
# Merging copy of original trees dataframe with new information gained from remote imagery

#making copy of original trees dataframe
trees_df_new = trees_df.copy()
#merging copy with relevant columns from remote imagery and previous calculations
trees_df_new = trees_df_new.merge(clean_df[['OBJECTID', 
                                            'new_height', 
                                            'area', 
                                            'mean_diameter', 
                                            'minor_axis_length', 
                                            'major_axis_length', 
                                            'cert_score']], 
                                              on ='OBJECTID', how='left')

# Exporting to csv (if needed)
#trees_df_new.to_csv('trees_data_new.csv')


# Subset
#-------
# Selecting only rows with top certainty score for further investigation
#   (reduces unnecessary processing since we are only interested in the most certain points)
# Ordering based on certainty score
clean_df = clean_df.sort_values(by='cert_score', ascending=False)

# Other option for subsetting, un-comment as required: 
# Selecting only records where age classified as 'Veteran', then sorting by certainty. Can change age classification, 
# or select based on other variables e.g. species. 
#clean_df = clean_df[clean_df['age'] == 'Veteran (51+)'].sort_values(by='cert_score', ascending=False)

# Creating new dataframe just of top 100, can change n as required.
highcert_df = clean_df.head(n=100)


# Statistical analysis
#---------------------
# un-comment if wanting to see results (for all points, not just subset created previously)
'''
# Calculating percentage of trees in different age groups that are found within crown polygon
perc_age = clean_df.groupby('age')['in_crown'].apply(lambda x: np.sum(x)/len(x))
print(perc_age)
#(shows that youngest trees picked up less frequently by crown algorithm - higher chance of being missed in imagery)

# Grouping by botanical name
#counting number of entries for each species
species_count = clean_df.groupby(['botanical_']).OBJECTID.agg('count').sort_values(ascending=False)
print(species_count)
#number of trees found within crown polygons, per species (shows most commonly identified species)
species_crowns = clean_df.groupby('botanical_')['in_crown'].agg('sum').sort_values(ascending=False)
print(species_crowns)
'''


# Generating report 
#------------------
# Goes through previous subset of rows with the highest cert scores, creates images showing RGB image, points, and polygons.

#adding empty array
images = []

for i, row in highcert_df.iterrows():
    
    #getting required spatial information for creating image/plot figure
    point_geom = gpd.GeoSeries((row['geometry'])) #saving the point geom info
    if pd.notnull(row['in_poly']): 
        in_poly_geom = gpd.GeoSeries((row['in_poly'])) #saving poly geom info
        centroid = gpd.GeoSeries(row['in_poly'].centroid) #getting centroid point
        bbox = gpd.GeoSeries(centroid.buffer(18, cap_style = 3)) #creating bbox around poly centroid
    else:
        bbox = point_geom.buffer(18, cap_style = 3) # creating square bounding box around point, if not in polygon
        near_poly_geom = gpd.GeoSeries((row['near_poly'])) #nearest polygon geometry
    left, bottom, right, top = bbox.bounds.iloc[0] #extracting and saving plot extent values based on min/max x/y
    img_crop, out_transform = mask(sat_img, bbox, crop=True) #cropping sat image to bounding box
    img_crop = np.transpose(img_crop, (1, 2, 0)) #transpose to swap order of numpy array for further operations (order of bands)
    
    #plotting results
    fig, ax = plt.subplots(figsize = (3, 3)) #creating figure for the row
    ax.imshow(img_crop, extent = [left, right, bottom, top]) #plot cropped image 
    if pd.notnull(row['in_poly']):
        in_poly_geom.plot(ax=ax, fc = "None", ec = "yellow") #plot poly
    else:
        near_poly_geom.plot(ax=ax, fc = "None", ec = "white") #plot nearest poly, if point not within one
    point_geom.plot(ax=ax, color='red', markersize = 15) #plot point
    fig = plt.gcf() #saving the current figure
    
    #storing in appropriate format
    figfile = io.BytesIO() #creating file-like binary object
    fig.savefig(figfile, format = 'png') #saving into the file-like object
    figfile.seek(0) #sets reference point to start of file
    string = base64.b64encode(figfile.read())
    
    #HTML tags
    uri = 'data:image/png;base64,' + urllib.parse.quote(string)
    html = '<img src = "%s"/>' % uri
    
    #clearing matplotlib cache
    plt.clf()
    
    #append image html strings to array
    images.append(html)

#adding image html strings to the dataframe in new column
highcert_df["image"] = np.array(images) 

#dropping columns that are no longer needed
final_df = highcert_df.drop(['geometry', 'in_poly', 'near_poly'], axis=1)

#getting the html format of the dataframe
html_df = final_df.to_html(escape=False)

#writing to file (uncomment when needed and change name as required)
#html_file = open("top100.html", "w")
#html_file.write(html_df)
#html_file.close()


    
















