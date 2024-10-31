import geopandas as gpd
import pandas as pd
from shapely import within, centroid, intersects, intersection
import os
## find the directory of the python (assures compatibility)
python_directory = os.path.abspath("")
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import copy
import math
import numpy as np
import warnings
warnings.filterwarnings("ignore")

### plotting
## define colours
colours = {
    "Blatchington Mill School": "steelblue",
    "Brighton Aldridge Community Academy": "orange",
    "Cardinal Newman Catholic School": "limegreen",
    "Dorothy Stringer School": "firebrick",
    "Hove Park School and Sixth Form Centre": "mediumpurple",
    "King's School": "sienna",
    "Longhill High School": "palevioletred",
    "Patcham High School": "gray",
    "Portslade Aldridge Community Academy": "darkkhaki",
    "Varndean School": "darkturquoise",
}

## create custom legend
def create_custom_legend_handles(colours=colours):
    handles, labels = plt.gca().get_legend_handles_labels()
    extend_list = []
    for school_str in colours:
        point = Line2D([0], [0], label=school_str, marker='o', markersize=10, markeredgecolor="black", markerfacecolor=colours[school_str], linestyle="")
        extend_list.append(point)
    handles.extend(extend_list)
    return(handles)

### reseting parameters and identifying 5 year olds (required when starting the models)
## Identify the catchment ID of a polygon geometry based based on the catchment that contains it
# Used when resetting the parameters
def polygon_catchment_ID(geometry_df, geometry_index, catchment):
    geometry = geometry_df.at[geometry_index, "geometry"]
    portion_list = list()
    ## identify the portion of the polygon which falls within the catchments
    for i in catchment.index:
        portion_list.append(portion_within(geometry, catchment.at[i,"geometry"]))
    ## report the catchment Id which includes the highest portion of the polygon
    i = portion_list.index(max(portion_list))
    return catchment.at[i, "catchment_ID"]

## Identify the catchment ID of a point geometry based based on the catchment that contains it
# Used when resetting the parameters
def point_catchment_ID(geometry_df, geometry_index, catchment):
    geometry = geometry_df.at[geometry_index, "geometry"]
    for i in catchment.index:
        # if the centroid of the geometry is within the catchment geometry
        if within(centroid(geometry), catchment.at[i,"geometry"]): return catchment.at[i, "catchment_ID"]

def portion_within(geometry_a, geometry_b):
    intersection_area = intersection(geometry_a, geometry_b).area
    portion = intersection_area / geometry_a.area
    return portion

## generate additional attribute columns
def reset_parameters(catchment, schools, students):
    ## Catchments
    # column for the catchment id
    catchment["catchment_ID"] = [index + 1 for index in catchment.index]
    ## Schools 
    # Column to assign the total number of students in each school point
    # Column to assign colours
    # Column to assign catchment ID
    schools["students_total"] = [0 for index in schools.index]
    schools["catchment_ID"] = [0 for index in schools.index]
    schools["catchment_ID"] = [point_catchment_ID(schools, i, catchment) for i in schools.index]
    schools["colour"] = ["" for index in schools.index]
    ## LSOA including number of students
    # Column to assign the name of the schools
    # Column to assign the estimated number of 5 year olds
    # Column to assign catchment ID
    # Column to assign colours
    students["school"] = ["" for index in students.index]
    students["5_est"] = [math.floor(n * 0.19288) for n in students["5_9_total"]]
    students["catchment_ID"] = [0 for index in students.index]
    students["catchment_ID"] = [polygon_catchment_ID(students, i, catchment) for i in students.index]
    students["catchment_ID_school"] = [0 for index in students.index]
    students["dist_to_school"] = [0 for index in students.index]
    students["colour"] = ["" for index in students.index]

    return {
        "schools": schools, 
        "students": students
    }


### Model version 1.1: optimise for schools ignoring catchments
def Optimise_PANs_Schools(
        schools,
        students_lsoa,
        PANs,
        PAN_year=2024,
        initial_school="Dorothy Stringer School",
        ):
    """
    A function that identifies the catchement areas based on the proposed PANs. 
    Loops through schools and assigns LSOAs. 
    This is not constrained to any predefined catchments.

    Parameters
    ----------
    `schools`: GeoPandas DataFrame
        School locations as points
    `students_losa`: Geopandas DataFrame
        LSOAs including an attribute for the number of students "est_5"
    `PANs`: Pandas DataFrame
        Planned PANs for each school
    `PAN_year`: int (default=2024)
        PAN year to extract the values from the `PANs` DataFrame
    `initial_school`: str (default="Dorothy Stringer School")
        The initial school in the optimisation loop

    Returns
    -------
    Dictionary including:
        - "schools": GeoPandas DataFrame,
        - "students": GeoPandas DataFrame
    """
    ## closest school variables
    schools_temp = copy.deepcopy(schools)
    lsoa_temp = copy.deepcopy(students_lsoa)
    previous_str = None

    ## Find order of schools by dist starting from initial school
    current_str = initial_school
    n_next_schools = len(schools_temp.index)
    schools_ordered = []
    while n_next_schools > 0:
        # update the ordered list
        schools_ordered.append(current_str)
        # find distances
        current_school = schools[schools["establishment_name"] == current_str]
        i_cSchool = current_school.index[0]
        schools_temp = schools_temp.drop(i_cSchool)
        ## if there is another school
        if len(schools_temp.index) > 0: 
            dists = schools.at[current_school.index[0],"geometry"].distance(schools_temp["geometry"])
            # find the next closest df (school)
            i_nSchool = dists[dists == min(dists)].index[0]
            next_school = schools.loc[[i_nSchool]]
            next_school_str = next_school.at[i_nSchool, "establishment_name"]
            # update current str 
            current_str = next_school_str
        n_next_schools = len(schools_temp.index)
        

    ## Extract the PANs for the input year
    target_PAN = {}
    saturated_PAN = {}
    for school_str in PANs["school"]:
        target_PAN[school_str] = int(PANs[PANs["school"] == school_str][f"pan{PAN_year}"])
        saturated_PAN[school_str] = False
    ## while any LSOA has not been adressed a school
    while len(students_lsoa[students_lsoa["school"] == ""].index) > 0:
        # Loop through the ordered schools
        for current_str in schools_ordered:
            ## initial school
            current_school = schools[schools["establishment_name"] == current_str]
            i_cSchool = current_school.index[0]

            ## Accumilate students from the closest LSOAs
            if saturated_PAN[current_str] == False:
                # find LSOA without an assigned establishment_name
                lsoa_temp = students_lsoa[students_lsoa["school"] == ""]
                dists_school_LSOAs = current_school.at[i_cSchool, "geometry"].distance(lsoa_temp["geometry"])
                i_lsoa = dists_school_LSOAs[dists_school_LSOAs == min(dists_school_LSOAs)].index[0]
                ## if adding the number of students in the LSOA will not lead to exceeding the PAN
                if schools.at[i_cSchool, "students_total"] + students_lsoa.at[i_lsoa, "5_est"] < target_PAN[current_str]:
                    schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    students_lsoa.at[i_lsoa, "school"] = current_str
                ## if this is the last school with any availability. Add the students to it
                elif schools.at[i_cSchool, "students_total"] + students_lsoa.at[i_lsoa, "5_est"] >= target_PAN[current_str] and list(saturated_PAN.values()).count(False) == 1:
                    schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    students_lsoa.at[i_lsoa, "school"] = current_str
                    saturated_PAN[current_str] = True
                else:
                    saturated_PAN[current_str] = True
            
            ## Accumilate to the student from the closest LSOA regardless of PAN, if all schools reached their PANs
            if list(saturated_PAN.values()).count(False) == 0:
                # find LSOA without an assigned establishment_name
                lsoa_temp = students_lsoa[students_lsoa["school"] == ""]
                dists_school_LSOAs = current_school.at[i_cSchool, "geometry"].distance(lsoa_temp["geometry"])
                i_lsoa = dists_school_LSOAs[dists_school_LSOAs == min(dists_school_LSOAs)].index[0]
                schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                students_lsoa.at[i_lsoa, "school"] = current_str

            ## if the school is saturated, skip it
            if saturated_PAN[current_str] == True:
                continue
                
        
    return {
        "schools": schools,
        "students": students_lsoa,
    }

### Model 1.2: optimise for LSOAs ignoring catchments
def Optimise_PANs_LSOAs(
        schools,
        students_lsoa,
        PANs,
        PAN_year=2024,
        ):
    """
    A function that identifies the catchement areas based on the proposed PANs. 
    Loops through LSOAs and assigns schools. 
    This is not constrained to any predefined catchments.

    Parameters
    ----------
    `schools`: GeoPandas DataFrame
        School locations as points
    `students_losa`: Geopandas DataFrame
        LSOAs including an attribute for the number of students "est_5"
    `PANs`: Pandas DataFrame
        Planned PANs for each school
    `PAN_year`: int (default=2024)
        PAN year to extract the values from the `PANs` DataFrame
    `initial_school`: str (default="Dorothy Stringer School")
        The initial school in the optimisation loop

    Returns
    -------
    Dictionary including:
        - "schools": GeoPandas DataFrame,
        - "students": GeoPandas DataFrame
    """
    ## closest school variables
    schools_temp = copy.deepcopy(schools)
    lsoa_temp = copy.deepcopy(students_lsoa)
    ## Extract the PANs for the input year
    target_PAN = {}
    saturated_PAN = {}
    for school_str in PANs["school"]:
        target_PAN[school_str] = int(PANs[PANs["school"] == school_str][f"pan{PAN_year}"])
        saturated_PAN[school_str] = False
    # Loop through the LSOAs
    for i_lsoa in students_lsoa.index:
        ## keep looping unti the LSOA is assing a school
        while students_lsoa.at[i_lsoa, "school"] == "":
            ## current LSOA under analysis
            current_lsoa = students_lsoa.iloc[[i_lsoa]]
            ## Distance from this LSOA to all schools
            dists_lsoa_schools = current_lsoa.at[i_lsoa, "geometry"].distance(schools["geometry"])
            dists_set = sorted(set(dists_lsoa_schools))
            ## Index of the closest school in the schools DataFrame
            test_PAN_status = True
            i_temp = 0
            current_str = ""
            while test_PAN_status == True and i_temp < len(saturated_PAN.values()):
                i_cSchool = dists_lsoa_schools[dists_lsoa_schools == dists_set[i_temp]].index[0]
                current_str = schools.at[i_cSchool, "establishment_name"]
                if saturated_PAN[current_str] == True:
                    i_temp += 1
                    continue
                else:
                    test_PAN_status = False
            ## Current school under analysis
            current_school = schools.iloc[[i_cSchool]]
            # current_str = current_school.at[i_cSchool, "establishment_name"]
            
            if saturated_PAN[current_str] == False:
                ## if adding the number of students in the LSOA will not lead to exceeding the PAN
                if schools.at[i_cSchool, "students_total"] + students_lsoa.at[i_lsoa, "5_est"] <= target_PAN[current_str]:
                    current_school.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    # schools.at[i_cSchool, "students_total"] += students.at[i_lsoa, "5_est"]
                    students_lsoa.at[i_lsoa, "school"] = current_str
                else:
                    saturated_PAN[current_str] = True
        
    return {
        "schools": schools,
        "students": students_lsoa,
    }

### Model version 2.1: optimise for schools while consideting catchments
def Optimise_PANsCatchment_Schools(
        schools,
        students_lsoa,
        PANs,
        PAN_year=2024,
        initial_school="Dorothy Stringer School",
        ):
    """
    A function that identifies the catchement areas based on the proposed PANs. 
    Loops through schools and assigns LSOAs. 
    This is constrained to a predefined catchment.
    The `schools` and `students_lsoa` must include a parameter labelled as "catchment_ID".

    Parameters
    ----------
    `schools`: GeoPandas DataFrame
        School locations as points
    `students_losa`: GeoPandas DataFrame
        LSOAs including an attribute for the number of students "est_5"
    `PANs`: Pandas DataFrame
        Planned PANs for each school
    `PAN_year`: int (default=2024)
        PAN year to extract the values from the `PANs` DataFrame
    `initial_school`: str (default="Dorothy Stringer School")
        The initial school in the optimisation loop

    Returns
    -------
    Dictionary including:
        - "schools": GeoPandas DataFrame,
        - "students": GeoPandas DataFrame
    """
    ## closest school variables
    schools_temp = copy.deepcopy(schools)
    lsoa_temp = copy.deepcopy(students_lsoa)
    previous_str = None

    ## Find order of schools by dist starting from initial school
    current_str = initial_school
    n_next_schools = len(schools_temp.index)
    schools_ordered = []
    schools_IDs = {}
    while n_next_schools > 0:
        # update the ordered list
        schools_ordered.append(current_str)
        # find distances
        current_school = schools[schools["establishment_name"] == current_str]
        i_cSchool = current_school.index[0]
        schools_temp = schools_temp.drop(i_cSchool)
        # update the IDs
        schools_IDs[current_str] = (schools.at[i_cSchool, "catchment_ID"])
        ## if there is another school
        if len(schools_temp.index) > 0: 
            dists = schools.at[current_school.index[0],"geometry"].distance(schools_temp["geometry"])
            # find the next closest df (school)
            i_nSchool = dists[dists == min(dists)].index[0]
            next_school = schools.loc[[i_nSchool]]
            next_school_str = next_school.at[i_nSchool, "establishment_name"]
            # update current str 
            current_str = next_school_str
        n_next_schools = len(schools_temp.index)
        

    ## Extract the PANs for the input year
    target_PAN = {}
    saturated_PAN = {}
    for school_str in PANs["school"]:
        target_PAN[school_str] = int(PANs[PANs["school"] == school_str][f"pan{PAN_year}"])
        saturated_PAN[school_str] = False
    ## while any LSOA has not been adressed a school
    while len(students_lsoa[students_lsoa["school"] == ""].index) > 0:
        # Loop through the ordered schools
        for current_str in schools_ordered:
            ## initial school
            current_school = schools[schools["establishment_name"] == current_str]
            i_cSchool = current_school.index[0]
            ID_cSchool = current_school.at[i_cSchool, "catchment_ID"]

            ## Accumilate students from the closest LSOAs
            if saturated_PAN[current_str] == False:
                # find LSOAs without an assignment establishment_name and within the schools catchment
                lsoa_in_catchment = students_lsoa[(students_lsoa["school"] == "") & (students_lsoa["catchment_ID"] == schools_IDs[current_str])]
                if len(lsoa_in_catchment) > 0:
                    lsoa_temp = lsoa_in_catchment
                # else, if no LSOAs are within the catchment, find any LSOAs without an assigned establishment_name
                else:
                    lsoa_temp = students_lsoa[students_lsoa["school"] == ""]

                ## find the distances between the selected LSOAs and the current school
                dists_school_LSOAs = current_school.at[i_cSchool, "geometry"].distance(lsoa_temp["geometry"])
                i_lsoa = dists_school_LSOAs[dists_school_LSOAs == min(dists_school_LSOAs)].index[0]
                ## if adding the number of students in the LSOA will not lead to exceeding the PAN
                if schools.at[i_cSchool, "students_total"] + students_lsoa.at[i_lsoa, "5_est"] < target_PAN[current_str]:
                    schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    students_lsoa.at[i_lsoa, "school"] = current_str
                    students_lsoa.at[i_lsoa, "catchment_ID_school"] = schools_IDs[current_str]
                    students_lsoa.at[i_lsoa, "dist_to_school"] = min(dists_school_LSOAs)
                ## if this is the last school with any availability. Add the students to it
                elif schools.at[i_cSchool, "students_total"] + students_lsoa.at[i_lsoa, "5_est"] >= target_PAN[current_str] and list(saturated_PAN.values()).count(False) == 1:
                    schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    students_lsoa.at[i_lsoa, "school"] = current_str
                    students_lsoa.at[i_lsoa, "catchment_ID_school"] = schools_IDs[current_str]
                    students_lsoa.at[i_lsoa, "dist_to_school"] = min(dists_school_LSOAs)
                    saturated_PAN[current_str] = True
                else:
                    saturated_PAN[current_str] = True
            
            ## Accumilate to the student from the closest LSOA regardless of PAN, if all schools reached their PANs
            if list(saturated_PAN.values()).count(False) == 0:
                # find LSOA without an assigned establishment_name
                lsoa_temp = students_lsoa[students_lsoa["school"] == ""]
                dists_school_LSOAs = current_school.at[i_cSchool, "geometry"].distance(lsoa_temp["geometry"])
                i_lsoa = dists_school_LSOAs[dists_school_LSOAs == min(dists_school_LSOAs)].index[0]
                schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                students_lsoa.at[i_lsoa, "school"] = current_str
                students_lsoa.at[i_lsoa, "catchment_ID_school"] = schools_IDs[current_str]
                students_lsoa.at[i_lsoa, "dist_to_school"] = min(dists_school_LSOAs)

            ## if the school is saturated, skip it
            if saturated_PAN[current_str] == True:
                continue
    
    ## generate a school
        
    return {
        "schools": schools,
        "students": students_lsoa,
    }


### Model 2.2: optimise for LSOAs while consideting catchments
## Work in progress, an error leading assigning more students that actually available (sum actual students is less than sum model assigned students)
def Optimise_PANsCatchment_LSOAs(
        schools,
        students_lsoa,
        PANs,
        PAN_year=2024,
        ):
    """
    A function that identifies the catchement areas based on the proposed PANs. 
    Loops through LSOAs and assigns schools. 
    This is constrained to a predefined catchment.

    Parameters
    ----------
    `schools`: GeoPandas DataFrame
        School locations as points
    `students_losa`: Geopandas DataFrame
        LSOAs including an attribute for the number of students "est_5"
    `PANs`: Pandas DataFrame
        Planned PANs for each school
    `PAN_year`: int (default=2024)
        PAN year to extract the values from the `PANs` DataFrame
    `initial_school`: str (default="Dorothy Stringer School")
        The initial school in the optimisation loop

    Returns
    -------
    Dictionary including:
        - "schools": GeoPandas DataFrame,
        - "students": GeoPandas DataFrame
    """
    ## Extract the PANs for the input year
    target_PAN = {}
    saturated_PAN = {}
    for school_str in PANs["school"]:
        target_PAN[school_str] = int(PANs[PANs["school"] == school_str][f"pan{PAN_year}"])
        saturated_PAN[school_str] = False
    # Loop through the LSOAs
    for i_lsoa in students_lsoa.index:
        ## keep looping unti the LSOA is assing a school
        while students_lsoa.at[i_lsoa, "school"] == "":
            ## current LSOA under analysis
            current_lsoa = students_lsoa.iloc[[i_lsoa]]
            ## current schools in catchment
            schools_in_catchment = schools[schools["catchment_ID"] == current_lsoa.at[i_lsoa, "catchment_ID"]]
            strs_in_catchment = schools_in_catchment["establishment_name"]
            saturated_in_catchment = [saturated_PAN[str_temp] for str_temp in strs_in_catchment]
            ## if all of the schools in the catchment are saturated with students, consider all the schools
            saturated_temp = []
            if saturated_in_catchment.count(False) == 0:
                schools_temp = schools
                saturated_temp = saturated_PAN.values()
            ## else, if any of the schools in catchment is not saturated with students, only consider the schools in the catchment
            else:
                schools_temp = schools_in_catchment
                saturated_temp = saturated_in_catchment
            
            print(saturated_temp)

            ## Distance from this LSOA to all schools
            dists_lsoa_schools = current_lsoa.at[i_lsoa, "geometry"].distance(schools_temp["geometry"])
            dists_set = sorted(set(dists_lsoa_schools))
            ## Index of the closest school in the schools DataFrame
            test_PAN_status = True
            i_temp = 0
            current_str = ""
            while test_PAN_status == True and i_temp < len(saturated_temp):
                i_cSchool = dists_lsoa_schools[dists_lsoa_schools == dists_set[i_temp]].index[0]
                current_str = schools_temp.at[i_cSchool, "establishment_name"]
                if saturated_PAN[current_str] == True:
                    i_temp += 1
                    continue
                else:
                    test_PAN_status = False
            ## Current school under analysis
            current_school = schools.iloc[[i_cSchool]]
            # current_str = current_school.at[i_cSchool, "establishment_name"]
            
            if saturated_PAN[current_str] == False:
                ## if adding the number of students in the LSOA will not lead to exceeding the PAN
                if schools_temp.at[i_cSchool, "students_total"] + students_lsoa.at[i_lsoa, "5_est"] <= target_PAN[current_str]:
                    current_school.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    schools_temp.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    schools.at[i_cSchool, "students_total"] += students_lsoa.at[i_lsoa, "5_est"]
                    # schools.at[i_cSchool, "students_total"] += students.at[i_lsoa, "5_est"]
                    students_lsoa.at[i_lsoa, "school"] = current_str
                else:
                    saturated_PAN[current_str] = True
            if list(saturated_PAN.values()).count(False) == 0: break
        
    return {
        "schools": schools,
        "students": students_lsoa,
    }