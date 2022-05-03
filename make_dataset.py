# -*- coding: utf-8 -*-

""" Importing the libraries """
import pandas as pd

""" LOADING THE DATA """

# donations file
donations = pd.read_csv("../../data/raw/donations/donations.csv")
# essay file 
essay = pd.read_csv("../../data/raw/essays/essays.csv")
# projects 
projects = pd.read_csv("../../data/raw/projects/projects.csv")
# resources 
resources = pd.read_csv("../../data/raw/resources/resources.csv")
# outcome 
outcomes = pd.read_csv("../../data/raw/outcomes/outcomes.csv")

""" DATA EXTRACTION """

# relevant columns from project table
projects = projects[['projectid', 
                               'teacher_acctid', 
                               'schoolid', 
                               'school_metro',
                               'school_district', 
                               'school_county',
                               'school_charter',
                               'school_magnet', 
                               'school_year_round', 
                               'school_nlns',
                               'school_kipp', 
                               'school_charter_ready_promise', 
                               'teacher_prefix',
                               'teacher_teach_for_america', 
                               'teacher_ny_teaching_fellow',
                               'primary_focus_subject', 
                               'primary_focus_area',
                               'secondary_focus_subject', 
                               'secondary_focus_area', 
                               'resource_type',
                               'poverty_level', 
                               'grade_level', 
                               'fulfillment_labor_materials',
                               'total_price_excluding_optional_support',
                               'total_price_including_optional_support', 
                               'students_reached',
                               'eligible_double_your_impact_match', 
                               'eligible_almost_home_match',
                               'date_posted'
                               ]]

# relevant coumns from resources table
resources = resources[['resourceid', 
                       'projectid', 
                       'vendorid',
                       'vendor_name',
                       'project_resource_type', 
                       'item_name', 
                       'item_number', 
                       'item_unit_price',
                       'item_quantity'
                        ]]

# relevant columns from the outcome table
outcomes = outcomes[[
                    'projectid', 
                    'is_exciting' 
                    ]]

# joining the tables
table = projects.merge(resources, left_on="projectid", right_on="projectid", how="inner")
table = table.merge(outcomes, left_on="projectid", right_on="projectid", how="inner")

# saving the table dataframe
table.to_csv("../../data/interim/project_resources_outcome.csv", index=False)
