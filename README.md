# Helper Function Groups 

Data Process 
-- 
Within `data_process.py`, there are several helper functions for 
* transform special values 
* identify missing values and percentage of missing values for each 
feature 
* make missing value strategy, drop column, drop observation, 
or filling in values based on certain rules (fill-in strategy still in 
beta)
* Label Encoder for multiple columns at a time 

Outlier Detection 
-- 
Within `OutlierDetect.py`, 
* Identify if data is univariate and multivariate. This is an internal 
method to help other functions identify if appropriate data dimension is 
applied
* `mean_z_score`, `median_z_score` and `iqr_base_range` for detecting 
univariate outliers. `iqr_base_range` will return a list with two 
elements that identify a range of inliers. Data that is not in the range 
are considered outliers. `mean_z_score` and `median_z_score` will return 
a binary list where `data.loc[list==1, :]` are outliers. 
