# MSAAI510
USD MS AAI - 510 - Team 2

# MS AAI - 510 - MACHINE LEARNING FUNDAMENTALS
# Final Project - Diabetes Predictor based on CDC Health Indicators
## Team 2 - Santosh Kumar, Michael Domingo, Israel Romero Olvera
This notebook contains our analysis and model for the selected dataset.

### Installation

To run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/isralennon/MSAAI510.git```
2. Run with Jypiter: https://jupyter.org/install 

## Jupyter Notebook File Location
Our Jupyter Notebook file can be found in this root folder under file name AAI_510_FinalProject_Team2.ipynb.
Previous versions of the notebook files and other materials are stored in the Archive subfolder, but they are only for reference and not needed for the final project to work.

## Project Intro/Objective
This project aims to develop a classification model to correctly identify if a patient has diabetes/pre-diabetes or not based on certain CDC health indicators.

### Technologies
- Python
- Jupyter Notebook

#### Dataset
- **Source**: https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicatorsA
- **Size**: The dataset contains 18 files with 45 minutes of activities captured with 2 accelerometers at 50Hz, for a total of over 2.5 million records.
- Python code for Package installation:
pip install ucimlrepo
- Python code for importing dataset into your code

from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
cdc_diabetes_health_indicators = fetch_ucirepo(id=891) 
  
# data (as pandas dataframes) 
X = cdc_diabetes_health_indicators.data.features 
y = cdc_diabetes_health_indicators.data.targets 
  
# metadata 
print(cdc_diabetes_health_indicators.metadata) 
  
# variable information 
print(cdc_diabetes_health_indicators.variables) 


#### Models Used
- TBD
- TBD

#### License
This project is licensed under the GNU License - see the [LICENSE.txt](LICENSE.txt) file for details.


