
# Module 3 Final Project - Predicting Customer churn within a telecom company. 


## Introduction

Within this custover data we will examine metrics and build a model to predict customer turnover for within a customer dataset. 


## Approach

- Data will be imported and cleaned 
- Certain values will be visualized to ensure accurace and identify patterns. 
- 3 classification models built:
    - K Nearest Neighbors
    - GridSearch 
    -XGBoost 
    
- Models will be refitted and optimized 


## Files

student.ipynb380 kB4 minutes ago
Running
CONTRIBUTING.md1.81 kB14 days ago
data.csv310 kB7 months ago
LICENSE.md1.35 kB14 days ago
module_3_project_rubric.pdf78.9 kB14 days ago
README.md



## Requirements 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

%matplotlib inline 
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score

## Findings 

All models performed extremly well with scores above 90% 
XGBoost was one of the better improvements with tuning and while other models scored higher I belive with continues data turning XGBoost could be the best model. 





