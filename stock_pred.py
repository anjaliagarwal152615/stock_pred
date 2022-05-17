from pyexpat import model
from re import X
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as LR
from sklearn.svm import LinearSVR
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.metrics import precision_score as ps
from sklearn.metrics import recall_score as rs
from sklearn.metrics import f1_score as f1s
from sklearn.metrics import accuracy_score as acc
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from scipy import linalg
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib as mpl
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.metrics import mean_squared_error as mse
from sklearn.model_selection import cross_val_score as cvs
from sklearn.metrics import r2_score as r2 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report, precision_score

data_xls = pd.read_excel(r'C:\Users\Anjali\Desktop\stock_pred\sample_data.xls', sheet_name='Sheet1', header=0)
data_xls.to_csv('data_csv', encoding= 'utf-8')
data_csv = pd.read_csv('data_csv', header=0)
data_csv.head()




