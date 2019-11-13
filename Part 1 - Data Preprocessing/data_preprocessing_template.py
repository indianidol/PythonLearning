# dat apreprocessing

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset

dataset = pd.read_csv('Data.csv')

X = pd.DataFrame(dataset.iloc[:, :-1].values)
#Y = pd.DataFrame(dataset.iloc[:, 3].values)
#x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values


# Fill Missing Values
from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0,copy="True")
missingvalues = missingvalues.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3]=missingvalues.transform(X.iloc[:, 1:3])

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_X = LabelEncoder()
#x[:,0]= labelencoder_X.fit_transform(x[:,0])
#X.iloc[:,0]= labelencoder_X.fit_transform(X.iloc[:,0])


ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
#x = np.array(ct.fit_transform(x), dtype=np.float)
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)


