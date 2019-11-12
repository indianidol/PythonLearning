# dat apreprocessing

# Importing libs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# import the dataset

dataset =pd.read_csv('Data.csv')

X = pd.DataFrame(dataset.iloc[:, :-1].values)
y = dataset.iloc[:, 3].values
Y = pd.DataFrame(dataset.iloc[:, 3].values)


from sklearn.impute import SimpleImputer
missingvalues = SimpleImputer(missing_values = np.nan, strategy = 'mean', verbose = 0,copy="True")
missingvalues = missingvalues.fit(X.iloc[:, 1:3])
X.iloc[:, 1:3]=missingvalues.transform(X.iloc[:, 1:3])
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,0]= labelencoder_X.fit_transform(X[:,0])
onehotencoder= OneHotEncoder(ColumnTransformer= [0])
X=onehotencoder.fit_transform(X).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)








# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer([('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X), dtype=np.float)
# Encoding Y data
from sklearn.preprocessing import LabelEncoder
y = LabelEncoder().fit_transform(y)
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train.reshape(-1,1))