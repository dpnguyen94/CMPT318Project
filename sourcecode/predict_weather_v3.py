import glob
import os

import numpy as np
import pandas as pd

from scipy import misc

from sklearn.preprocessing import MultiLabelBinarizer, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

katkam_path = 'katkam-rescaled'
weather_file = 'cleaned_data_v2.csv'

##Read weather observation data##

weather = pd.read_csv(weather_file);
weather['datetime'] = pd.to_datetime(weather['datetime'])

#print(weather)

##Read kitkam weather images##

katkam_files = glob.glob(katkam_path + '/*.jpg')
katkam = []
datetime = []
for file in katkam_files:
    img = misc.imread(file, 'F')
    img = img.reshape(-1)
    
    file_name = os.path.basename(file)
    datetime.append(file_name[7:21])
    katkam.append(img)
    
katkam = pd.DataFrame(katkam)
katkam['datetime'] = pd.to_datetime(datetime)

#print(katkam)

##Join two dataframes by datetime##

weather = weather.merge(katkam, on='datetime')
#print(weather)

##Train data to predict weather##

#mlb = MultiLabelBinarizer()
#y = mlb.fit_transform(weather['observed'].values)
y = weather['clear']

#y = mlb.inverse_transform(y)
#print(y)

X = weather.drop(['cloudy', 'datetime', 'rain', 'fog', 'snow'], 1) #re-use original DataFrame to save execution time

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

model = make_pipeline(
    SVC(kernel='linear', C=2.0)
    #I have tried other classifiers to test accuracy score, here is the results:
        #DecisionTree and RandomForest give significantly lower score than kNN
        #SVC(kernel='linear') takes way too long to produce results
        #kNN(7) gives better score than kNN(5) and takes less time to execute than kNN(9) or kNN(11) (with same accuracy score)
    #Other than that, I'm completely stuck. Can you think of any other way? Maybe trying to manipulate the image somehow?       
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))



