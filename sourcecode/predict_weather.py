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

katkam_path = 'katkam-rescaled' #I have re-scaled the image to half the size -> it didnt lose any accuracy score and it runs faster
weather_path = 'yvr-weather'

##Read weather observation data##

weather_files = glob.glob(weather_path + '/*.csv')

weather = []
for file in weather_files:
    df = pd.read_csv(file, skiprows=16)
    weather.append(df)

weather = pd.concat(weather).reset_index()

#columns = ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather']
columns = ['Date/Time', 'Weather']
    
weather = weather[columns]
weather = weather[weather['Weather'].notnull()].reset_index()
weather['datetime'] = pd.to_datetime(weather['Date/Time'])

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

#Transform weather observation to list (to be able to use MultiLabelBinarizer)
def string_to_list(s):
    return s.split(',')

weather['observed'] = weather['Weather'].apply(string_to_list)
print(weather)

##Train data to predict weather##

mlb = MultiLabelBinarizer()
y = mlb.fit_transform(weather['observed'].values)
#y = weather['Weather']

#y = mlb.inverse_transform(y)
#print(y)

X = weather.drop(['index', 'Date/Time', 'datetime', 'Weather', 'observed'], 1) #re-use original DataFrame to save execution time

print(X.shape, y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y)
print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

model = make_pipeline(
    PCA(250),
    OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7))
    #I have tried other classifiers to test accuracy score, here is the results:
        #DecisionTree and RandomForest give significantly lower score than kNN
        #SVC(kernel='linear') takes way too long to produce results
        #kNN(7) gives better score than kNN(5) and takes less time to execute than kNN(9) or kNN(11) (with same accuracy score)
    #Other than that, I'm completely stuck. Can you think of any other way? Maybe trying to manipulate the image somehow?       
)

model.fit(X_train, y_train)
print(model.score(X_test, y_test))



















