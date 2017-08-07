import glob
import os
import time

import numpy as np
import pandas as pd

from scipy import misc

from sklearn.preprocessing import MultiLabelBinarizer#, Imputer, MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
#from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.svm import SVC

'''
from skmultilearn.adapt.brknn import BRkNNaClassifier, BRkNNbClassifier, BinaryRelevanceKNN
from skmultilearn.adapt.mlknn import MLkNN

from skmultilearn.problem_transform.br import BinaryRelevance
from skmultilearn.problem_transform.cc import ClassifierChain
from skmultilearn.problem_transform.lp import LabelPowerset
'''

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

katkam_path = 'katkam-rescaled' #I have re-scaled the image to half the size -> it didnt lose any accuracy score and it runs faster
weather_path = 'yvr-weather'
start_time = time.time()

### PART I ###


## Read weather observation data ##

weather_files = glob.glob(weather_path + '/*.csv')

weather = []
for file in weather_files:
    df = pd.read_csv(file, skiprows=16)
    weather.append(df)

weather = pd.concat(weather).reset_index()

#columns = ['Date/Time', 'Temp (°C)', 'Dew Point Temp (°C)', 'Rel Hum (%)', 'Wind Dir (10s deg)', 'Wind Spd (km/h)', 'Visibility (km)', 'Stn Press (kPa)', 'Weather']
columns = ['Date/Time', 'Weather'] # we are only concerned about date/time of observation and weather descrpiption for now    
weather = weather[columns]
weather = weather[weather['Weather'].notnull()].reset_index() # filter out missing weather observation values
weather['datetime'] = pd.to_datetime(weather['Date/Time']) # convert date/time string to datetime64 object (for merging later on)
#weather = weather[(weather['datetime'].dt.hour >= 8) & (weather['datetime'].dt.hour <= 18)]
#print(weather)

## Read kitkam weather images ##

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



### PART II ###



## Merge two dataframes by datetime ##

# merge weather images data to weather observation DataFrame
weather = weather.merge(katkam, on='datetime') 
#print(weather)

## Transform weather observation to list (to be able to use MultiLabelBinarizer) ##
def string_to_list(w):
    return w.split(',')

def filter_weather(w):
    new_weather = ['Clear', 'Cloudy', 'Drizzle', 'Fog', 'Rain', 'Snow', 'Thunderstorms']
    for weather in new_weather:
        if weather in w:
            return weather
    return ''

def filter_list(wlist):
    new_list = []
    for w in wlist:
        new_list.append(filter_weather(w))
    return new_list

def list_to_string(wlist):
    s = ''
    for w in wlist:
        s = s + str(w) + ','
    return s

weather['Weather_list'] = weather['Weather'].apply(string_to_list)
weather['filtered'] = weather['Weather_list'].apply(filter_list)
weather['filtered_Weather'] = weather['filtered'].apply(list_to_string)
print(weather[['datetime', 'Weather_list', 'filtered']])




### PART III ###




## Train data to predict weather ##

mlb = MultiLabelBinarizer() #this is a MultiLabelClassification problem
y = mlb.fit_transform(weather['filtered'].values)
wlist = pd.DataFrame(mlb.classes_)
print('List of weather conditions:')
print(wlist)

#y = mlb.inverse_transform(y)
#print(y)

#y = weather['Weather_filtered']

# re-use original DataFrame to reduce execution time
X = weather.drop(['index', 'Date/Time', 'datetime', 
                  'Weather', 'Weather_list', 'filtered', 
                  'filtered_Weather'], 1)
 
      
print('Input X shape (n_samples,n_features): ', X.shape)
print('Output y shape (n_samples,n_labels): ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X,y)
print('Training data shape:')
print('X_train: ',X_train.shape, ' ', 'X_test: ',X_test.shape)
print('y_train: ',y_train.shape, ' ', 'y_test: ',y_test.shape)

model = make_pipeline(
    PCA(250),
    OneVsRestClassifier(KNeighborsClassifier(n_neighbors=9))
    #OneVsRestClassifier(DecisionTreeClassifier())
    #OneVsRestClassifier(RandomForestClassifier(n_estimators=10))
    
    # I have tried other classifiers to test accuracy score, here is the results:
        # DecisionTree and RandomForest give significantly lower score than kNN
        # SVC(kernel='linear') takes way too long to produce results(?)
        # kNN(9) gives better score than kNN(5) and kNN(7) and takes less time to execute than kNN(11) (with same accuracy score)
)

model.fit(X_train, y_train)
print('Accuracy score: %.2f' % model.score(X_test, y_test))
end_time = time.time()

print('Total runtime: %.1fs' % (end_time - start_time))

















