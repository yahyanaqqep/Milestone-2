import pickle as pk
import ensemble as ens
import numpy as np
import pandas as pd
import sklearn.preprocessing as spre
import matplotlib.pyplot as plt

Dataset_File = pd.read_csv("ElecDeviceRatingPrediction_Milestone2.csv")

Y = Dataset_File['rating']
X = Dataset_File

brandLE = spre.LabelEncoder()
X['brand'] = brandLE.fit_transform(X['brand'])
# fileHandler = open("brand_Label_Encoder.pk","wb")
# pk.dump(brandLE,fileHandler)
processor_brandLE = spre.LabelEncoder()
X['processor_brand'] = processor_brandLE.fit_transform(X['processor_brand'])
processor_nameLE = spre.LabelEncoder()
X['processor_name'] = processor_nameLE.fit_transform(X['processor_name'])
ram_typeLE = spre.LabelEncoder()
X['ram_type'] = ram_typeLE.fit_transform(X['ram_type'])
osLE = spre.LabelEncoder()
X['os'] = osLE.fit_transform(X['os'])
weightLE = spre.LabelEncoder()
X['weight'] = weightLE.fit_transform(X['weight'])
TouchscreenLE = spre.LabelEncoder()
X['Touchscreen'] = TouchscreenLE.fit_transform(X['Touchscreen'])
msofficeLE = spre.LabelEncoder()
X['msoffice'] = msofficeLE.fit_transform(X['msoffice'])
ratingLE = spre.LabelEncoder()
X['rating'] = ratingLE.fit_transform(X['rating'])


X['processor_gnrtn'] = [0 if x=='Not Available' else int(x.replace('th','')) for x in X['processor_gnrtn']]

X['ram_gb'] = [int(x.replace('GB','')) for x in X['ram_gb']]

X['ssd'] = [int(x.replace('GB','')) for x in X['ssd']]

X['hdd'] = [int(x.replace('GB','')) for x in X['hdd']]

X['graphic_card_gb'] = [int(x.replace('GB','')) for x in X['graphic_card_gb']]

X['warranty'] = [0 if x=='No warranty' else int(x.replace('year','').replace('s','')) for x in X['warranty']]
corr = X.corr()
corr = corr[:]['rating']
corr = corr.abs()
print(corr)
plt.barh(corr.sort_values().index, corr.sort_values().values)
plt.show()
