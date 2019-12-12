import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import json

plt.style.use('default')
plt.rcParams.update(plt.rcParamsDefault)

base_temperatures = pd.read_csv('surowe_atmo_r9_co_h_sur_u_niest.csv', sep=';') #parse_dates=False, index_col='Time'
y = base_temperatures.loc[::1]['Temperature']
t = base_temperatures.loc[::1]['Time']
print(y)
print(len(y))
print(t)

my_list_y = []
my_list_t = []

for i in range(0, len(y)):
    my_list_y.append(y[i])
    my_list_t.append(t[i])

print(my_list_t)
print(my_list_y)

temp_dict = {}
for k, v in zip(my_list_t, my_list_y):
    temp_dict[k] = v


print(temp_dict)

def dict_to_json(dict, json_path):

    with open(json_path, 'w') as json_file:
        json.dump(dict, json_file)
    print('File {} saved successfully'.format(
        json_path))

# JSON
output_filename = 'temp_ibis.json'
json_folder = 'json/'

dict_to_json(temp_dict, json_folder + output_filename)