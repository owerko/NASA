import matplotlib.pyplot as plt
import pandas as pd

# IMPORTANT TO READ
# prepare csv file with weather data in that schema:
# Time;Atm_pressure;Temperature;Humidity
# 2017-05-09 22:00;1015.27001953;1.45000005;94.02999878

def calc_artificial_displacement():
    name_weather_data_file = (input('Write the name weather data file with extension .csv: '))
    distance = float(input('Write the distance between IBIS and observed point [m]: '))
    weather_data = pd.read_csv(name_weather_data_file, sep=';')
    list_a_d = []

    for i in range(0, len(weather_data.index)):
        t = weather_data.iloc[i][2]
        p = weather_data.iloc[i][1]
        h = weather_data.iloc[i][3]
        k0 = 273.15

        ee = 6.11*10**(7.5*t/(237.30+t))
        e = ee*h/100
        a_d = (7.76*(10**(-5))*p/(k0+t)*2*distance+3.73*(10**(-1))*e/((t+k0)**2)*2*distance)*1000
        list_a_d.append(a_d)

    return list_a_d

def calc_artificial_displacement_reduced(list):
    list_a_d_r = []
    for i in range(0, len(list)):
        a_d_r = list[i]-list[0]
        list_a_d_r.append(a_d_r)
    return list_a_d_r

def add_calc_artificial_displacement_reduced_to_dataframe(list):
    name_weather_data_file = (input('Write the name weather data file with extension .csv: '))
    weather_data = pd.read_csv(name_weather_data_file, sep=';', parse_dates=False, index_col='Time')
    series_a_d_r = pd.Series(list)
    weather_data['art_disp_reduced'] = series_a_d_r
    weather_data_plus_a_d_r = weather_data
    return weather_data_plus_a_d_r

def write_dataframe_to_csv(dt):
    name_for_file = (input('Write the name for file with extension .csv: '))
    dt.to_csv(name_for_file, sep=';')

if __name__ == '__main__':

    list_adr = (calc_artificial_displacement_reduced(calc_artificial_displacement()))
    df_adr = (add_calc_artificial_displacement_reduced_to_dataframe(list_adr))
    write_dataframe_to_csv(df_adr)

