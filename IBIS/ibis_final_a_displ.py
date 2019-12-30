import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rcParams.update(plt.rcParamsDefault)

# IMPORTANT !!!!!!!!!!!!!!!

# in file Json add on start data: {"full_h": and close dictionary on end put additional}

base_temperatures = pd.read_json('json/a_displ_ibis.json')
base_temperatures.sort_index(inplace=True)
temperatures = base_temperatures.copy()
y = base_temperatures['full_h']
print(y)

fig, ax = plt.subplots()

ax = y.plot(figsize=(20, 12))
y.plot(figsize=(15, 6))
# plt.xlabel('')
# plt.ylabel('')
ax.set_xlabel('Date', fontsize=14)  # xlabel
ax.set_ylabel('Artificial Displacement [mm]', fontsize=14)  # ylabel

plt.title('Artificial displacement data series calculated from weather data obtained during IBIS measurement', fontsize=16)
plt.tight_layout()

fig.savefig('fig1_d.eps', format='eps', dpi=600)

plt.show()

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)

# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))

print(pdq[:4])

# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 24) for x in list(itertools.product(p, d, q))]

print(seasonal_pdq[:4])

print('Examples of parameter combinations for Seasonal ARIMA...')
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[1]))
print('SARIMAX: {} x {}'.format(pdq[1], seasonal_pdq[2]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[3]))
print('SARIMAX: {} x {}'.format(pdq[2], seasonal_pdq[4]))

# warnings.filterwarnings("ignore") # specify to ignore warning messages
#
# list_aci = []
# list_print_param_aci = []
#
# for param in pdq:
#     for param_seasonal in seasonal_pdq:
#         try:
#             mod = sm.tsa.statespace.SARIMAX(y,
#                                             order=param,
#                                             seasonal_order=param_seasonal,
#                                             enforce_stationarity=False,
#                                             enforce_invertibility=False)
#
#             results = mod.fit()
#             list_aci.append(results.aic)
#             print(f'SARIMAX{param}x{param_seasonal}24 - AIC:{results.aic}')
#             a = (f'SARIMAX{param}x{param_seasonal}24 - AIC:{results.aic}')
#             list_print_param_aci.append(a)
#         except:
#             continue
#
# print(list_aci)
# print(len(list_aci))
# print(min(list_aci))
# print('###########')
# series_list_print_param_aci = pd.Series(list_print_param_aci)
# print(series_list_print_param_aci)
# series_list_print_param_aci.to_csv('ibis_fin_param_aci_a_displ_co_h_u.csv', sep=';')

# IMPORTANT TO READ
# FULL INFO ABOUT PARAMS and ACI IN CSV FILE


# Najmniejsze AIC
# SARIMAX(1, 0, 1)x(0, 1, 1, 24)24 - AIC:824.9345305328652

mod = sm.tsa.statespace.SARIMAX(y,
                                order=(1, 0, 1),
                                seasonal_order=(0, 1, 1, 24),
                                enforce_stationarity=False,
                                enforce_invertibility=False)

results = mod.fit()

print(results.summary().tables[1])

fig, ax = plt.subplots()
# results.plot_diagnostics(figsize=(15, 12))
ax = results.plot_diagnostics(figsize=(15, 12))
ax.savefig('fig2_d.eps', format='eps', dpi=600)
plt.show()

pred = results.get_prediction(start=457, dynamic=False)
pred_ci = pred.conf_int()

fig, ax = plt.subplots()

ax = y[:].plot(label='observed', figsize=(20, 12))
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set(xlabel='Date', ylabel='Artificial Displacement [mm]', title='SARIMAX Static')
# ax.set_axis_bgcolor("white")
# ax.get_ticklines()
ax.grid(True)
plt.legend()

fig.savefig('fig3_d.eps', format='eps', dpi=600)
plt.show()

y_forecasted = pred.predicted_mean
y_truth = y[:]

# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('The Mean Squared Error of our forecasts is {}'.format(round(mse, 2)))

pred_dynamic = results.get_prediction(start=457, dynamic=True, full_results=True)
pred_dynamic_ci = pred_dynamic.conf_int()
print(pred_dynamic.predicted_mean)
print(pred_dynamic.conf_int())

fig, ax = plt.subplots()

ax = y[:].plot(label='Observed', figsize=(20, 15))
pred_dynamic.predicted_mean.plot(label='Dynamic Forecast', ax=ax)

ax.fill_between(pred_dynamic_ci.index,
                pred_dynamic_ci.iloc[:, 0],
                pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)

# ax.fill_betweenx(ax.get_ylim(), 0, y.index[-1],
#                 alpha=.1, zorder=-1)


ax.set(xlabel='Date', ylabel='Artificial Displacement [mm]', title='SARIMAX Dynamic')
# ax.set_axis_bgcolor("white")
# ax.get_ticklines()
ax.grid(True)



plt.legend()
fig.tight_layout()


fig.savefig('fig4_d.eps', format='eps', dpi=600)

plt.show()





pom = pd.DataFrame(pred_dynamic.predicted_mean)
pom.to_csv('pom.csv', sep=';')


szare = pd.DataFrame(pred_dynamic.conf_int())
szare.to_csv('szare.csv', sep=';')