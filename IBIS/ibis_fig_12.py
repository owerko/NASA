import matplotlib.pyplot as plt
import pandas as pd

fig_12 = pd.read_csv('ibis_r12.csv', sep=';', parse_dates=False, index_col='Time_t')

print(fig_12.info())
print()
print(fig_12.head())

# Próba na jednym pliku ale nie idzie jak trzeba

# fig, ax = plt.subplots()
# ax.plot(fig_12.index[:200:1], fig_12.theoretical[:200:1], linestyle='solid', color='r')
# ax.plot(fig_12.Time_o[:200:1], fig_12.observed[:200:1], linestyle='solid', color='b')
#
# ax.grid()
# ax.set_xlabel('Data(yy/mm/dd)')
# ax.set_ylabel('Displacement [mm]')
# ax.tick_params('y', colors='r')
# ax.xaxis.set_tick_params(rotation=90)
#
# plt.show()