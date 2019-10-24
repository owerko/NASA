import matplotlib.pyplot as plt
import pandas as pd

fig_11 = pd.read_csv('ibis_r11.csv', sep=';', parse_dates=False, index_col='Time')

print(fig_11.info())
print()
print(fig_11.head())

fig, ax = plt.subplots()
ax.plot(fig_11.index[:200:1], fig_11.R14[:200:1], linestyle='solid', color='r')
ax.plot(fig_11.index[:200:1], fig_11.R9[:200:1], linestyle='solid', color='b')
ax.plot(fig_11.index[:200:1], fig_11.R19[:200:1], linestyle='solid', color='y')
ax.grid()
ax.set_xlabel('Data(yy/mm/dd)')
ax.set_ylabel('Displacement [mm]')
ax.set_title('Raw observations of displacements for selected points recorded by radar')
ax.tick_params('y', colors='r')
ax.xaxis.set_tick_params(rotation=90)

plt.show()