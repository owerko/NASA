import matplotlib.pyplot as plt
import pandas as pd

fig_9 = pd.read_csv('surowe_atmo_r9.csv', sep=';', parse_dates=False, index_col='Time')

print(fig_9.info())
print()
print(fig_9.head(3))

fig, ax = plt.subplots()
ax.plot(fig_9.index[:800:2], fig_9.Temperature[:800:2], linestyle='solid', color='r')
ax.grid()
ax.set_xlabel('Data(yy/mm/dd)')
ax.set_ylabel('Temperature [°C]')
ax.set_title('Changes temperature and pressure during the observation')
ax.tick_params('y', colors='r')
ax.xaxis.set_tick_params(rotation=90)
every_nth = 16
for n, label in enumerate(ax.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)

ax1 = ax.twinx()
ax1.plot(fig_9.index[:800:2], fig_9.Atm_pressure[:800:2], linestyle='--', color='b')
ax1.set_ylabel('Pressure [hPa]')
ax1.tick_params('y', colors='b')

fig, ax2 = plt.subplots()

ax2.plot(fig_9.index[:800:2], fig_9.Humidity[:800:2], linestyle='solid', color='g')
ax2.grid()
ax2.set_xlabel('Data(yy/mm/dd)')
ax2.set_ylabel('Relative Humidity [%]')
ax2.set_title('Changes relative humidity during the observation and artificial displacement values')
ax2.tick_params('y', colors='g')
ax2.xaxis.set_tick_params(rotation=90)
every_nth = 16
for n, label in enumerate(ax2.xaxis.get_ticklabels()):
    if n % every_nth != 0:
        label.set_visible(False)

ax3 = ax2.twinx()
ax3.plot(fig_9.index[:800:2], fig_9.Displacement[:800:2], linestyle='--', color='b')
ax3.set_ylabel('Displacement [mm]')
ax3.tick_params('y', colors='b')

plt.show()

