import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

r9 = pd.read_csv('surowe_atmo_r9_1.csv', sep=';', parse_dates=False, index_col='Time')
print(r9.columns)
print()
print(r9.index)
print()
print(r9.info())
print()
print(r9.head(-3))
print()
print(r9.iloc[:4, [1, 2]])

fig, ax = plt.subplots()
ax.plot(r9.index[:50], r9.Temperature[:50], linestyle='solid', color='r')
ax.grid()
ax.set_xlabel('Data(yy/mm/dd)')
ax.set_ylabel('Temperature')
ax.tick_params('y', colors='r')
ax.xaxis.set_tick_params(rotation=90)


ax1 = ax.twinx()
ax1.plot(r9.index[:50], r9.Atm_pressure[:50], linestyle='--', color='b')
ax1.set_ylabel('Pressure')
ax1.tick_params('y', colors='b')

plt.show()

# axs[0].plot(t, s1, t, s2)
# axs[0].set_xlim(0, 2)
# axs[0].set_xlabel('time')
# axs[0].set_ylabel('s1 and s2')
# axs[0].grid(True)
#
# cxy, f = axs[1].cohere(s1, s2, 256, 1. / dt)
# axs[1].set_ylabel('coherence')
#
# fig.tight_layout()
# plt.show()