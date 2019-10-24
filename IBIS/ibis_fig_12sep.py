import matplotlib.pyplot as plt
import pandas as pd

fig_12t = pd.read_csv('ibis_r12t.csv', sep=';', parse_dates=False, index_col='Time_t')
fig_12o = pd.read_csv('ibis_r12o.csv', sep=';', parse_dates=False, index_col='Time_o')

print(fig_12t.info())
print()
print(fig_12t.head())
print()
print(fig_12o.info())
print()
print(fig_12o.head())

fig, ax = plt.subplots()
ax.plot(fig_12t.index[:200:1], fig_12t.theoretical[:200:1], linestyle='solid', color='r')
ax.grid()
ax.set_xlabel('Data(yy/mm/dd)')
ax.set_ylabel('Displacement [mm]')
ax.set_title('Atmospheric modeling')
ax.tick_params('y', colors='r')
ax.xaxis.set_tick_params(rotation=90)

fig, ax2 = plt.subplots()
ax2.plot(fig_12o.index[:200:1], fig_12o.observed[:200:1], linestyle='--', color='b')
ax2.grid()
ax2.set_xlabel('Data(yy/mm/dd)')
ax2.set_ylabel('Displacement [mm]')
ax2.set_title('Observed movements')
ax2.tick_params('y', colors='b')
ax2.xaxis.set_tick_params(rotation=90)

plt.show()


# Próby zrobienia razem

# fig, ax = plt.subplots()
# ax.plot(fig_12t.index[:200:1], fig_12t.theoretical[:200:1], linestyle='solid', color='r')
# ax.plot(fig_12o.index[:200:1], fig_12o.observed[:200:1], linestyle='solid', color='b')
# ax.grid()
# ax.set_xlabel('Data(yy/mm/dd)')
# ax.set_ylabel('Displacement [mm]')
# ax.tick_params('y', colors='r')
# ax.xaxis.set_tick_params(rotation=90)
#
# plt.show()

# na jednym dwa nie wychodzi prawidłowo

# fig, ax = plt.subplots()
# ax.plot(fig_12t.index[:300:4], fig_12t.theoretical[:300:4], linestyle='solid', color='r')
# ax.grid()
# ax.set_xlabel('Data(yy/mm/dd)')
# ax.set_ylabel('Theoretical Displacement [mm]')
# ax.tick_params('y', colors='r')
# ax.xaxis.set_tick_params(rotation=90)
#
# ax1 = ax.twinx()
# ax1.plot(fig_12o.index[:300:4], fig_12o.observed[:300:4], linestyle='--', color='b')
# ax1.set_ylabel('Observed Displacement [mm]')
# ax1.tick_params('y', colors='b')
#
# plt.show()
