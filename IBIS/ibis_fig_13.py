import matplotlib.pyplot as plt
import pandas as pd

fig_13 = pd.read_csv('ibis_atm_r13.csv', sep=';', parse_dates=False, index_col='Time')

print(fig_13.info())
print()
print(fig_13.head())

fig, ax = plt.subplots()
ax.plot(fig_13.index[:200:2], fig_13.R14_po_red_a_g[:200:2], linestyle='solid', color='r')
ax.plot(fig_13.index[:200:2], fig_13.R9_po_red_a_g[:200:2], linestyle='solid', color='b')
ax.plot(fig_13.index[:200:2], fig_13.R19_po_red_a_g[:200:2], linestyle='solid', color='y')
ax.grid()
ax.set_xlabel('Data(yy/mm/dd)')
ax.set_ylabel('Displacement [mm]')
ax.set_title('Displacements of selected points after the atmospheric and geometric correction')
ax.tick_params('y', colors='r')
ax.xaxis.set_tick_params(rotation=90)

plt.show()