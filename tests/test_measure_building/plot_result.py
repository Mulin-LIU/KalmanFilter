import matplotlib.pyplot as plt
import math
import numpy as np

# Define variables 

real_building_height = 50.0
init_guess = 60.0
init_std_dev = 10.0

list_estimate = [init_guess]
list_estimate_std_dev = [init_std_dev]
list_measurement = []

# Load Data

file_input = open("./test_data", 'r')
content_input = file_input.readlines()
file_input.close()

for line in content_input:
    line_elem = line.split()
    list_measurement.append(float(line_elem[1]))
    list_estimate.append(float(line_elem[2]))
    list_estimate_std_dev.append(math.sqrt(float(line_elem[3])))

# Reconfigure data

list_real_height = np.zeros(len(list_estimate))
list_real_height = list_real_height + real_building_height

list_time_measure = range(1, len(list_estimate))
list_time_estimate = range(0, len(list_estimate))

list_estimate = np.array(list_estimate)
list_estimate_std_dev = np.array(list_estimate_std_dev)

# Plot data 

plt.scatter(list_time_estimate, list_estimate, label="Estimate", color='b', s=5)
plt.plot(list_time_estimate, list_estimate, color='b', linewidth=1)
plt.fill_between(list_time_estimate,
                 list_estimate - 3 * list_estimate_std_dev, 
                 list_estimate + 3 * list_estimate_std_dev, 
                 color='b', 
                 alpha=0.2,
                 label=r"$3 \sigma$ Range")
plt.scatter(list_time_measure, list_measurement, label="Measurement", color='g', s=5)
plt.plot(list_time_measure, list_measurement, color='g', linewidth=1)
plt.plot(list_time_estimate, list_real_height, label="Real Height", color='r', linewidth=1)
plt.xlabel("Iteration Index")
plt.ylabel("Height / (m)")

plt.legend()
plt.savefig("test_figure", dpi=300)



