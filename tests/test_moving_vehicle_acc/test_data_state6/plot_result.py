import numpy as np
import matplotlib.pyplot as plt
import math

# Define variables 

init_state_guess = [2.0, 2.0, 2.0, 2.0, 0.0, 0.0]
init_state_guess_std_dev = [2.0, 2.0, 1.0, 1.0, 1.0, 1.0]
init_real_state = [0.0, 0.0, 1.0, 1.2, 0.5, 0.3]

list_state_names = [r"$x \ / \ \mathrm{m}$", 
                    r"$y \ / \ \mathrm{m}$", 
                    r"$\dot{x} \ / \ \mathrm{m} \cdot \mathrm{s}^{-1}$",
                    r"$\dot{y} \ / \ \mathrm{m} \cdot \mathrm{s}^{-1}$",
                    r"$\ddot{x} \ / \ \mathrm{m} \cdot \mathrm{s}^{-2}$",
                    r"$\ddot{y} \ / \ \mathrm{m} \cdot \mathrm{s}^{-2}$"]
str_time_name = r"$\mathrm{Time} \ / \ \mathrm{s}$"

measure_color = "g"
estimate_color = "b"
real_color = 'r'
scatter_size = 5
plot_line_width = 1

filePath_input_data = "./test_data"
dt = 1.0 # s

# Load data 

list_real_state = [init_real_state]
list_estimate_state = [init_state_guess]
list_measure_state = []
list_estimate_std_dev = [init_state_guess_std_dev]

file_input = open(filePath_input_data, 'r')
content_input = file_input.readlines()
file_input.close()

number_of_measures = int(len(content_input) / 4)
for i_data in range(0, number_of_measures):

    # Configure real state 
    list_line_elem = content_input[4 * i_data + 0].split()
    temp_real_state = []
    for i in range(0, 6):
        temp_real_state.append(float(list_line_elem[i + 1]))
    list_real_state.append(temp_real_state)

    # Configure measure state 
    list_line_elem = content_input[4 * i_data + 1].split()
    temp_measure_state = [float(list_line_elem[0]), float(list_line_elem[1])]
    list_measure_state.append(temp_measure_state)

    # Configure estimate state 
    list_line_elem = content_input[4 * i_data + 2].split()
    temp_estimate_state = []
    for i in range(0, 6):
        temp_estimate_state.append(float(list_line_elem[i]))
    list_estimate_state.append(temp_estimate_state)
    
    # Configure std dev
    list_line_elem = content_input[4 * i_data + 3].split()
    temp_estimate_state_std_dev = []
    for i in range(0, 6):
        temp_estimate_state_std_dev.append(float(list_line_elem[i]))
    for i in range(0, 6):
        temp_estimate_state_std_dev[i] = math.sqrt(temp_estimate_state_std_dev[i])
    list_estimate_std_dev.append(temp_estimate_state_std_dev)

# Re-configure data 

list_estimate_state = np.array(list_estimate_state)
list_estimate_std_dev = np.array(list_estimate_std_dev)
list_measure_state = np.array(list_measure_state)
list_real_state = np.array(list_real_state)

list_estimate_state = list_estimate_state.transpose()
list_estimate_std_dev = list_estimate_std_dev.transpose()
list_measure_state = list_measure_state.transpose()
list_real_state = list_real_state.transpose()

list_time_measure = np.arange(0, len(list_measure_state[0]))
list_time_measure = list_time_measure + 1
list_time_measure = list_time_measure * dt

list_time_estimate = np.arange(0, len(list_estimate_state[0]))
list_time_estimate = list_time_estimate * dt

# Plot data 

for i_fig in range(1, 7):
    plt.subplot(3, 2, i_fig)
    plt.scatter(list_time_estimate, list_estimate_state[i_fig - 1], color=estimate_color, s=scatter_size, label="Estimate")
    plt.fill_between(list_time_estimate,
                     list_estimate_state[i_fig - 1] - 3 * list_estimate_std_dev[i_fig - 1],
                     list_estimate_state[i_fig - 1] + 3 * list_estimate_std_dev[i_fig - 1],
                     color=estimate_color,
                     alpha=0.2,
                     label=r"$3\sigma$ Range")
    plt.plot(list_time_estimate, list_estimate_state[i_fig - 1], color=estimate_color, linewidth=plot_line_width)

    plt.plot(list_time_estimate, list_real_state[i_fig - 1], color=real_color, label="Real", linewidth=plot_line_width)

    if i_fig <=2:
        plt.scatter(list_time_measure, list_measure_state[i_fig - 1], color=measure_color, s=scatter_size, label="Measurement")
        plt.plot(list_time_measure, list_measure_state[i_fig - 1], color=measure_color, linewidth=plot_line_width)
    else:
        pass

    plt.xlabel(str_time_name)
    plt.ylabel(list_state_names[i_fig - 1])
    plt.legend(fontsize=7)

plt.tight_layout()
plt.savefig("test_figure", dpi=300)



