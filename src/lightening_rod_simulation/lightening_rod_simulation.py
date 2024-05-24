import matplotlib.pyplot as plt
import numpy as np
import math
import os
import tkinter as tk
from tkinter import filedialog

def initialize_constants(N):
    """Initialize and return the physical and computational constants."""
    return {
        'a': 0.005,  # Radius of the rods
        'L': 1.2,  # Length of the vertical rod
        'D': 1.2,  # Length of the horizontal rod
        'h': 0.75,  # Depth of the horizontal rod from the ground surface
        'sigma_0': 1/150,  # Specific conductivity of the ground
        'I': 400,  # Current to be distributed into the ground
        'N': N,  # Number of sections each rod is divided into
        'xmin': -5,  # Min x for potential plot
        'xmax': 5,  # Max x for potential plot
        'plot_points': 100  # Number of points in the x range for plotting
    }

def calculate_section_lengths(constants):
    """Calculate the lengths of the pieces for each rod section."""
    section_lengths_horizontal = np.full(constants['N'], constants['D'] / constants['N'])
    section_lengths_vertical = np.full(constants['N'], constants['L'] / constants['N'])
    return np.concatenate((section_lengths_horizontal, section_lengths_vertical))

def calculate_section_centers(constants):
    """Calculate the center coordinates for each rod section."""
    x_centers_horizontal = np.linspace(-constants['D'] / 2, constants['D'] / 2, constants['N'] + 1)
    x_centers_horizontal = (x_centers_horizontal[:-1] + x_centers_horizontal[1:]) / 2

    z_centers_vertical = np.linspace(-constants['h'] - constants['L'], -constants['h'], constants['N'] + 1)
    z_centers_vertical = (z_centers_vertical[:-1] + z_centers_vertical[1:]) / 2

    x_centers = np.concatenate((x_centers_horizontal, np.zeros(constants['N'])))
    z_centers = np.concatenate((np.full(constants['N'], -constants['h']), z_centers_vertical))

    return x_centers, z_centers

def calculate_vdf_matrix(x_centers, z_centers, section_lengths, constants):
    """Calculate the Voltage to Current (VDF) matrix for mutual resistances."""
    N = constants['N'] * 2
    VDF = np.zeros((N, N))

    for i in range(N):
        for j in range(N):
            if i == j:
                R1 = section_lengths[i] / 2 + np.sqrt((section_lengths[i] / 2)**2 + constants['a']**2)
                R2 = -section_lengths[i] / 2 + np.sqrt((section_lengths[i] / 2)**2 + constants['a']**2)
                VDF[i, j] = (np.log(R1 / R2) + section_lengths[i] / abs(2 * z_centers[i])) / (4 * math.pi * constants['sigma_0'])
            else:
                R1 = np.sqrt((x_centers[i] - x_centers[j])**2 + (z_centers[i] - z_centers[j])**2)
                R2 = np.sqrt((x_centers[i] - x_centers[j])**2 + (z_centers[i] + z_centers[j])**2)
                VDF[i, j] = section_lengths[j] * (1 / R1 + 1 / R2) / (4 * math.pi * constants['sigma_0'])

    return VDF

def calculate_current_distribution(VDF, section_length, constants):
    """Calculate the current distribution across each rod section."""
    i_0 = np.dot(np.linalg.inv(VDF), np.ones(2*constants['N']))
    i_total = np.sum(i_0 * section_length)
    current_distribution = i_0 * constants['I'] / i_total
    return current_distribution

def calculate_ground_potential(x_centers, z_centers, currents, section_lengths, constants, x_position):
    """Calculate the potential at the ground level for a given x position."""
    potential = sum(currents[i] * section_lengths[i] * 2 / np.sqrt((x_position - x_centers[i])**2 + z_centers[i]**2)
                    for i in range(len(currents)))
    return potential / (4 * math.pi * constants['sigma_0'])

def plot_results(x_values, potentials, currents):
    """Plot the potential at ground level and the current distribution."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Plot potential at ground level
    ax1.plot(x_values, potentials, 'b-')
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('Potential (V)')
    ax1.set_title('Potential at Ground Level')

    # Plot current distribution as a line plot
    ax2.plot(range(1, len(currents) + 1), currents, 'r-')
    ax2.set_xlabel('Section')
    ax2.set_ylabel('Current (A)')
    ax2.set_title('Current Distribution')

    plt.tight_layout()
    return fig

def save_results(fig):
    """Save the plotted results as an image."""
    file_path = filedialog.askdirectory(title='Select Directory to Save Results')
    if file_path:
        fig.savefig(os.path.join(file_path, 'Results.png'))
        tk.messagebox.showinfo('Save Results', 'Results saved successfully!')

def main():
    # Create a Tkinter window
    root = tk.Tk()
    root.title('Ground Potential and Current Distribution')
    root.geometry('300x200')

    def on_run():
        N = int(entry_N.get())
        constants = initialize_constants(N)
        section_lengths = calculate_section_lengths(constants)
        x_centers, z_centers = calculate_section_centers(constants)
        VDF = calculate_vdf_matrix(x_centers, z_centers, section_lengths, constants)
        currents = calculate_current_distribution(VDF, section_lengths, constants)
        x_values = np.linspace(constants['xmin'], constants['xmax'], constants['plot_points'])
        potentials = np.array([calculate_ground_potential(x_centers, z_centers, currents, section_lengths, constants, x) for x in x_values])
        fig = plot_results(x_values, potentials, currents)
        plt.show()
        save_results(fig)

    # Create input field and button
    tk.Label(root, text='Enter N:').pack(pady=10)
    entry_N = tk.Entry(root)
    entry_N.pack(pady=10)
    tk.Button(root, text='Run Simulation', command=on_run).pack(pady=20)

    root.mainloop()

if __name__ == '__main__':
    main()

