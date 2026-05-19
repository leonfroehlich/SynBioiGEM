import numpy as np
import matplotlib.pyplot as plt


class PetriDishDiffusionSolver:
    def __init__(
        self,
        dish_size_mm=90,
        dx_mm=1.0,
        diffusion_coefficient=1.0,  # mm^2 / hour
        dt_hours=None
    ):
        self.dish_size_mm = dish_size_mm
        self.dx = dx_mm
        self.dy = dx_mm
        self.D = diffusion_coefficient

        self.nx = int(dish_size_mm / dx_mm)
        self.ny = int(dish_size_mm / dx_mm)

        if dt_hours is None:
            # Stability condition for 2D explicit diffusion
            self.dt = self.dx**2 / (4 * self.D)
        else:
            self.dt = dt_hours

        max_dt = self.dx**2 / (4 * self.D)
        if self.dt > max_dt:
            raise ValueError(
                f"dt is too large and unstable. Use dt <= {max_dt:.4f} hours."
            )

        self.C = np.zeros((self.ny, self.nx))

    def add_disk(self, x_mm, y_mm, radius_mm, concentration):
        """
        Adds an antibiotic disk at position x,y with given radius and initial concentration.
        """
        x0 = int(x_mm / self.dx)
        y0 = int(y_mm / self.dy)
        r = radius_mm / self.dx

        Y, X = np.ogrid[:self.ny, :self.nx]
        mask = (X - x0)**2 + (Y - y0)**2 <= r**2

        self.C[mask] = concentration

    def step(self):
        """
        Runs one diffusion timestep using Forward Euler finite differences.
        """
        C_new = self.C.copy()

        C_new[1:-1, 1:-1] = self.C[1:-1, 1:-1] + self.D * self.dt * (
            (self.C[2:, 1:-1] - 2*self.C[1:-1, 1:-1] + self.C[:-2, 1:-1]) / self.dy**2
            +
            (self.C[1:-1, 2:] - 2*self.C[1:-1, 1:-1] + self.C[1:-1, :-2]) / self.dx**2
        )

        # Petri dish boundary: no antibiotic outside dish
        C_new[0, :] = 0
        C_new[-1, :] = 0
        C_new[:, 0] = 0
        C_new[:, -1] = 0

        self.C = C_new

    def run(self, total_time_hours, save_every_hours=1.0):
        """
        Returns:
        times: array of saved time points
        concentrations: array with shape [time, y, x]
        """
        total_steps = int(total_time_hours / self.dt)
        save_interval = max(1, int(save_every_hours / self.dt))

        saved_times = []
        saved_concentrations = []

        for step in range(total_steps + 1):
            time = step * self.dt

            if step % save_interval == 0:
                saved_times.append(time)
                saved_concentrations.append(self.C.copy())

            self.step()

        return np.array(saved_times), np.array(saved_concentrations)

    def plot(self, concentration, title="Ampicillin concentration"):
        plt.imshow(
            concentration,
            origin="lower",
            extent=[0, self.dish_size_mm, 0, self.dish_size_mm]
        )
        plt.colorbar(label="Concentration")
        plt.xlabel("x position [mm]")
        plt.ylabel("y position [mm]")
        plt.title(title)
        plt.show()


# -------------------------------
# Example setup for your experiment
# -------------------------------

solver = PetriDishDiffusionSolver(
    dish_size_mm=90,
    dx_mm=1.0,
    diffusion_coefficient=0.5540  # adjust/fit this later
)

# Example disk positions in mm
# Change these to match your actual plate image
solver.add_disk(x_mm=50.9, y_mm=66.6, radius_mm=3.9, concentration=2)
solver.add_disk(x_mm=23.8, y_mm=40.6, radius_mm=3.9, concentration=10)
solver.add_disk(x_mm=61.8, y_mm=30.7, radius_mm=3.9, concentration=50)

times, concentrations = solver.run(
    total_time_hours=17,
    save_every_hours=1
)

# concentration at every point and every saved time:
# concentrations[time_index, y, x]

print("Times saved:", times)
print("Output shape:", concentrations.shape)

# Plot final concentration after 17 hours
solver.plot(
    concentrations[-1],
    title="Simulated ampicillin diffusion after 17 h"
)