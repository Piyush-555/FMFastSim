import argparse

import numpy as np

from core.constants import INIT_DIR, GEN_DIR, N_CELLS_PHI, N_CELLS_R, N_CELLS_Z
from utils.observables import LongitudinalProfile, LateralProfile, Energy
from utils.plotters import ProfilePlotter, EnergyPlotter
from utils.preprocess import load_showers


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--geometry", type=str, default="")
    p.add_argument("--energy", type=int, default="")
    p.add_argument("--angle", type=int, default="")
    args = p.parse_args()
    return args


def validate(e_layer_g4, e_layer_vae, particle_energy, particle_angle, geometry):
    # Reshape the events into 3D
    e_layer_vae = e_layer_vae.reshape((len(e_layer_vae), N_CELLS_R, N_CELLS_PHI, N_CELLS_Z))

    # 3. Create observables from raw data.
    full_sim_long = LongitudinalProfile(_input=e_layer_g4)
    full_sim_lat = LateralProfile(_input=e_layer_g4)
    full_sim_energy = Energy(_input=e_layer_g4)
    ml_sim_long = LongitudinalProfile(_input=e_layer_vae)
    ml_sim_lat = LateralProfile(_input=e_layer_vae)
    ml_sim_energy = Energy(_input=e_layer_vae)

    # 4. Plot observables
    longitudinal_profile_plotter = ProfilePlotter(particle_energy, particle_angle, geometry, full_sim_long, ml_sim_long,
                                                  _plot_gaussian=False)
    lateral_profile_plotter = ProfilePlotter(particle_energy, particle_angle,
                                             geometry, full_sim_lat, ml_sim_lat, _plot_gaussian=False)
    energy_plotter = EnergyPlotter(particle_energy, particle_angle, geometry, full_sim_energy, ml_sim_energy)

    longitudinal_profile_plotter.plot_and_save()
    lateral_profile_plotter.plot_and_save()
    energy_plotter.plot_and_save()


# main function
def main():
    # Parse commandline arguments
    args = parse_args()
    particle_energy = args.energy
    particle_angle = args.angle
    geometry = args.geometry
    # 1. Full simulation data loading
    # Load energy of showers from a single geometry, energy and angle
    e_layer_g4 = load_showers(INIT_DIR, geometry, particle_energy,
                              particle_angle)
    # 2. Fast simulation data loading, scaling to original energy range & reshaping
    e_layer_vae = np.load(f"{GEN_DIR}/Geo_{geometry}_E_{particle_energy}_Angle_{particle_angle}.npy")

    print("Data has been loaded.")
    validate(e_layer_g4, e_layer_vae, particle_energy, particle_angle, geometry)
    print("Done.")


if __name__ == "__main__":
    exit(main())
