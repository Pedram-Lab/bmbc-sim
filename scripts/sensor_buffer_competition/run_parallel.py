import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path


def run_simulation(buffer_conc, buffer_kd):
    """Run a single simulation with specified parameters."""
    print(f"Running simulation with Buffer_Conc = {buffer_conc} mM, Buffer_KD = {buffer_kd} uM")
    result = subprocess.run(
        [
            "python", 
            str(Path(__file__).parent / "simulation.py"),
            "--buffer_conc", str(buffer_conc),
            "--buffer_kd", str(buffer_kd)
        ],
        capture_output=True
    )
    return result.returncode


def main():
    buffer_concs = [1e-6, 1e-3, 1e0, 1e3]
    buffer_kds = [1e-6, 1e-3, 1e0, 1e3]
    max_workers = 2

    # Create all combinations of parameters
    simulations = [(conc, kd) for conc in buffer_concs for kd in buffer_kds]
    print(f"Starting {len(simulations)} simulations with max {max_workers} workers")

    # Run simulations in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        futures = {executor.submit(run_simulation, conc, kd): (conc, kd) 
                   for conc, kd in simulations}

        # Process results as they complete
        for future in as_completed(futures):
            conc, kd = futures[future]
            try:
                return_code = future.result()
                if return_code == 0:
                    print(f"Simulation with Buffer_Conc = {conc} mM, Buffer_KD = {kd} uM completed successfully")
                else:
                    print(f"Simulation with Buffer_Conc = {conc} mM, Buffer_KD = {kd} uM failed with code {return_code}")
            except Exception as e:
                print(f"Simulation with Buffer_Conc = {conc} mM, Buffer_KD = {kd} uM raised an exception: {e}")


if __name__ == "__main__":
    main()
