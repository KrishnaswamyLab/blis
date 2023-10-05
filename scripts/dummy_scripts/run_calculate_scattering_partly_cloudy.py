import subprocess

def run_calculate_scattering():
    base_command = [
        "python", "classify_scattering.py",
        "--wavelet_type", "W2",
        "--largest_scale", "4",
        "--highest_moment", "3",
        "--dataset", "partly_cloudy"
    ]
    
    scattering_types = ["blis", "modulus"]
    
    for scattering_type in scattering_types:
        for i in range(155):  # 155 because it will generate numbers from 0 to 154 inclusive
            # Formatting the number to be 4 digits
            sub_dataset_value = f"{i:04d}"
            # Construct the command
            command = base_command + [
                "--scattering_type", scattering_type,
                "--sub_dataset", sub_dataset_value
            ]
            # Run the command
            subprocess.run(command)

if __name__ == "__main__":
    run_calculate_scattering()
