import os

# list of preprocessing scripts to run
scripts = [
    "technology/preprocess_technology_close.py",
    "technology/preprocess_technology_volume.py",
    "technology/preprocess_technology_market_cap.py",
    "technology/preprocess_technology.py",
    "energy/preprocess_energy_close.py",
    "energy/preprocess_energy_volume.py",
    "energy/preprocess_energy_market_cap.py",
    "energy/preprocess_energy.py"
]

# run each preprocessing script
for script in scripts:
    os.system(f"python {script}")

print("[+] All preprocessing scripts completed successfully!")