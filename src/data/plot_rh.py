import pandas as pd
import matplotlib.pyplot as plt

from src.run_paths import segmented_rh_run_file


df = pd.read_csv(segmented_rh_run_file())

# Filter PIT data
pit_df = df[df["segment_final"] == "pit"].copy()

# Extract variables
x = pit_df["carspeed_art"]
y = pit_df["rh_f"]

# Clean NaNs (important for plotting)
mask = x.notna() & y.notna()
x = x[mask]
y = y[mask]

# Plot
plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6)

plt.xlabel("Speed (km/h)")
plt.ylabel("Front Ride Height (rh_f)")
plt.title("rh_f vs Speed (Pit Lane Only)")

plt.grid(True)
plt.show()
