import pandas as pd
import polyscope as ps

df = pd.read_csv("final_particles_test_test.csv")

centers = df[["x", "y", "z"]].to_numpy(float)
radii = df["r"].to_numpy(float)

ps.init()
ps.set_ground_plane_mode("none")

pc = ps.register_point_cloud("final_particles", centers)

# 반지름 quantity 추가
pc.add_scalar_quantity("radius", radii, enabled=True)

# 핵심: autoscale=False
pc.set_point_radius_quantity("radius", autoscale=False)

ps.show()