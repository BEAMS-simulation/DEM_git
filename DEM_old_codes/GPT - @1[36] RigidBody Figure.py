import pandas as pd
import numpy as np
import polyscope as ps
import matplotlib.pyplot as plt
import trimesh

filename = "test.csv"
# filename = "final_bodies.csv"

df = pd.read_csv(filename)

ps.init()
ps.set_ground_plane_mode("tile")
ps.set_up_dir("z_up")
ps.set_front_dir("y_front")

body_ids = sorted(df["body id"].unique())
cmap = plt.get_cmap("tab20", len(body_ids))

# 단위 구 템플릿
sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
V0 = sphere.vertices
F = sphere.faces

for k, bid in enumerate(body_ids):
    sub = df[df["body id"] == bid]
    color = cmap(k)[:3]

    for row_idx, row in sub.iterrows():
        center = np.array([row["x"], row["y"], row["z"]], dtype=float)
        radius = float(row["r"])

        V = V0 * radius + center[None, :]

        ps_mesh = ps.register_surface_mesh(
            f"body_{bid}_particle_{int(row['particle id'])}",
            V,
            F
        )
        ps_mesh.set_color(color)

ps.show()