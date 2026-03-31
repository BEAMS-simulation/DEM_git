import pandas as pd
import numpy as np
import polyscope as ps
import trimesh

filename = "final_particles_test_test.csv"   # 파일명 맞게 바꾸세요
df = pd.read_csv(filename)

# 필요한 컬럼: id, x, y, z, r
centers = df[["x", "y", "z"]].to_numpy(dtype=float)
radii = df["r"].to_numpy(dtype=float)
ids = df["id"].to_numpy()

ps.init()
ps.set_ground_plane_mode("tile")   # 착시 줄이려면 none 추천
ps.set_up_dir("z_up")
ps.set_front_dir("y_front")

# 단위 구 템플릿 1회 생성
sphere = trimesh.creation.icosphere(subdivisions=3, radius=1.0)
V0 = sphere.vertices
F = sphere.faces

for pid, center, radius in zip(ids, centers, radii):
    V = V0 * radius + center[None, :]

    ps_mesh = ps.register_surface_mesh(
        f"particle_{int(pid)}",
        V,
        F
    )
    ps_mesh.set_color((0.2, 0.5, 0.9))

print("min(z-r) =", np.min(df["z"].to_numpy(float) - radii))

ps.show()