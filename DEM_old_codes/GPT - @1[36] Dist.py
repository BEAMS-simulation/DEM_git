import pandas as pd
import numpy as np


def load_particle_csv(filename):
    """
    CSV에서 입자 데이터를 읽는다.
    필요한 열: x, y, z, r
    선택 열: id
    """
    df = pd.read_csv(filename)

    required = ["x", "y", "z", "r"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV 파일에 '{col}' 열이 없습니다.")

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    return df


def compute_center_distance_matrix(df):
    """
    중심 좌표 기준 거리행렬 D_ij = ||x_i - x_j||
    """
    coords = df[["x", "y", "z"]].to_numpy(dtype=float)

    diff = coords[:, None, :] - coords[None, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))

    return dist


def compute_surface_gap_matrix(df, center_dist):
    """
    표면 간 거리행렬
    gap_ij = center_dist_ij - (r_i + r_j)

    gap < 0 이면 겹침
    gap = 0 이면 접촉
    gap > 0 이면 떨어져 있음
    """
    r = df["r"].to_numpy(dtype=float)
    gap = center_dist - (r[:, None] + r[None, :])
    return gap


def find_overlapping_pairs(df, center_dist=None, tol=1e-12):
    """
    겹치는 입자쌍 찾기
    overlap_ij = r_i + r_j - d_ij > 0 이면 겹침
    """
    if center_dist is None:
        center_dist = compute_center_distance_matrix(df)

    ids = df["id"].to_numpy()
    r = df["r"].to_numpy(dtype=float)

    n = len(df)
    overlaps = []

    for i in range(n):
        for j in range(i + 1, n):
            d = center_dist[i, j]
            overlap = r[i] + r[j] - d
            if overlap > tol:
                overlaps.append({
                    "id_i": ids[i],
                    "id_j": ids[j],
                    "center_distance": d,
                    "r_i": r[i],
                    "r_j": r[j],
                    "overlap": overlap
                })

    return pd.DataFrame(overlaps)


def save_results(df, center_dist, gap, overlap_df, prefix="particle"):
    """
    결과 저장
    """
    ids = df["id"].to_numpy()

    center_dist_df = pd.DataFrame(center_dist, index=ids, columns=ids)
    gap_df = pd.DataFrame(gap, index=ids, columns=ids)

    center_dist_df.to_csv(f"{prefix}_center_distance_matrix.csv", encoding="utf-8-sig")
    gap_df.to_csv(f"{prefix}_surface_gap_matrix.csv", encoding="utf-8-sig")
    overlap_df.to_csv(f"{prefix}_overlapping_pairs.csv", index=False, encoding="utf-8-sig")


def main():
    input_csv = "final_particles_test.csv"   # 파일명 바꿔서 사용
    output_prefix = "particle"

    df = load_particle_csv(input_csv)

    center_dist = compute_center_distance_matrix(df)
    gap = compute_surface_gap_matrix(df, center_dist)
    overlap_df = find_overlapping_pairs(df, center_dist)

    save_results(df, center_dist, gap, overlap_df, prefix=output_prefix)

    print("완료")
    print(f"입자 수: {len(df)}")
    print(f"거리행렬 저장: {output_prefix}_center_distance_matrix.csv")
    print(f"표면간 거리행렬 저장: {output_prefix}_surface_gap_matrix.csv")
    print(f"겹침 목록 저장: {output_prefix}_overlapping_pairs.csv")
    print(f"겹치는 입자쌍 개수: {len(overlap_df)}")


if __name__ == "__main__":
    main()