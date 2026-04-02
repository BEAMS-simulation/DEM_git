"""DEM 기반 강체(aggregate-of-spheres) 시뮬레이터.
- 소프트-스피어(패널티) 방식의 DEM 접촉(수직/접선/마찰/롤링) 지원
- KDK(velocity Verlet) 기반 시간적분
- 브로드페이즈: 균일 격자(cell linked-list) 기반 후보 쌍 생성
"""

__all__ = [
"config",
"math3d",
"model",
"broadphase",
"contact",
"world",
"simulator",
"io_csv",
]
__version__ = "0.1.0"