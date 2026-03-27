import numpy as np

def aaa(a: int = 1, b: int = 2, c: int = 3, d: int = 4):
    return a + b + c + d

def test(t, *args):
    match t:
        case 1 | 2:
            return "happy"
        case 3 | 2:
            return "ppap"
        case _:
            raise RuntimeError("aaaaaaaaaaaa")
    return aaa(*args)

print(test(4141))