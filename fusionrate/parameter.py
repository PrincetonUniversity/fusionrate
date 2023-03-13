from typing import NamedTuple


class Parameter(NamedTuple):
    name: str
    bounds: list
    extrapolable_bounds: list
    unit: str


if __name__ == "__main__":
    p1 = Parameter(name="Energy", bounds=[1, 2], extrapolable_bounds=[0.5,3], unit="keV")
    print(p1)
