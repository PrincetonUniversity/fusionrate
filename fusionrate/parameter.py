from typing import NamedTuple


class Parameter(NamedTuple):
    name: str
    bounds: list
    unit: str


if __name__ == "__main__":
    p1 = Parameter(name="Energy", bounds=[1, 2], unit="keV")
    print(p1)
    p2 = Parameter("Cow", [3, 4, 5], "keV")
    print(p2)
    print(p2[1])
    print(p2.bounds)
    # This works too, though I don't like it.
    p3 = Parameter("BadCow", "keV", [5, 6])
    print(p3)
