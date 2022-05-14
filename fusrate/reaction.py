from halebosch import HaleBoschCrossSection
from halebosch import HaleBoschReactivity

# canonical reaction names


class Reaction:
    def __init__(self, name):
        # try hb name resolver. If that fails,
        # try pB11 name resolver. If that fails, try
        # # try pLi6
        self.name = name

    def canonical_name():
        return self.name


if __name__ == "__main__":
    pass
    # do the thing
