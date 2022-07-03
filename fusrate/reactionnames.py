DT_NAME = "T(d,n)⁴He"
DHE3_NAME = "³He(d,p)⁴He"
DDT_NAME = "D(d,p)T"
DDHE3_NAME = "D(d,n)³He"

PLI6_NAME = "⁶Li(p,α)³He"
PB11_NAME = "¹¹B(p,α)2⁴He"

TT_NAME = "T(t,2n)⁴He"
HT_NAME = "³He(t,pn)⁴He"
HH_NAME = "³He(h,2p)⁴He"
DLI6A_NAME = "⁶Li(d,α)⁴He"
DLI6N_NAME = "⁶Li(d,n)⁷Be"
DLI6P_NAME = "⁶Li(d,p)⁷Li"

ALL_REACTIONS = [
    DT_NAME,
    DHE3_NAME,
    DDT_NAME,
    DDHE3_NAME,
    PLI6_NAME,
    PB11_NAME,
]


def name_resolver(reaction_raw_name):
    r"""Recognize a canonical fusion reaction

    The name resolver aims to recognize the common fusion reactions
    from a string, in a reasonable number of unambiguous formats.

    Parameters
    ----------
    reaction_raw_name: string
        The name resolver aims to recognize the common fusion reactions
        from a string, in a reasonable number of unambiguous formats.

    Examples
    --------
    name_resolver("DT")      --> T(d,n)⁴He
    name_resolver("D+T→α+n") --> T(d,n)⁴He

    name_resolver("D+3He")   --> ³He(d,p)T

    Notes
    -----
    This could be implemented in a better way by parsing the
    reactants and products rather than doing a set of string replacements.
    """
    canonical_name = None
    if canonical_name is None:
        try:
            canonical_name = bosch_hale_name_resolver(reaction_raw_name)
        except ValueError:
            pass

    if canonical_name is None:
        try:
            canonical_name = proton_boron_name_resolver(reaction_raw_name)
        except ValueError:
            pass

    if canonical_name is not None:
        return canonical_name
    else:
        raise ValueError(
            """
        The reaction name resolver could not determine the reaction
        {reaction_raw_name}. The best options are {ALL_REACTIONS}.
        """
        )


def reaction_name_simplify(reaction_name_raw):
    s = reaction_name_raw.replace(" ", "")
    s = s.replace("¹", "1")
    s = s.replace("²", "2")
    s = s.replace("³", "3")
    s = s.replace("⁴", "4")
    s = s.replace("⁶", "6")
    s = s.replace("⁷", "7")
    s = s.replace("->", "→")
    s = s.replace("-", "+")
    s = s.replace("α", "a")
    return s


def reaction_name_to_endf(canonical_reaction_name):
    s = canonical_reaction_name
    s = s.replace("4He", "a")
    s = s.replace("3He", "h")
    s = s.replace("T", "t")
    s = s.replace("D", "d")
    return s


def proton_boron_name_resolver(reaction_raw_name):
    NAMES = [
        PB11_NAME,
        "pB",
        "pB11",
        "p+B",
        "p+B11",
        "p+11B",
        "p+11B→3α",
        "p+11B→3 ⁴He",
    ]

    reaction_name = reaction_name_simplify(reaction_raw_name)

    for n in NAMES:
        if reaction_name == reaction_name_simplify(n):
            return PB11_NAME

    raise ValueError(
        f"""
        In the proton_boron_name_resolver, {reaction_raw_name} could not be
        resolved. Possible options are {NAMES}.
        """
    )


def bosch_hale_name_resolver(reaction_raw_name):
    DT_NAMES = [DT_NAME, "DT", "D+T", "D+T→n+α", "D+T→α+n"]

    DHE3_NAMES = [
        DHE3_NAME,
        "DHe",
        "D3He",
        "D+3He",
        "DHe3",
        "D+³He→p+⁴He",
        "D+³He→⁴He+p",
        "D+³He→p+α",
        "D+³He→α+p",
    ]

    DDT_NAMES = [
        DDT_NAME,
        "D+D→p+T",
        "D+D→T+p",
        "²H+²H→³H+¹H",
        "²H+²H→¹H+³H",
    ]

    DDHE3_NAMES = [
        DDHE3_NAME,
        "D(d,n)3He",
        "D+D→n+3He",
        "D+D→3He+n",
        "²H+²H→n+3He",
        "²H+²H→3He+n",
    ]

    reaction_name = reaction_name_simplify(reaction_raw_name)

    for n in DT_NAMES:
        if reaction_name == reaction_name_simplify(n):
            return DT_NAME

    for n in DHE3_NAMES:
        if reaction_name == reaction_name_simplify(n):
            return DHE3_NAME

    for n in DDHE3_NAMES:
        if reaction_name == reaction_name_simplify(n):
            return DDHE3_NAME

    for n in DDT_NAMES:
        if reaction_name == reaction_name_simplify(n):
            return DDT_NAME

    raise ValueError(
        f"""
        In the bosch_hale_name_resolver, {reaction_raw_name} could not be
        resolved. Possible options are {DT_NAMES}, {DHE3_NAMES}, {DDT_NAMES},
        {DDHE3_NAMES}."""
    )
