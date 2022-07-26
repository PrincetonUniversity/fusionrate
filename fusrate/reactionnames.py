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
    TT_NAME,
    HT_NAME,
    HH_NAME,
    DLI6A_NAME,
    DLI6N_NAME,
    DLI6P_NAME,
]


def target_species(s):
    r"""Name of reaction target species
    Parameters
    ----------
    s: string
        canonical reaction name

    Returns
    -------
    Canonical non-charge-specific species form

    Examples
    --------
    >>> reaction_target_species("T(d,n)⁴He")
    'T'

    >>> reaction_target_species("³He(t,pn)⁴He")
    '³He'
    """
    return s.split("(")[0]


def particle_form_to_target_form(s):
    r"""For supported species only

    These species appear in the canonical reaction names.
    Here we convert to 'uppercase' species forms.

    Parameters
    ----------
    s : {'p', 'd', 't', 'h'}
        lowercase species form

    Returns
    -------
    Canonical non-charge-specific species form

    Examples
    --------
    >>> particle_form_to_target_form('p')
    'H'

    >>> particle_form_to_target_form('h')
    '³He'
    """
    d = {"p": "H", "d": "D", "t": "T", "h": "³He"}
    return d[s]


def beam_species(s):
    r"""Name of reaction beam species

    Parameters
    ----------
    s: string
        canonical reaction name

    Returns
    -------
    Canonical non-charge-specific species form

    Examples
    --------
    >>> reaction_beam_species("T(d,n)⁴He")
    'D'

    >>> reaction_beam_species("³He(t,pn)⁴He")
    'T'
    """
    reactants = s.split(",")[0]
    beam_sp = reactants.split("(")[1]
    return particle_form_to_target_form(beam_sp)


def reactants(s):
    r"""Names of beam, target species

    Parameters
    ----------
    s: string
        canonical reaction name

    Returns
    -------
    Tuple of particles in canonical non-charge-specific species form

    Examples
    --------
    >>> reactants("T(d,n)⁴He")
    'D', 'T'

    >>> reactants("³He(t,pn)⁴He")
    'T', '³He'
    """
    beam = beam_species(s)
    target = target_species(s)
    return beam, target


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
    >>> name_resolver("DT")
    'T(d,n)⁴He'

    >>> name_resolver("D+T→α+n")
    'T(d,n)⁴He'

    >>> name_resolver("D+3He")
    '³He(d,p)T'

    Notes
    -----
    This could be implemented in a better way by parsing the
    reactants and products rather than doing a set of string replacements.
    """
    canonical_name = None
    if canonical_name is None:
        try:
            canonical_name = bosch_name_resolver(reaction_raw_name)
        except ValueError:
            pass

    if canonical_name is None:
        try:
            canonical_name = proton_boron_name_resolver(reaction_raw_name)
        except ValueError:
            pass

    if canonical_name is None:
        try:
            canonical_name = proton_lithium_name_resolver(reaction_raw_name)
        except ValueError:
            pass

    if canonical_name is not None:
        return canonical_name
    else:
        raise ValueError(
            f"""
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


def reaction_filename_part(canonical_reaction_name):
    s = reaction_name_simplify(canonical_reaction_name)
    s = reaction_name_to_endf(s)
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


def proton_lithium_name_resolver(reaction_raw_name):
    NAMES = [
        PLI6_NAME,
        "pLi6",
        "p+Li6",
        "p+6Li",
        "p+Li6→α+³He",
        "p+Li6→³He+α",
        "p+Li6→³He+⁴He",
        "p+Li6→⁴He+³He",
        "Li6+p→α+³He",
        "Li6+p→³He+α",
        "Li6+p→³He+⁴He",
        "Li6+p→⁴He+³He",
        "p+6Li→α+³He",
        "p+6Li→³He+α",
        "p+6Li→³He+⁴He",
        "p+6Li→⁴He+³He",
    ]

    reaction_name = reaction_name_simplify(reaction_raw_name)

    for n in NAMES:
        if reaction_name == reaction_name_simplify(n):
            return PLI6_NAME

    raise ValueError(
        f"""
        In the proton_lithium_name_resolver, {reaction_raw_name} could not be
        resolved. Possible options are {NAMES}.
        """
    )


def bosch_name_resolver(reaction_raw_name):
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
        In the bosch_name_resolver, {reaction_raw_name} could not be
        resolved. Possible options are {DT_NAMES}, {DHE3_NAMES}, {DDT_NAMES},
        {DDHE3_NAMES}."""
    )
