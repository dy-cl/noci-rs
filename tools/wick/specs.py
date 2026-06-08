from __future__ import annotations

from dataclasses import dataclass

from core import Idx, Space

@dataclass(frozen = True)
class ExcitationSpec:
    """
    One GNOCCSD excitation type.

    Notation:
        \hat \tau_\mu = \hat E_{q_1 \cdots q_k}^{p_1 \cdots p_k},
        where k is 1 or 2.

    Examples:
        i \in C, u \in A ---> \hat t_\mu = \hat E_i^u 
        t, u in A, a, b in V ---> \hat t_\mu = \hat E_{tu}^{ab}
    """
    
    # Symbolic name for this excitation class.
    name: str
    # Upper indices of \hat E_{q_1 \cdots q_k}^{p_1 \cdots p_k}. 
    creators: tuple[Idx, ...]
    # Lower indices of \hat E_{q_1 \cdots q_k}^{p_1 \cdots p_k}.
    annihilators: tuple[Idx, ...]
    # Name used for emission of Rust functions.
    rustName: str
    # Name used for emission of Latex expressions. 
    latexName: str
    # Variable names used to unpack this excitation's indices.
    unpack: tuple[str, ...]

@dataclass(frozen = True)
class OverlapBlockSpec:
    """
    One generated overlap block. Left excitation stored in daggered form.

    Notation:
        S_{\mu\nu} = \langle \Phi| \hat \tau_\mu^\dagger \hat \tau_\nu | \Phi \rangle.

    Examples:
        Left: CA ---> AV with w \in A, a \in V, i \in C, u \in A ---> \hat E_{iu}^{wa}
        Right: CA ---> VA with b \in V, y \in A, j \in C, x \in A ---> \hat E_{jx}^{by}
        ---> S_{\mu\nu} = \langle \Phi | \hat E_{iu}^{wa} \hat E_{jx}^{by} | \Phi \rangle.

    """
    
    # Label of the overlap block.
    name: str
    # Name of left excitation class.
    left: str
    # Name of right excitation class.
    right: str
    # Name of generated Rust function for this overlap block.
    rustName: str
    # Name of Latex label used when emitting Latex.
    latexName: str
    # Variable names used to unpack left and right excitation indices.
    unpack: tuple[tuple[str, ...], tuple[str, ...]]

def C(name: str) -> Idx:
    """
    Construct a core orbital index.

    Notation:
        i, j, k, l, ... \in C.

    Example:
        C("i") represents i \in C.
    """
    return Idx(name, Space.CORE)

def A(name: str) -> Idx:
    """
    Construct an active orbital index.

    Notation:
        t, u, v, w, ... \in A.
    
    Example:
        A("u") represents u \in A.
    """
    return Idx(name, Space.ACTIVE)

def V(name: str) -> Idx:
    """
    Construct a virtual orbital index.

    Notation:
        a, b, c, d, ... \in V.

    Example:
        V("a") represents a \in V.
    """
    return Idx(name, Space.VIRTUAL)

EXCITATIONS = {
    "CToA": ExcitationSpec("CToA", (A("u"),), (C("i"),), "c_to_a", "CToA", ("u", "i")),
    "CToV": ExcitationSpec("CToV", (V("a"),), (C("i"),), "c_to_v", "CToV", ("a", "i")),
    "AToA": ExcitationSpec("AToA", (A("v"),), (A("u"),), "a_to_a", "AToA", ("v", "u")),
    "AToV": ExcitationSpec("AToV", (V("a"),), (A("u"),), "a_to_v", "AToV", ("a", "u")),
    "CCToAA": ExcitationSpec("CCToAA", (A("u"), A("v")), (C("i"), C("j")), "c_c_to_a_a", "CCToAA", ("u", "v", "i", "j")),
    "CCToAV": ExcitationSpec("CCToAV", (A("u"), V("a")), (C("i"), C("j")), "c_c_to_a_v", "CCToAV", ("u", "a", "i", "j")),
    "CAToAA": ExcitationSpec("CAToAA", (A("v"), A("w")), (C("i"), A("u")), "c_a_to_a_a", "CAToAA", ("v", "w", "i", "u")),
    "CAToAV": ExcitationSpec("CAToAV", (A("v"), V("a")), (C("i"), A("u")), "c_a_to_a_v", "CAToAV", ("v", "a", "i", "u")),
    "CAToVA": ExcitationSpec("CAToVA", (V("a"), A("v")), (C("i"), A("u")), "c_a_to_v_a", "CAToVA", ("a", "v", "i", "u")),
    "CAToVV": ExcitationSpec("CAToVV", (V("a"), V("b")), (C("i"), A("u")), "c_a_to_v_v", "CAToVV", ("a", "b", "i", "u")),
    "AAToAA": ExcitationSpec("AAToAA", (A("t"), A("v")), (A("u"), A("w")), "a_a_to_a_a", "AAToAA", ("t", "v", "u", "w")),
    "AAToAV": ExcitationSpec("AAToAV", (A("v"), V("a")), (A("t"), A("u")), "a_a_to_a_v", "AAToAV", ("v", "a", "t", "u")),
    "AAToVV": ExcitationSpec("AAToVV", (V("a"), V("b")), (A("t"), A("u")), "a_a_to_v_v", "AAToVV", ("a", "b", "t", "u")),
}

OVERLAP_BLOCKS = (
    OverlapBlockSpec("C1", "CToA", "CToA", "overlap_c_to_a_c_to_a", "S_{\\mathbb{C}\\rightarrow\\mathbb{A}}", (("u", "i"), ("v", "j"))),
    OverlapBlockSpec("C2", "AToV", "AToV", "overlap_a_to_v_a_to_v", "S_{\\mathbb{A}\\rightarrow\\mathbb{V}}", (("a", "t"), ("b", "u"))),
    OverlapBlockSpec("C3", "AToA", "AToA", "overlap_a_to_a_a_to_a", "S_{\\mathbb{A}\\rightarrow\\mathbb{A}}", (("v", "u"), ("x", "w"))),
    OverlapBlockSpec("C4", "CAToAV", "CAToAV", "overlap_ca_to_av_ca_to_av", "S_{\\mathbb{CA}\\rightarrow\\mathbb{AV}}", (("v", "a", "i", "u"), ("x", "b", "j", "w"))),
    OverlapBlockSpec("C5", "CAToVA", "CAToVA", "overlap_ca_to_va_ca_to_va", "S_{\\mathbb{CA}\\rightarrow\\mathbb{VA}}", (("a", "v", "i", "u"), ("b", "x", "j", "w"))),
    OverlapBlockSpec("C6", "CAToVV", "CAToVV", "overlap_ca_to_vv_ca_to_vv", "S_{\\mathbb{CA}\\rightarrow\\mathbb{VV}}", (("a", "b", "i", "u"), ("c", "d", "j", "v"))),
    OverlapBlockSpec("C7", "CCToAV", "CCToAV", "overlap_cc_to_av_cc_to_av", "S_{\\mathbb{CC}\\rightarrow\\mathbb{AV}}", (("u", "a", "i", "j"), ("v", "b", "k", "l"))),
    OverlapBlockSpec("C8", "CCToAA", "CCToAA", "overlap_cc_to_aa_cc_to_aa", "S_{\\mathbb{CC}\\rightarrow\\mathbb{AA}}", (("u", "v", "i", "j"), ("w", "x", "k", "l"))),
    OverlapBlockSpec("C9", "CAToAA", "CAToAA", "overlap_ca_to_aa_ca_to_aa", "S_{\\mathbb{CA}\\rightarrow\\mathbb{AA}}", (("v", "w", "i", "u"), ("y", "z", "j", "x"))),
    OverlapBlockSpec("C10", "AAToAV", "AAToAV", "overlap_aa_to_av_aa_to_av", "S_{\\mathbb{AA}\\rightarrow\\mathbb{AV}}", (("v", "a", "t", "u"), ("z", "b", "x", "y"))),
    OverlapBlockSpec("C11", "AAToVV", "AAToVV", "overlap_aa_to_vv_aa_to_vv", "S_{\\mathbb{AA}\\rightarrow\\mathbb{VV}}", (("a", "b", "t", "u"), ("c", "d", "v", "w"))),
    OverlapBlockSpec("C12", "AAToAA", "AAToAA", "overlap_aa_to_aa_aa_to_aa", "S_{\\mathbb{AA}\\rightarrow\\mathbb{AA}}", (("q", "s", "p", "r"), ("t", "v", "u", "w"))),
    OverlapBlockSpec("C13", "AToV", "AAToAV", "overlap_a_to_v_aa_to_av", "S_{\\mathbb{A}\\rightarrow\\mathbb{V},\\,\\mathbb{AA}\\rightarrow\\mathbb{AV}}", (("a", "u"), ("x", "b", "v", "w"))),
    OverlapBlockSpec("C14", "CToA", "CAToAA", "overlap_c_to_a_ca_to_aa", "S_{\\mathbb{C}\\rightarrow\\mathbb{A},\\,\\mathbb{CA}\\rightarrow\\mathbb{AA}}", (("u", "i"), ("w", "x", "j", "v"))),
    OverlapBlockSpec("C15", "AToA", "AAToAA", "overlap_a_to_a_aa_to_aa", "S_{\\mathbb{A}\\rightarrow\\mathbb{A},\\,\\mathbb{AA}\\rightarrow\\mathbb{AA}}", (("u", "t"), ("y", "z", "w", "x"))),
    OverlapBlockSpec("C16", "CAToAV", "CAToVA", "overlap_ca_to_av_ca_to_va", "S_{\\mathbb{CA}\\rightarrow\\mathbb{AV},\\,\\mathbb{CA}\\rightarrow\\mathbb{VA}}", (("w", "a", "i", "u"), ("b", "y", "j", "x"))),
    OverlapBlockSpec("C17", "CToV", "CToV", "overlap_c_to_v_c_to_v", "S_{\\mathbb{C}\\rightarrow\\mathbb{V}}", (("a", "i"), ("b", "j"))),
    OverlapBlockSpec("C18", "CToV", "CAToAV", "overlap_c_to_v_ca_to_av", "S_{\\mathbb{C}\\rightarrow\\mathbb{V},\\,\\mathbb{CA}\\rightarrow\\mathbb{AV}}", (("a", "i"), ("x", "b", "j", "w"))),
    OverlapBlockSpec("C19", "CToV", "CAToVA", "overlap_c_to_v_ca_to_va", "S_{\\mathbb{C}\\rightarrow\\mathbb{V},\\,\\mathbb{CA}\\rightarrow\\mathbb{VA}}", (("a", "i"), ("b", "x", "j", "w"))),
)

def availableExcitations() -> tuple[str, ...]:
    """
    Return the names of all excitation classes known.
    """
    return tuple(EXCITATIONS.keys())

def availableBlocks() -> tuple[str, ...]:
    """
    Return the names of all overlap blocks known.
    """
    return tuple(block.name for block in OVERLAP_BLOCKS)

def overlapBlock(name: str) -> OverlapBlockSpec:
    """
    Look up an overlap block by its name.
    """
    for block in OVERLAP_BLOCKS:
        if block.name == name:
            return block

    raise ValueError(f"unknown overlap block {name}")
