"""config.py — All configurable constants for the FCC Event Display.

Modify this file to adapt the display to a different detector.
"""

# ── Detector collection groups ────────────────────────────────────────────────
# Each group maps to a list of EDM4HEP collection names and a default color.
DETECTOR_GROUPS: dict = {
    "tracker": {"collections": ["SiTracks_Refitted"], "color": "#9b59b6"},
    "ecal":    {"collections": ["ECALBarrel", "ECALEndcap", "ECALOther"], "color": "#f39c12"},
    "hcal":    {"collections": ["HCALBarrel", "HCALEndcap", "HCALOther"], "color": "#2ecc71"},
    "muon":    {"collections": ["MUON"], "color": "#3498db"},
}

# Flat lookup: collection name → group name
COLLECTION_TO_GROUP: dict = {
    col: grp
    for grp, info in DETECTOR_GROUPS.items()
    for col in info["collections"]
}

# ── Special collections ───────────────────────────────────────────────────────
TRACK_COLLECTION      = "SiTracks_Refitted"
MC_COLLECTION         = "MCParticles"
PFO_COLLECTION        = "PandoraPFOs"
CALO_LINK_COLLECTION  = "CalohitMCTruthLink"

# SimTrackerHit collections (EDM4HEP sim hits in tracker, carry direct MCParticle link)
SIM_TRACKER_COLLECTIONS = [
    "VertexBarrelHits",
    "VertexEndcapHits",
    "InnerTrackerBarrelHits",
    "InnerTrackerEndcapHits",
    "OuterTrackerBarrelHits",
    "OuterTrackerEndcapHits",
]

# RecoMCTruthLink collection names to try (in order)
TRUTH_LINK_VARIANTS = [
    "RecoMCTruthLink",
    "PandoraPFOsToMCParticles",
]

# ── Particle filters ──────────────────────────────────────────────────────────
NEUTRINO_PDGS    = {12, 14, 16}
VALID_GEN_STATUS = {1}

# PDG → human-readable name
PDG_NAMES: dict = {
    11:    "e⁻",    -11:   "e⁺",
    13:    "μ⁻",    -13:   "μ⁺",
    15:    "τ⁻",    -15:   "τ⁺",
    12:    "νe",    -12:   "ν̄e",
    14:    "νμ",    -14:   "ν̄μ",
    16:    "ντ",    -16:   "ν̄τ",
    22:    "γ",
    111:   "π⁰",
    211:   "π⁺",   -211:  "π⁻",
    130:   "K⁰L",
    310:   "K⁰S",
    321:   "K⁺",   -321:  "K⁻",
    2212:  "p",    -2212: "p̄",
    2112:  "n",    -2112: "n̄",
    3122:  "Λ",    -3122: "Λ̄",
    3222:  "Σ⁺",   -3222: "Σ̄⁻",
    3112:  "Σ⁻",   -3112: "Σ̄⁺",
    411:   "D⁺",   -411:  "D⁻",
    421:   "D⁰",   -421:  "D̄⁰",
    431:   "Ds⁺",  -431:  "Ds⁻",
    521:   "B⁺",   -521:  "B⁻",
    511:   "B⁰",   -511:  "B̄⁰",
}


def pdg_label(pdg: int) -> str:
    """Return 'name (pdg)' string, e.g. 'e⁻ (11)'."""
    name = PDG_NAMES.get(pdg, f"PDG {pdg}")
    return f"{name} ({pdg})"


# ── Themes ────────────────────────────────────────────────────────────────────
THEMES: dict = {
    "dark": {
        "app_bg":    "#1a1a2e",
        "card_bg":   "#16213e",
        "plot_bg":   "#0f3460",
        "sidebar_bg":"#0d1b35",
        "text":      "#e0e0e0",
        "muted":     "#888888",
        "accent":    "#e94560",
        "grid":      "#334455",
        "border":    "#334",
        "tab_bg":    "#16213e",
        "tab_sel":   "#0f3460",
        "input_bg":  "#0f1a30",
        "btn_bg":    "#e94560",
        "btn_text":  "#ffffff",
        "table_header_bg":  "#0f3460",
        "table_header_txt": "#e0e0e0",
        "table_cell_bg":    "#16213e",
        "table_cell_txt":   "#e0e0e0",
        "table_border":     "#334",
    },
    "light": {
        "app_bg":    "#f0f4f8",
        "card_bg":   "#ffffff",
        "plot_bg":   "#eef2ff",
        "sidebar_bg":"#e8edf5",
        "text":      "#1a1a2e",
        "muted":     "#555555",
        "accent":    "#c0392b",
        "grid":      "#cccccc",
        "border":    "#d0d0d0",
        "tab_bg":    "#ffffff",
        "tab_sel":   "#dce6ff",
        "input_bg":  "#f9f9f9",
        "btn_bg":    "#c0392b",
        "btn_text":  "#ffffff",
        "table_header_bg":  "#dce6ff",
        "table_header_txt": "#1a1a2e",
        "table_cell_bg":    "#ffffff",
        "table_cell_txt":   "#1a1a2e",
        "table_border":     "#d0d0d0",
    },
}
DEFAULT_THEME = "dark"

# ── Default geometry ──────────────────────────────────────────────────────────
# Relative path from k4geo_DIR env var
CLD_COMPACT_RELPATH = "FCCee/CLD/compact/CLD_o2_v06/CLD_o2_v06.xml"

# Approximate CLD geometry fallback (mm) used when XML is not available
CLD_GEOMETRY_FALLBACK = {
    "ecal_barrel": {"rmin": 1470.0, "rmax": 1788.0, "zhalf": 2350.0},
    "ecal_endcap": {"rmin": 200.0,  "rmax": 2088.0, "zpos":  2450.0, "zthick": 400.0},
    "hcal_barrel": {"rmin": 1808.0, "rmax": 3188.0, "zhalf": 2850.0},
    "hcal_endcap": {"rmin": 300.0,  "rmax": 3188.0, "zpos":  2900.0, "zthick": 1100.0},
    "muon_barrel": {"rmin": 4200.0, "rmax": 4600.0, "zhalf": 5800.0},
}
