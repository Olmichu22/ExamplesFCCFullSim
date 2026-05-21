# TTree Structure — `TTreesTausLong.py`

The output ROOT file contains a single TTree named **`Tau_tree`**. All vectors are indexed **per tau / per photon** within the event — scalar branches hold one value per event.

---

## Scalar branches (one value per event)

| Branch | Type | Description |
|--------|------|-------------|
| `numGenTaus` | `int` | Number of generator-level taus |
| `numRecoTaus` | `int` | Number of reconstructed objects (taus + electrons + muons) |
| `numGenPhotons` | `int` | Number of gen-level photons (`generatorStatus == 1`, PDG 22) |
| `numRecoPhotons` | `int` | Number of reco photons (from PandoraPFOs, PDG 22) |
| `beamE` | `float` | Beam energy, read from `MCParticles[0].getEnergy()` |

---

## Generator-level taus

Each entry corresponds to one gen tau (up to `numGenTaus` entries per event).

| Branch | Description |
|--------|-------------|
| `GenEventId` | Global event ID (across all files) |
| `GenTauPt` | Total 4-momentum — pT |
| `GenTauP` | Total 4-momentum — \|p\| |
| `GenTauEta` | Total 4-momentum — η |
| `GenTauTheta` | Total 4-momentum — θ |
| `GenTauPhi` | Total 4-momentum — φ |
| `GenTauMass` | Total 4-momentum — invariant mass |
| `GenVisTauPt` | Visible 4-momentum — pT |
| `GenVisTauP` | Visible 4-momentum — \|p\| |
| `GenVisTauEta` | Visible 4-momentum — η |
| `GenVisTauTheta` | Visible 4-momentum — θ |
| `GenVisTauPhi` | Visible 4-momentum — φ |
| `GenVisTauMass` | Visible 4-momentum — invariant mass |
| `GenTauQ` | Electric charge |
| `GenTauType` | Decay mode ID (see table below) |
| `GenTauDR` | Max angular distance between visible daughters |
| `GenTauNConsts` | Number of final-state visible constituents |
| `GenTauNConstKey` | Tau index (same as position in this array — indexing key) |
| `GenMatchedKey` | Gen tau index that was matched to a reco tau (`RecoMatchedKey`) |

### Gen tau constituent branches (one entry per constituent, across all taus)

| Branch | Description |
|--------|-------------|
| `GenTauConstKey` | Index of the parent tau in the gen tau arrays |
| `GenConstPDG` | PDG code of the constituent |
| `GenConstP` | Constituent \|p\| |
| `GenConstTheta` | Constituent θ |
| `GenConstEta` | Constituent η |
| `GenConstPi0Key` | π⁰ ancestor index within this tau's decay tree; `-1` if the constituent does not descend from a π⁰ (see §Pi0 association) |

---

## Reconstructed taus / electrons / muons

Electrons and muons are reconstructed independently and then **merged into the same array** as hadronic taus. They are distinguished by `RecoTauType` (see decay ID table).

Each entry corresponds to one reconstructed object (up to `numRecoTaus` entries).

| Branch | Description |
|--------|-------------|
| `RecoTauPt` | 4-momentum — pT |
| `RecoTauP` | 4-momentum — \|p\| |
| `RecoTauEta` | 4-momentum — η |
| `RecoTauTheta` | 4-momentum — θ |
| `RecoTauPhi` | 4-momentum — φ |
| `RecoTauMass` | 4-momentum — invariant mass |
| `RecoTauQ` | Electric charge |
| `RecoTauType` | Raw decay ID (see table below); `-11` = electron, `-13` = muon |
| `RecoTauDM` | Compressed decay mode (see §Decay ID scheme) |
| `RecoTauDR` | Max angular distance between constituents (cone size) |
| `RecoTauNConsts` | Number of constituents |
| `RecoTauNConstKey` | Reco tau index (indexing key) |
| `RecoMatchedKey` | Index of the matched gen tau in the gen tau arrays; `-1` if unmatched |

### Reco tau constituent branches (one entry per constituent, across all reco taus)

| Branch | Description |
|--------|-------------|
| `RecoTauConstKey` | Index of the parent reco tau |
| `RecoConstPDG` | PDG code |
| `RecoConstP` | \|p\| |
| `RecoConstTheta` | θ |
| `RecoConstEta` | η |

---

## Generator-level photons

All MCParticles with PDG 22 and `generatorStatus == 1`, stored event-wide.

| Branch | Description |
|--------|-------------|
| `GenPhotonP` | \|p\| |
| `GenPhotonPt` | pT |
| `GenPhotonEta` | η |
| `GenPhotonTheta` | θ |
| `GenPhotonPhi` | φ |
| `GenPhotonMCIdx` | MCParticle object index (internal PODIO index) |
| `GenPhotonTauKey` | Index into the gen tau arrays of the parent tau; `-1` if the photon does not come from a tau |

---

## Reconstructed photons

PandoraPFOs with PDG 22, stored event-wide (always from PandoraPFOs, even when MLPF is used for tau reconstruction).

| Branch | Description |
|--------|-------------|
| `RecoPhotonP` | \|p\| |
| `RecoPhotonPt` | pT |
| `RecoPhotonEta` | η |
| `RecoPhotonTheta` | θ |
| `RecoPhotonPhi` | φ |
| `RecoPhotonPFOIdx` | PFO object index |
| `RecoPhotonTauKey` | Index into the reco tau arrays of the parent tau; `-1` if not part of any reconstructed tau (see §Reco photon–tau association) |
| `RecoPhotonGenMatchIdx` | Position in the `GenPhoton*` arrays of the matched gen photon; `-1` if no gen match |

---

## Association schemes

### Gen–reco tau matching

Matching is done via ΔR (angular distance) between each gen tau and all reco taus, using the cut `dRMatch` from config. For each gen tau `i`, `MatchRecoGenTau` returns the index `j` of the best-matching reco tau.

The result is stored as a **parallel pair of vectors**:

```
GenMatchedKey[i]  = i        # always the gen tau's own position
RecoMatchedKey[i] = j        # index into reco tau arrays; -1 if no match found
```

To find which reco tau corresponds to gen tau `i`:
```python
reco_idx = RecoMatchedKey[i]   # -1 means unmatched
```

To find which gen tau corresponds to reco tau `j`, scan `RecoMatchedKey` for the value `j`.

---

### Gen photon–tau association (`GenPhotonTauKey`)

Before filling photon branches, the code builds a map from MCParticle object index → gen tau array index, by walking the **final-state constituent list** of each gen tau and keeping only those with PDG 22.

```
GenPhotonTauKey[k] = i   →  GenPhoton k comes from gen tau i
GenPhotonTauKey[k] = -1  →  GenPhoton k is not a tau decay product
```

---

### Reco photon–tau association (`RecoPhotonTauKey`)

Reco photon–tau membership is matched by **momentum signature**: for each reco tau, the rounded 3-momentum `(px, py, pz)` of its photon constituents is stored in a lookup dict. Each PandoraPFO photon is looked up against this dict.

```
RecoPhotonTauKey[k] = j   →  RecoPFO photon k is a constituent of reco tau j
RecoPhotonTauKey[k] = -1  →  not matched to any reco tau
```

> **Note:** When MLPF is used for tau reconstruction, reco photons still come from PandoraPFOs. Since MLPF and PFO constituents differ, `RecoPhotonTauKey` will be `-1` for all photons in that case — this is correct by design.

---

### Reco photon–gen photon matching (`RecoPhotonGenMatchIdx`)

Uses the `RecoMCTruthLink` collection (best-weight, dedup by reco object) to map each PFO object index to an MCParticle object index. That MCParticle index is then looked up in the gen photon list to get a position in the `GenPhoton*` arrays.

```
RecoPhotonGenMatchIdx[k] = m  →  RecoPFO photon k matches GenPhoton[m]
RecoPhotonGenMatchIdx[k] = -1 →  no gen match
```

---

### Photon–π⁰ association (`GenConstPi0Key`)

Gen tau constituents are produced by recursively traversing the decay tree starting from the tau's direct daughters. Each π⁰ encountered during the traversal is assigned a sequential local index (0, 1, 2, …) within that tau. All photons that descend from that π⁰ are tagged with its index.

```
GenConstPi0Key = n   →  this constituent descends from the n-th π⁰ in this tau's decay
GenConstPi0Key = -1  →  not a π⁰ descendant (e.g. a charged pion)
```

To group the two photons of a π⁰: select all constituents of tau `i` (via `GenTauConstKey == i`) where `GenConstPi0Key == n`.

---

## Decay ID scheme

### Gen tau — `GenTauType`

Assigned by examining the direct daughters of the generator-level tau:

| ID | Decay |
|----|-------|
| `-13` | Leptonic → μ |
| `-11` | Leptonic → e |
| `-2` | Other / unexpected charged particle |
| `-1` | Not assigned |
| `0` | 1π⁺ + 0π⁰ |
| `1` | 1π⁺ + 1π⁰ |
| `2` | 1π⁺ + 2π⁰ |
| `n` | 1π⁺ + nπ⁰ |
| `10` | 3π + 0π⁰ |
| `11` | 3π + 1π⁰ |
| `12` | 3π + 2π⁰ |
| `10+n` | 3π + nπ⁰ |

### Reco tau — `RecoTauType`

Assigned by counting **photons** (not π⁰s) and charged pions in the reconstructed cluster. The scheme mirrors the gen ID but counts individual photons instead of π⁰s.

| ID | Reco content |
|----|--------------|
| `-13` | Muon (from `muonReco`) |
| `-11` | Electron (from `electronReco`) |
| `0` | 1π⁺ + 0 photons |
| `1` | 1π⁺ + 1 photon |
| `2` | 1π⁺ + 2 photons |
| `n` | 1π⁺ + n photons (capped at 9) |
| `10` | 3π + 0 photons |
| `11` | 3π + 1 photon |
| `10+n` | 3π + n photons |

### Reco tau — `RecoTauDM` (compressed decay mode)

`RecoTauDM` groups neighboring photon counts using `math.ceil(ID / 2)` for the 1-pion topology and `10 + math.ceil((ID−10) / 2)` for the 3-pion topology. This maps pairs of photon counts (which correspond to one π⁰) onto the same DM value:

| `RecoTauType` (1π) | `RecoTauDM` | Physical interpretation |
|--------------------|-------------|------------------------|
| 0 | 0 | 1π, 0γ |
| 1, 2 | 1 | 1π, ~1π⁰ (1–2 γ) |
| 3, 4 | 2 | 1π, ~2π⁰ (3–4 γ) |
| 5, 6 | 3 | 1π, ~3π⁰ |
| 7, 8 | 4 | 1π, ~4π⁰ |
| 9 | 5 | 1π, ≥5π⁰ |

| `RecoTauType` (3π) | `RecoTauDM` | Physical interpretation |
|--------------------|-------------|------------------------|
| 10 | 10 | 3π, 0γ |
| 11, 12 | 11 | 3π, ~1π⁰ |
| 13, 14 | 12 | 3π, ~2π⁰ |
| … | … | … |
