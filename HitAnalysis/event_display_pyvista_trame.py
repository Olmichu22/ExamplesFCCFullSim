#!/usr/bin/env python3
"""Visualizador interactivo de eventos EDM4HEP con PyVista + Trame.

Características principales:
- Apertura de uno o varios ficheros .root.
- Selector de evento global.
- Resumen de partículas generadas finales (filtrado por generatorStatus).
- Resumen de PFOs reconstruidas.
- Asociación Gen ↔ PFO por dR.
- Vista 3D de hits del detector con modo de color por asociación a Gen o a PFO.
- Visualización opcional de vectores de momento para tracks asociados a PFOs.

Dependencias: podio, ROOT, numpy, pyvista, trame.
"""

from __future__ import annotations

import argparse
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

try:
    from podio import root_io
except ImportError as exc:
    raise SystemExit("Error: podio no está instalado. Ejecuta en entorno Key4HEP.") from exc

try:
    import ROOT
except ImportError as exc:
    raise SystemExit("Error: ROOT no está disponible.") from exc

try:
    import pyvista as pv
    from pyvista.trame.ui import plotter_ui
except ImportError as exc:
    raise SystemExit("Error: pyvista o pyvista.trame no está disponible.") from exc

try:
    from trame.app import get_server
    from trame.ui.vuetify import SinglePageLayout
    from trame.widgets import html, vuetify
except ImportError as exc:
    raise SystemExit("Error: trame no está instalado.") from exc

from modules.myutils import dRAngle
from modules.NeutralRecover import omega_to_pt

VALID_GEN_STATUS = {1}
NEUTRINO_PDGS = {12, 14, 16}

DETECTOR_TYPES = {
    "INNER_TRACKER": 0,
    "ECAL": 1,
    "HCAL": 2,
    "MUON_TRACKER": 3,
}

DETECTOR_LABELS = {
    DETECTOR_TYPES["INNER_TRACKER"]: "Tracker",
    DETECTOR_TYPES["ECAL"]: "ECAL",
    DETECTOR_TYPES["HCAL"]: "HCAL",
    DETECTOR_TYPES["MUON_TRACKER"]: "MUON",
}

DETECTOR_COLORS = {
    DETECTOR_TYPES["INNER_TRACKER"]: "#9b59b6",
    DETECTOR_TYPES["ECAL"]: "#f39c12",
    DETECTOR_TYPES["HCAL"]: "#2ecc71",
    DETECTOR_TYPES["MUON_TRACKER"]: "#3498db",
}

ECAL_COLLECTIONS = [
    ("ECALBarrel", DETECTOR_TYPES["ECAL"]),
    ("ECALEndcap", DETECTOR_TYPES["ECAL"]),
    ("ECALOther", DETECTOR_TYPES["ECAL"]),
]
HCAL_COLLECTIONS = [
    ("HCALBarrel", DETECTOR_TYPES["HCAL"]),
    ("HCALEndcap", DETECTOR_TYPES["HCAL"]),
    ("HCALOther", DETECTOR_TYPES["HCAL"]),
]
MUON_COLLECTIONS = [("MUON", DETECTOR_TYPES["MUON_TRACKER"])]
TRACK_COLLECTION = "SiTracks_Refitted"
ALL_CALO_COLLECTIONS = ECAL_COLLECTIONS + HCAL_COLLECTIONS + MUON_COLLECTIONS


@dataclass
class HitRecord:
    key: Tuple[int, int]
    detector_type: int
    position: Tuple[float, float, float]
    energy: float
    gen_idx: Optional[int]
    pfo_idx: Optional[int]


@dataclass
class TrackVector:
    pfo_idx: int
    origin: Tuple[float, float, float]
    momentum: Tuple[float, float, float]


@dataclass
class GenRecord:
    idx: int
    pid: int
    status: int
    energy: float
    p: float
    theta: float
    phi: float


@dataclass
class PFORecord:
    idx: int
    pid: int
    energy: float
    charge: float
    p: float
    theta: float
    phi: float


@dataclass
class EventRecord:
    global_event: int
    file_path: str
    local_event: int
    gen_particles: List[GenRecord]
    pfos: List[PFORecord]
    dr_associations: List[Dict[str, float]]
    hits: List[HitRecord]
    track_vectors: List[TrackVector]


def _momentum_to_angles(px: float, py: float, pz: float) -> Tuple[float, float, float]:
    p = math.sqrt(px * px + py * py + pz * pz)
    if p < 1e-12:
        return 0.0, 0.0, 0.0
    theta = math.acos(max(-1.0, min(1.0, pz / p)))
    phi = math.atan2(py, px)
    return p, theta, phi


def build_hit_type_map(event) -> Tuple[Dict[Tuple[int, int], int], Dict[Tuple[int, int], float], Dict[Tuple[int, int], Tuple[float, float, float]]]:
    hit_type_map = {}
    hit_energy_map = {}
    hit_pos_map = {}

    for coll_name, detector_type in ALL_CALO_COLLECTIONS:
        try:
            coll = event.get(coll_name)
        except Exception:
            continue

        for hit in coll:
            obj_id = hit.getObjectID()
            key = (obj_id.collectionID, obj_id.index)
            hit_type_map[key] = detector_type
            hit_energy_map[key] = float(hit.getEnergy())
            pos = hit.getPosition()
            hit_pos_map[key] = (float(pos.x), float(pos.y), float(pos.z))

    return hit_type_map, hit_energy_map, hit_pos_map


def build_hit_to_gen_map(event, hit_type_map: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    hit_to_gen = {}
    try:
        links = list(event.get("CalohitMCTruthLink"))
    except Exception:
        return hit_to_gen

    for link in links:
        try:
            hit_obj = link.getRec()
            gen_obj = link.getSim()
            key = (hit_obj.getObjectID().collectionID, hit_obj.getObjectID().index)
            if key in hit_type_map:
                hit_to_gen[key] = int(gen_obj.getObjectID().index)
        except Exception:
            continue

    return hit_to_gen


def build_hit_to_pfo_map(event, hit_type_map: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    hit_to_pfo = {}
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        return hit_to_pfo

    for pfo in pfos:
        pfo_idx = int(pfo.getObjectID().index)
        for cluster in pfo.getClusters():
            for hit in cluster.getHits():
                key = (hit.getObjectID().collectionID, hit.getObjectID().index)
                if key in hit_type_map:
                    hit_to_pfo[key] = pfo_idx

    return hit_to_pfo


def extract_gen_particles(event, generator_status: int) -> List[GenRecord]:
    out: List[GenRecord] = []
    try:
        mc_particles = event.get("MCParticles")
    except Exception:
        return out

    for part in mc_particles:
        try:
            status = int(part.getGeneratorStatus())
            if status != generator_status:
                continue
            pid = int(part.getPDG())
            if abs(pid) in NEUTRINO_PDGS:
                continue
            mom = part.getMomentum()
            p, theta, phi = _momentum_to_angles(float(mom.x), float(mom.y), float(mom.z))
            if p <= 0:
                continue
            out.append(
                GenRecord(
                    idx=int(part.getObjectID().index),
                    pid=pid,
                    status=status,
                    energy=float(part.getEnergy()),
                    p=p,
                    theta=theta,
                    phi=phi,
                )
            )
        except Exception:
            continue

    return out


def extract_pfos(event) -> List[PFORecord]:
    out: List[PFORecord] = []
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        return out

    for pfo in pfos:
        try:
            mom = pfo.getMomentum()
            p, theta, phi = _momentum_to_angles(float(mom.x), float(mom.y), float(mom.z))
            out.append(
                PFORecord(
                    idx=int(pfo.getObjectID().index),
                    pid=int(pfo.getPDG()),
                    energy=float(pfo.getEnergy()),
                    charge=float(pfo.getCharge()),
                    p=p,
                    theta=theta,
                    phi=phi,
                )
            )
        except Exception:
            continue

    return out


def associate_by_dr(gen_particles: Sequence[GenRecord], pfos: Sequence[PFORecord], max_dr: float) -> List[Dict[str, float]]:
    associations: List[Dict[str, float]] = []
    for gen in gen_particles:
        gen_p4 = ROOT.TLorentzVector()
        pt = gen.p * math.sin(gen.theta)
        gen_px = pt * math.cos(gen.phi)
        gen_py = pt * math.sin(gen.phi)
        gen_pz = gen.p * math.cos(gen.theta)
        gen_p4.SetXYZT(gen_px, gen_py, gen_pz, gen.energy)

        best = None
        for pfo in pfos:
            pfo_p4 = ROOT.TLorentzVector()
            reco_pt = pfo.p * math.sin(pfo.theta)
            reco_px = reco_pt * math.cos(pfo.phi)
            reco_py = reco_pt * math.sin(pfo.phi)
            reco_pz = pfo.p * math.cos(pfo.theta)
            pfo_p4.SetXYZM(reco_px, reco_py, reco_pz, 0.0)
            dr = float(dRAngle(gen_p4, pfo_p4))
            if best is None or dr < best["dR"]:
                best = {"gen_idx": gen.idx, "gen_pid": gen.pid, "pfo_idx": pfo.idx, "pfo_pid": pfo.pid, "dR": dr}

        if best is None or best["dR"] > max_dr:
            associations.append({"gen_idx": gen.idx, "gen_pid": gen.pid, "pfo_idx": -999, "pfo_pid": -999, "dR": float("nan")})
        else:
            associations.append(best)

    return associations


def extract_track_vectors(event) -> List[TrackVector]:
    vectors: List[TrackVector] = []
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        return vectors

    for pfo in pfos:
        pfo_idx = int(pfo.getObjectID().index)
        for track in pfo.getTracks():
            try:
                states = track.getTrackStates()
                if len(states) == 0:
                    continue
                st = states[0]
                pt = omega_to_pt(float(st.omega), isclic=False)
                px = pt * math.cos(float(st.phi))
                py = pt * math.sin(float(st.phi))
                pz = float(st.tanLambda) * pt
                vectors.append(TrackVector(pfo_idx=pfo_idx, origin=(0.0, 0.0, 0.0), momentum=(px, py, pz)))
            except Exception:
                continue

    return vectors


def extract_hits(event) -> List[HitRecord]:
    hit_type_map, hit_energy_map, hit_pos_map = build_hit_type_map(event)
    hit_to_gen = build_hit_to_gen_map(event, hit_type_map)
    hit_to_pfo = build_hit_to_pfo_map(event, hit_type_map)

    records: List[HitRecord] = []
    for key, detector_type in hit_type_map.items():
        records.append(
            HitRecord(
                key=key,
                detector_type=detector_type,
                position=hit_pos_map[key],
                energy=hit_energy_map.get(key, 0.0),
                gen_idx=hit_to_gen.get(key),
                pfo_idx=hit_to_pfo.get(key),
            )
        )
    return records


def load_events(files: Sequence[str], max_events_per_file: Optional[int], generator_status: int, dr_max: float) -> List[EventRecord]:
    all_events: List[EventRecord] = []
    global_event = 0

    for path in files:
        reader = root_io.Reader(path)
        events = reader.get("events")
        for local_event, event in enumerate(events):
            if max_events_per_file is not None and local_event >= max_events_per_file:
                break

            gen_particles = extract_gen_particles(event, generator_status)
            pfos = extract_pfos(event)
            dr_associations = associate_by_dr(gen_particles, pfos, dr_max)
            hits = extract_hits(event)
            track_vectors = extract_track_vectors(event)

            all_events.append(
                EventRecord(
                    global_event=global_event,
                    file_path=path,
                    local_event=local_event,
                    gen_particles=gen_particles,
                    pfos=pfos,
                    dr_associations=dr_associations,
                    hits=hits,
                    track_vectors=track_vectors,
                )
            )
            global_event += 1

    return all_events


def as_text_table_gen(gen: Sequence[GenRecord]) -> str:
    if not gen:
        return "Sin partículas gen (con filtros aplicados)."
    header = "idx pid status      E         p      theta     phi"
    rows = [header]
    for g in gen:
        rows.append(f"{g.idx:>3} {g.pid:>5} {g.status:>6} {g.energy:>8.3f} {g.p:>8.3f} {g.theta:>8.3f} {g.phi:>8.3f}")
    return "\n".join(rows)


def as_text_table_pfo(pfos: Sequence[PFORecord]) -> str:
    if not pfos:
        return "Sin PFOs reconstruidas."
    header = "idx pid charge      E         p      theta     phi"
    rows = [header]
    for p in pfos:
        rows.append(f"{p.idx:>3} {p.pid:>5} {p.charge:>6.1f} {p.energy:>8.3f} {p.p:>8.3f} {p.theta:>8.3f} {p.phi:>8.3f}")
    return "\n".join(rows)


def as_text_table_dr(assoc: Sequence[Dict[str, float]]) -> str:
    if not assoc:
        return "Sin asociaciones dR."
    header = "gen_idx gen_pid -> pfo_idx pfo_pid      dR"
    rows = [header]
    for a in assoc:
        dr = a["dR"]
        dr_txt = "nan" if math.isnan(dr) else f"{dr:.4f}"
        rows.append(f"{a['gen_idx']:>7} {a['gen_pid']:>7} -> {a['pfo_idx']:>7} {a['pfo_pid']:>7} {dr_txt:>7}")
    return "\n".join(rows)


def build_plot(plotter: pv.Plotter, event: EventRecord, association_mode: str, show_unassociated: bool, show_vectors: bool, momentum_scale: float) -> None:
    plotter.clear()
    plotter.set_background("#0f1116")

    grouped = defaultdict(list)
    for hit in event.hits:
        owner = hit.gen_idx if association_mode == "gen" else hit.pfo_idx
        if owner is None and not show_unassociated:
            continue
        grouped[(hit.detector_type, owner)].append(hit)

    for (detector_type, owner), hits in grouped.items():
        points = np.array([h.position for h in hits], dtype=float)
        color = DETECTOR_COLORS[detector_type] if owner is None else None
        label_owner = f"unassoc" if owner is None else f"{association_mode}:{owner}"
        label = f"{DETECTOR_LABELS[detector_type]} | {label_owner} | N={len(hits)}"

        cloud = pv.PolyData(points)
        if owner is not None:
            cloud["owner"] = np.full(points.shape[0], int(owner))
            plotter.add_mesh(
                cloud,
                scalars="owner",
                point_size=8,
                render_points_as_spheres=True,
                cmap="tab20",
                name=label,
                label=label,
            )
        else:
            plotter.add_mesh(
                cloud,
                color=color,
                point_size=6,
                render_points_as_spheres=False,
                style="points",
                opacity=0.6,
                name=label,
                label=label,
            )

    if show_vectors and event.track_vectors:
        origins = np.array([v.origin for v in event.track_vectors], dtype=float)
        momenta = np.array([v.momentum for v in event.track_vectors], dtype=float)
        pdata = pv.PolyData(origins)
        pdata["vectors"] = momenta
        pdata["mag"] = np.linalg.norm(momenta, axis=1)
        glyphs = pdata.glyph(orient="vectors", scale="mag", factor=momentum_scale)
        plotter.add_mesh(glyphs, color="white", name="track_vectors", label="Track vectors")

    plotter.add_axes()
    plotter.show_grid(color="gray")
    plotter.add_legend(size=(0.25, 0.25), face="triangle")
    plotter.reset_camera()


def run_app(records: List[EventRecord], port: int, host: str, momentum_scale: float) -> None:
    if not records:
        raise SystemExit("No se han cargado eventos válidos.")

    server = get_server(client_type="vue2")
    state, ctrl = server.state, server.controller

    state.event_index = 0
    state.association_mode = "gen"
    state.show_unassociated = True
    state.show_track_vectors = True

    plotter = pv.Plotter()
    view = plotter_ui(plotter, add_menu=False)
    ctrl.view_update = view.update

    def _refresh_ui() -> None:
        idx = int(state.event_index)
        idx = max(0, min(idx, len(records) - 1))
        event = records[idx]
        build_plot(
            plotter,
            event=event,
            association_mode=state.association_mode,
            show_unassociated=bool(state.show_unassociated),
            show_vectors=bool(state.show_track_vectors),
            momentum_scale=momentum_scale,
        )
        state.event_meta = f"Evento global {event.global_event} | fichero: {event.file_path} | evento local: {event.local_event}"
        state.gen_summary = as_text_table_gen(event.gen_particles)
        state.pfo_summary = as_text_table_pfo(event.pfos)
        state.dr_summary = as_text_table_dr(event.dr_associations)
        ctrl.view_update()

    @state.change("event_index", "association_mode", "show_unassociated", "show_track_vectors")
    def _on_state_change(**_):
        _refresh_ui()

    with SinglePageLayout(server) as layout:
        layout.title.set_text("Event Debug Viewer (PyVista + Trame)")
        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VSelect(
                label="Evento",
                v_model=("event_index", 0),
                items=("event_items", [{"text": f"{r.global_event} ({r.file_path.split('/')[-1]}:{r.local_event})", "value": r.global_event} for r in records]),
                dense=True,
                hide_details=True,
                style="max-width: 360px",
            )
            vuetify.VSelect(
                label="Asociación de hits",
                v_model=("association_mode", "gen"),
                items=("assoc_modes", [{"text": "Gen", "value": "gen"}, {"text": "PFO", "value": "pfo"}]),
                dense=True,
                hide_details=True,
                style="max-width: 180px",
            )
            vuetify.VCheckbox(label="Mostrar no asociados", v_model=("show_unassociated", True), dense=True, hide_details=True)
            vuetify.VCheckbox(label="Mostrar vectores track", v_model=("show_track_vectors", True), dense=True, hide_details=True)

        with layout.content:
            with vuetify.VContainer(fluid=True):
                with vuetify.VRow():
                    with vuetify.VCol(cols=12, md=8):
                        html.Div("{{ event_meta }}", style="font-weight: 600; margin-bottom: 8px;")
                        html.Div(view, style="height: 72vh;")
                    with vuetify.VCol(cols=12, md=4):
                        vuetify.VCardTitle("Resumen Gen")
                        html.Pre("{{ gen_summary }}", style="max-height: 20vh; overflow: auto; font-size: 12px;")
                        vuetify.VCardTitle("Resumen PFO")
                        html.Pre("{{ pfo_summary }}", style="max-height: 20vh; overflow: auto; font-size: 12px;")
                        vuetify.VCardTitle("Asociación dR")
                        html.Pre("{{ dr_summary }}", style="max-height: 22vh; overflow: auto; font-size: 12px;")

    _refresh_ui()
    server.start(port=port, host=host)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualizador interactivo de eventos con PyVista + Trame")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Ficheros ROOT de entrada")
    parser.add_argument("--max-events", type=int, default=None, help="Máximo de eventos por fichero")
    parser.add_argument("--generator-status", type=int, default=1, help="GeneratorStatus a mostrar en resumen gen")
    parser.add_argument("--dr-max", type=float, default=0.1, help="Corte de asociación Gen-PFO por dR")
    parser.add_argument("--host", default="0.0.0.0", help="Host para servidor Trame")
    parser.add_argument("--port", type=int, default=8080, help="Puerto para servidor Trame")
    parser.add_argument("--momentum-scale", type=float, default=60.0, help="Escala visual para longitud de vectores de track")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_events(
        files=args.input,
        max_events_per_file=args.max_events,
        generator_status=args.generator_status,
        dr_max=args.dr_max,
    )
    run_app(records=records, port=args.port, host=args.host, momentum_scale=args.momentum_scale)


if __name__ == "__main__":
    main()
