#!/usr/bin/env python3
"""Visualizador PyVista + Trame independiente de Key4HEP."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import pyvista as pv
from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import html, vuetify

DETECTOR_LABELS = {
    0: "Tracker",
    1: "ECAL",
    2: "HCAL",
    3: "MUON",
}

DETECTOR_STYLE = {
    0: {"point_size": 5, "spheres": True, "opacity": 1.0},
    1: {"point_size": 7, "spheres": True, "opacity": 1.0},
    2: {"point_size": 9, "spheres": False, "opacity": 0.9},
    3: {"point_size": 11, "spheres": False, "opacity": 0.8},
}

UNASSOC_COLORS = {
    0: "#b38bd4",
    1: "#ffbf70",
    2: "#7fe7a6",
    3: "#7fb8ff",
}


def load_events(path: str) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        payload = json.load(f)

    if isinstance(payload, list):
        events = payload
    elif isinstance(payload, dict):
        events = payload.get("events", [])
    else:
        events = []

    if not isinstance(events, list):
        raise RuntimeError("Formato JSON no válido: 'events' debe ser una lista")

    return events


def _fmt_or_none(value: Any, fmt: str = ".3f") -> str:
    if value is None:
        return "None"
    try:
        return format(float(value), fmt)
    except Exception:
        return str(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    if isinstance(value, str):
        low = value.strip().lower()
        if low in {"1", "true", "yes", "y", "t"}:
            return True
        if low in {"0", "false", "no", "n", "f", "", "none", "null"}:
            return False
    return bool(value)


def _coerce_float_or_none(value: Any) -> Any:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _normalize_track_vector(v: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(v, dict):
        return {"origin": [0.0, 0.0, 0.0], "momentum": [0.0, 0.0, 0.0], "associated": False, "charge": None}

    pfo_idx = v.get("pfo_idx", v.get("pfoIndex", v.get("pfo")))
    raw_assoc = v.get("associated", v.get("assoc", v.get("is_associated")))
    assoc = (pfo_idx is not None) if raw_assoc is None else _coerce_bool(raw_assoc)
    charge = _coerce_float_or_none(v.get("charge", v.get("q")))

    out = dict(v)
    out["associated"] = bool(assoc)
    out["charge"] = charge
    return out


def normalize_track_vectors(vectors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [_normalize_track_vector(v) for v in (vectors or [])]


def track_items(vectors: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convierte track_vectors en filas para una VDataTable."""
    items: List[Dict[str, Any]] = []
    for i, v in enumerate(normalize_track_vectors(vectors)):
        o = v.get("origin", [None, None, None])
        p = v.get("momentum", [None, None, None])
        assoc = bool(v.get("associated", False))
        charge = v.get("charge", None)
        # magnitud de p
        try:
            mag = float(np.linalg.norm(np.array(p, dtype=float)))
        except Exception:
            mag = None

        items.append(
            {
                "i": i,
                "associated": assoc,
                "ox": _fmt_or_none(o[0]),
                "oy": _fmt_or_none(o[1]),
                "oz": _fmt_or_none(o[2]),
                "px": _fmt_or_none(p[0]),
                "py": _fmt_or_none(p[1]),
                "pz": _fmt_or_none(p[2]),
                "p": _fmt_or_none(mag),
                "charge": _fmt_or_none(charge, ".1f"),
            }
        )
    return items

def as_text_table_gen(gen: Sequence[Dict[str, Any]]) -> str:
    if not gen:
        return "Sin partículas gen (con filtros aplicados)."

    rows = ["idx pid status      E         p      theta     phi"]
    for g in gen:
        rows.append(
            f"{int(g.get('idx', -1)):>3} "
            f"{int(g.get('pid', 0)):>5} "
            f"{int(g.get('status', 0)):>6} "
            f"{_fmt_or_none(g.get('energy')):>8} "
            f"{_fmt_or_none(g.get('p')):>8} "
            f"{_fmt_or_none(g.get('theta')):>8} "
            f"{_fmt_or_none(g.get('phi')):>8}"
        )
    return "\n".join(rows)


def as_text_table_pfo(pfos: Sequence[Dict[str, Any]]) -> str:
    if not pfos:
        return "Sin PFOs reconstruidas."

    rows = ["idx pid charge      E         p      theta     phi"]
    for p in pfos:
        rows.append(
            f"{int(p.get('idx', -1)):>3} "
            f"{int(p.get('pid', 0)):>5} "
            f"{_fmt_or_none(p.get('charge'), '.1f'):>6} "
            f"{_fmt_or_none(p.get('energy')):>8} "
            f"{_fmt_or_none(p.get('p')):>8} "
            f"{_fmt_or_none(p.get('theta')):>8} "
            f"{_fmt_or_none(p.get('phi')):>8}"
        )
    return "\n".join(rows)


def as_text_table_dr(assoc: Sequence[Dict[str, Any]]) -> str:
    if not assoc:
        return "Sin asociaciones dR."

    rows = ["gen_idx gen_pid -> pfo_idx pfo_pid      dR"]
    for a in assoc:
        dr = a.get("dR")
        dr_txt = "None" if dr is None else f"{float(dr):.4f}"
        pfo_idx = a.get("pfo_idx")
        pfo_pid = a.get("pfo_pid")
        rows.append(
            f"{int(a.get('gen_idx', -1)):>7} "
            f"{int(a.get('gen_pid', 0)):>7} -> "
            f"{str(pfo_idx if pfo_idx is not None else 'None'):>7} "
            f"{str(pfo_pid if pfo_pid is not None else 'None'):>7} "
            f"{dr_txt:>7}"
        )
    return "\n".join(rows)


def _coerce_event_index(raw_value: Any, num_events: int) -> int:
    try:
        idx = int(raw_value)
    except Exception:
        idx = 0
    if num_events <= 0:
        return 0
    return max(0, min(idx, num_events - 1))


def build_plot(
    plotter: pv.Plotter,
    event: Dict[str, Any],
    association_mode: str,
    show_unassociated: bool,
    show_vectors: bool,
    momentum_scale: float,
) -> None:
    plotter.clear()
    plotter.set_background("#0f1116")

    hits = event.get("hits", [])
    grouped: Dict[tuple, List[Dict[str, Any]]] = defaultdict(list)
    detector_counts: Dict[int, int] = defaultdict(int)
    valid_gen_ids = {int(g.get("idx")) for g in event.get("gen_particles", []) if g.get("idx") is not None}

    for hit in hits:
        detector_type = int(hit.get("detector_type", 1))
        detector_counts[detector_type] += 1
        owner = hit.get("gen_idx") if association_mode == "gen" else hit.get("pfo_idx")
        if association_mode == "gen" and owner is not None and int(owner) not in valid_gen_ids:
            owner = None
        if owner is None and not show_unassociated:
            continue
        grouped[(detector_type, owner)].append(hit)

    associated_owners = sorted({int(owner) for (_, owner) in grouped.keys() if owner is not None})
    if associated_owners:
        owner_min = float(associated_owners[0])
        owner_max = float(associated_owners[-1])
        if owner_min == owner_max:
            owner_min -= 0.5
            owner_max += 0.5
        owner_clim = [owner_min, owner_max]
    else:
        owner_clim = [0.0, 1.0]

    scalar_bar_added = False
    for (detector_type, owner), bucket in grouped.items():
        points = np.array([[float(h["x"]), float(h["y"]), float(h["z"])] for h in bucket], dtype=float)
        if points.size == 0:
            continue

        cloud = pv.PolyData(points)
        style = DETECTOR_STYLE.get(detector_type, DETECTOR_STYLE[1])
        det_label = DETECTOR_LABELS.get(detector_type, f"DET{detector_type}")
        owner_label = "unassoc" if owner is None else f"{association_mode}:{owner}"
        label = f"{det_label} | {owner_label} | N={len(bucket)}"

        if owner is not None:
            cloud["owner"] = np.full(points.shape[0], int(owner))
            plotter.add_mesh(
                cloud,
                scalars="owner",
                cmap="tab20",
                point_size=style["point_size"],
                render_points_as_spheres=style["spheres"],
                opacity=style["opacity"],
                show_scalar_bar=not scalar_bar_added,
                clim=owner_clim,
                scalar_bar_args={
                    "title": f"ID {association_mode.upper()}",
                    "fmt": "%.0f",
                    "n_labels": 8,
                    "color": "white",
                },
                name=label,
                label=label,
            )
            scalar_bar_added = True
        else:
            plotter.add_mesh(
                cloud,
                color=UNASSOC_COLORS.get(detector_type, "white"),
                style="points",
                point_size=style["point_size"],
                render_points_as_spheres=style["spheres"],
                opacity=style["opacity"],
                name=label,
                label=label,
            )

    if show_vectors:
        vectors = normalize_track_vectors(event.get("track_vectors", []))

        if vectors:
            origins = np.array([v.get("origin", [0.0, 0.0, 0.0]) for v in vectors], dtype=float)
            momenta = np.array([v.get("momentum", [0.0, 0.0, 0.0]) for v in vectors], dtype=float)
            associated = np.array([v.get("associated", False) for v in vectors], dtype=int)

            if origins.size > 0 and momenta.size > 0:
                pdata = pv.PolyData(origins)
                pdata["vectors"] = momenta
                pdata["mag"] = np.linalg.norm(momenta, axis=1)
                pdata["associated"] = associated  # <-- añadimos el atributo

                glyphs = pdata.glyph(
                    orient="vectors",
                    scale="mag",
                    factor=momentum_scale
                )

                plotter.add_mesh(
                    glyphs,
                    scalars="associated",
                    cmap=["red", "lime"],   # 0 = red, 1 = green
                    clim=[0, 1],
                    name="track_vectors",
                    label="Track vectors",
                    show_scalar_bar=False
                )

    plotter.add_axes()
    plotter.show_grid(color="gray")
    plotter.add_legend(size=(0.32, 0.28), face="triangle", bcolor="#22252d")

    hit_summary_lines = ["Hits por detector:"]
    for det_type in (0, 1, 2, 3):
        label = DETECTOR_LABELS.get(det_type, f"DET{det_type}")
        count = detector_counts.get(det_type, 0)
        hit_summary_lines.append(f"{label}: {count}")
    plotter.add_text("\n".join(hit_summary_lines), position="upper_left", font_size=14, color="white")

    plotter.reset_camera()


def run_app(events: List[Dict[str, Any]], port: int, host: str, momentum_scale: float) -> None:
    if not events:
        raise SystemExit("No hay eventos para visualizar. Revisa el JSON de entrada.")

    server = get_server(client_type="vue2")
    state, ctrl = server.state, server.controller

    pv.OFF_SCREEN = True
    plotter = pv.Plotter(off_screen=True)
    view = plotter_ui(plotter, add_menu=False)
    ctrl.view_update = view.update
    state.track_items = []
    state.track_headers = [
        {"text": "#", "value": "i", "width": 40},
        {"text": "assoc", "value": "associated", "width": 70},
        {"text": "ox", "value": "ox"},
        {"text": "oy", "value": "oy"},
        {"text": "oz", "value": "oz"},
        {"text": "px", "value": "px"},
        {"text": "py", "value": "py"},
        {"text": "pz", "value": "pz"},
        {"text": "|p|", "value": "p"},
        {"text": "charge", "value": "charge", "width": 60},
    ]
    state.event_index = 0
    state.association_mode = "gen"
    state.show_unassociated = True
    state.show_track_vectors = True
    state.event_meta = ""
    state.gen_summary = ""
    state.pfo_summary = ""
    state.dr_summary = ""
    state.event_items = [
        {
            "text": f"{i}: {str(e.get('file_path', 'unknown')).split('/')[-1]}:{int(e.get('local_event', i))}",
            "value": i,
        }
        for i, e in enumerate(events)
    ]

    def refresh_ui() -> None:
        idx = _coerce_event_index(state.event_index, len(events))
        state.event_index = idx
        event = events[idx]

        build_plot(
            plotter=plotter,
            event=event,
            association_mode=str(state.association_mode),
            show_unassociated=bool(state.show_unassociated),
            show_vectors=bool(state.show_track_vectors),
            momentum_scale=momentum_scale,
        )

        total_hits = len(event.get("hits", []))
        unassoc_gen = sum(1 for h in event.get("hits", []) if h.get("gen_idx") is None)
        unassoc_pfo = sum(1 for h in event.get("hits", []) if h.get("pfo_idx") is None)
        state.track_items = track_items(event.get("track_vectors", []))
        state.event_meta = (
            f"Evento {idx} | file={event.get('file_path', 'n/a')} | "
            f"local={event.get('local_event', 'n/a')} | "
            f"hits={total_hits} | unassoc(gen)={unassoc_gen} | unassoc(pfo)={unassoc_pfo}"
        )
        state.gen_summary = as_text_table_gen(event.get("gen_particles", []))
        state.pfo_summary = as_text_table_pfo(event.get("pfos", []))
        state.dr_summary = as_text_table_dr(event.get("associations", []))

        ctrl.view_update()

    @state.change("event_index", "association_mode", "show_unassociated", "show_track_vectors")
    def _on_state_change(**_):
        refresh_ui()

    with SinglePageLayout(server) as layout:
        layout.title.set_text("Event Debug Viewer (JSON + PyVista + Trame)")

        with layout.toolbar:
            vuetify.VSpacer()
            vuetify.VSelect(
                label="Evento",
                v_model=("event_index", 0),
                items=("event_items", state.event_items),
                dense=True,
                hide_details=True,
                style="max-width: 360px",
            )
            vuetify.VSelect(
                label="Asociación de hits",
                v_model=("association_mode", "gen"),
                items=(
                    "assoc_modes",
                    [
                        {"text": "Gen", "value": "gen"},
                        {"text": "PFO", "value": "pfo"},
                    ],
                ),
                dense=True,
                hide_details=True,
                style="max-width: 200px",
            )
            vuetify.VCheckbox(
                label="Mostrar no asociados",
                v_model=("show_unassociated", True),
                dense=True,
                hide_details=True,
            )
            vuetify.VCheckbox(
                label="Mostrar vectores track",
                v_model=("show_track_vectors", True),
                dense=True,
                hide_details=True,
            )

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
                        vuetify.VCardTitle("Trazas (track_vectors)")
                        vuetify.VDataTable(
                            headers=("track_headers", state.track_headers),
                            items=("track_items", []),
                            dense=True,
                            disable_pagination=True,
                            hide_default_footer=True,
                            fixed_header=True,
                            height="22vh",
                            style="font-size: 12px;",
                        )
    refresh_ui()
    server.start(port=port, host=host, open_browser=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualizador interactivo JSON con PyVista + Trame")
    parser.add_argument("-i", "--input", required=True, help="Fichero JSON de entrada (generado con event_export_key4hep.py)")
    parser.add_argument("--host", default="0.0.0.0", help="Host del servidor Trame")
    parser.add_argument("--port", type=int, default=8080, help="Puerto del servidor Trame")
    parser.add_argument("--momentum-scale", type=float, default=60.0, help="Escala visual para longitud de vectores de track")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = load_events(args.input)
    run_app(events=events, port=args.port, host=args.host, momentum_scale=args.momentum_scale)


if __name__ == "__main__":
    main()
