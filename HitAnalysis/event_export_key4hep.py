#!/usr/bin/env python3
"""Exporta eventos EDM4HEP a JSON para visualización con PyVista + Trame."""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import ROOT
from podio import root_io

from modules.NeutralRecover import omega_to_pt
from modules.myutils import dRAngle

NEUTRINO_PDGS = {12, 14, 16}

DETECTOR_TYPES = {
    "INNER_TRACKER": 0,
    "ECAL": 1,
    "HCAL": 2,
    "MUON_TRACKER": 3,
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
ALL_CALO_COLLECTIONS = ECAL_COLLECTIONS + HCAL_COLLECTIONS + MUON_COLLECTIONS


def momentum_to_angles(px: float, py: float, pz: float) -> Tuple[float, float, float]:
    p = math.sqrt(px * px + py * py + pz * pz)
    if p < 1e-12:
        return 0.0, 0.0, 0.0
    theta = math.acos(max(-1.0, min(1.0, pz / p)))
    phi = math.atan2(py, px)
    return p, theta, phi


def build_hit_type_maps(event) -> Tuple[
    Dict[Tuple[int, int], int],
    Dict[Tuple[int, int], float],
    Dict[Tuple[int, int], Tuple[float, float, float]],
]:
    hit_type_map: Dict[Tuple[int, int], int] = {}
    hit_energy_map: Dict[Tuple[int, int], float] = {}
    hit_pos_map: Dict[Tuple[int, int], Tuple[float, float, float]] = {}

    for coll_name, detector_type in ALL_CALO_COLLECTIONS:
        try:
            coll = event.get(coll_name)
        except Exception:
            continue

        for hit in coll:
            try:
                obj_id = hit.getObjectID()
                key = (int(obj_id.collectionID), int(obj_id.index))
                hit_type_map[key] = detector_type
                hit_energy_map[key] = float(hit.getEnergy())
                pos = hit.getPosition()
                hit_pos_map[key] = (float(pos.x), float(pos.y), float(pos.z))
            except Exception:
                continue

    return hit_type_map, hit_energy_map, hit_pos_map


def build_hit_to_gen_map(event, hit_type_map: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    hit_to_gen: Dict[Tuple[int, int], int] = {}
    try:
        links = event.get("CalohitMCTruthLink")
    except Exception:
        return hit_to_gen

    for link in links:
        try:
            hit_obj = link.getRec()
            gen_obj = link.getSim()
            hit_id = hit_obj.getObjectID()
            key = (int(hit_id.collectionID), int(hit_id.index))
            if key in hit_type_map:
                hit_to_gen[key] = int(gen_obj.getObjectID().index)
        except Exception:
            continue

    return hit_to_gen


def build_hit_to_pfo_map(event, hit_type_map: Dict[Tuple[int, int], int]) -> Dict[Tuple[int, int], int]:
    hit_to_pfo: Dict[Tuple[int, int], int] = {}
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        return hit_to_pfo

    for pfo in pfos:
        try:
            pfo_idx = int(pfo.getObjectID().index)
            for cluster in pfo.getClusters():
                for hit in cluster.getHits():
                    hit_id = hit.getObjectID()
                    key = (int(hit_id.collectionID), int(hit_id.index))
                    if key in hit_type_map:
                        hit_to_pfo[key] = pfo_idx
        except Exception:
            continue

    return hit_to_pfo


def extract_gen_particles(event, generator_status: int) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
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
            p, theta, phi = momentum_to_angles(float(mom.x), float(mom.y), float(mom.z))
            out.append(
                {
                    "idx": int(part.getObjectID().index),
                    "pid": pid,
                    "status": status,
                    "energy": float(part.getEnergy()),
                    "p": p,
                    "theta": theta,
                    "phi": phi,
                }
            )
        except Exception:
            continue

    return out


def extract_pfos(event) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        return out

    for pfo in pfos:
        try:
            mom = pfo.getMomentum()
            p, theta, phi = momentum_to_angles(float(mom.x), float(mom.y), float(mom.z))
            out.append(
                {
                    "idx": int(pfo.getObjectID().index),
                    "pid": int(pfo.getPDG()),
                    "energy": float(pfo.getEnergy()),
                    "charge": float(pfo.getCharge()),
                    "p": p,
                    "theta": theta,
                    "phi": phi,
                }
            )
        except Exception:
            continue

    return out


def associate_by_dr(gen_particles: Sequence[Dict[str, Any]], pfos: Sequence[Dict[str, Any]], max_dr: float) -> List[Dict[str, Any]]:
    associations: List[Dict[str, Any]] = []

    for gen in gen_particles:
        gen_p4 = ROOT.TLorentzVector()
        gen_pt = float(gen["p"]) * math.sin(float(gen["theta"]))
        gen_px = gen_pt * math.cos(float(gen["phi"]))
        gen_py = gen_pt * math.sin(float(gen["phi"]))
        gen_pz = float(gen["p"]) * math.cos(float(gen["theta"]))
        gen_p4.SetXYZT(gen_px, gen_py, gen_pz, float(gen["energy"]))

        best: Optional[Dict[str, Any]] = None
        for pfo in pfos:
            pfo_p4 = ROOT.TLorentzVector()
            pfo_pt = float(pfo["p"]) * math.sin(float(pfo["theta"]))
            pfo_px = pfo_pt * math.cos(float(pfo["phi"]))
            pfo_py = pfo_pt * math.sin(float(pfo["phi"]))
            pfo_pz = float(pfo["p"]) * math.cos(float(pfo["theta"]))
            pfo_p4.SetXYZM(pfo_px, pfo_py, pfo_pz, 0.0)
            dr = float(dRAngle(gen_p4, pfo_p4))

            if best is None or dr < float(best["dR"]):
                best = {
                    "gen_idx": int(gen["idx"]),
                    "gen_pid": int(gen["pid"]),
                    "pfo_idx": int(pfo["idx"]),
                    "pfo_pid": int(pfo["pid"]),
                    "dR": dr,
                }

        if best is None or float(best["dR"]) > max_dr:
            associations.append(
                {
                    "gen_idx": int(gen["idx"]),
                    "gen_pid": int(gen["pid"]),
                    "pfo_idx": None,
                    "pfo_pid": None,
                    "dR": None,
                }
            )
        else:
            associations.append(best)

    return associations


def extract_track_vectors(event) -> List[Dict[str, Any]]:
    vectors: List[Dict[str, Any]] = []
    try:
        pfos = event.get("PandoraPFOs")
    except Exception:
        return vectors
    associated_tracks = set()
    for pfo in pfos:
        pfo_idx = int(pfo.getObjectID().index)
        for track in pfo.getTracks():
            try:
                track_id = track.getObjectID().index
                associated_tracks.add(track_id)
                states = track.getTrackStates()
                if len(states) == 0:
                    continue
                st = states[0]
                pt = omega_to_pt(float(st.omega), isclic=False)
                px = pt * math.cos(float(st.phi))
                py = pt * math.sin(float(st.phi))
                pz = float(st.tanLambda) * pt
                charge = 1 if st.omega > 0 else -1
                vectors.append(
                    {
                        "pfo_idx": pfo_idx,
                        "origin": [0.0, 0.0, 0.0],
                        "momentum": [float(px), float(py), float(pz)],
                        "charge": charge,
                        "associated": True,
                    }
                )
            except Exception:
                continue
    print("Trazas asociadas a PFOs:", len(associated_tracks))
    try:
        tracks = event.get("SiTracks_Refitted")
    except Exception:
        print("SiTracks_Refitted collection not found, skipping track extraction")
        return vectors
    for track in tracks:
        print("Track ID:", track.getObjectID().index)
        if track.getObjectID().index in associated_tracks:
            continue
        try:
            states = track.getTrackStates()
            if len(states) == 0:
                continue
            st = states[0]
            pt = omega_to_pt(float(st.omega), isclic=False)
            px = pt * math.cos(float(st.phi))
            py = pt * math.sin(float(st.phi))
            pz = float(st.tanLambda) * pt
            charge = 1 if st.omega > 0 else -1
            vectors.append(
                {
                    "pfo_idx": None,
                    "origin": [0.0, 0.0, 0.0],
                    "momentum": [float(px), float(py), float(pz)],
                    "charge": charge,
                    "associated": False,
                }
            )
        except Exception:
            continue

    return vectors


def extract_hits(event) -> List[Dict[str, Any]]:
    hit_type_map, hit_energy_map, hit_pos_map = build_hit_type_maps(event)
    hit_to_gen = build_hit_to_gen_map(event, hit_type_map)
    hit_to_pfo = build_hit_to_pfo_map(event, hit_type_map)

    hits: List[Dict[str, Any]] = []
    for key, detector_type in hit_type_map.items():
        pos = hit_pos_map[key]
        hits.append(
            {
                "collection_id": int(key[0]),
                "index": int(key[1]),
                "detector_type": int(detector_type),
                "x": float(pos[0]),
                "y": float(pos[1]),
                "z": float(pos[2]),
                "energy": float(hit_energy_map.get(key, 0.0)),
                "gen_idx": hit_to_gen.get(key),
                "pfo_idx": hit_to_pfo.get(key),
            }
        )

    return hits


def serialize_event(event, global_event: int, file_path: str, local_event: int, generator_status: int, dr_max: float) -> Dict[str, Any]:
    gen_particles = extract_gen_particles(event, generator_status)
    pfos = extract_pfos(event)
    return {
        "global_event": global_event,
        "file_path": file_path,
        "local_event": local_event,
        "gen_particles": gen_particles,
        "pfos": pfos,
        "associations": associate_by_dr(gen_particles, pfos, dr_max),
        "hits": extract_hits(event),
        "track_vectors": extract_track_vectors(event),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta eventos EDM4HEP a JSON para visualizador externo")
    parser.add_argument("-i", "--input", nargs="+", required=True, help="Ficheros ROOT de entrada")
    parser.add_argument("-o", "--output", default="event_dump.json", help="Fichero JSON de salida")
    parser.add_argument("--max-events", type=int, default=None, help="Máximo de eventos por fichero (por defecto: todos)")
    parser.add_argument("--generator-status", type=int, default=1, help="GeneratorStatus para resumen gen")
    parser.add_argument("--dr-max", type=float, default=0.1, help="Corte dR para asociaciones Gen-PFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    all_events: List[Dict[str, Any]] = []
    global_event = 0

    for path in args.input:
        reader = root_io.Reader(path)
        events = reader.get("events")

        for local_event, event in enumerate(events):
            if args.max_events is not None and local_event >= args.max_events:
                break

            all_events.append(
                serialize_event(
                    event=event,
                    global_event=global_event,
                    file_path=path,
                    local_event=local_event,
                    generator_status=args.generator_status,
                    dr_max=args.dr_max,
                )
            )
            global_event += 1

    payload = {
        "schema_version": 2,
        "generator_status": args.generator_status,
        "dr_max": args.dr_max,
        "num_events": len(all_events),
        "events": all_events,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f)

    print(f"Exportado {len(all_events)} eventos a {args.output}")


if __name__ == "__main__":
    main()
