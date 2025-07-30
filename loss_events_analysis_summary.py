#!/usr/bin/env python3
import os
import sys
import json
import glob
import re
import math
import argparse
import pandas as pd

# Patrón para extraer nº de archivo y nº de evento del path
regex = re.compile(r'File_(\d+)_.*?/event_info_file_\d+_event_(\d+)\.json$')

def charged_daughter(daughters):
    """
    Filtra sólo muones (±13), electrones (±11) y piones cargados (±211).
    Devuelve la primera hija válida o None si no hay.
    """
    for d in daughters:
        if abs(d.get("PDGID", 0)) in (11, 13, 211):
            return d
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extrae información de taus y sus hijas cargadas de JSONs a un CSV usando pandas'
    )
    parser.add_argument(
        'input_dir',
        help='Directorio raíz que contiene las carpetas File_*_effis0.4_tph0.0_tpi1.0_n1_g0.0'
    )
    parser.add_argument(
        '-o', '--output',
        default='tau_summary.csv',
        help='Nombre del archivo CSV de salida (default: tau_summary.csv)'
    )
    parser.add_argument("--fileid", type=int, default=None,
                        help="ID del archivo a prcesar (opcional, por defecto procesa todos)")
    return parser.parse_args()


def main():
    args = parse_args()
    records = []

    # Patrón glob para buscar los JSON
    pattern = os.path.join(
        args.input_dir,
        '**',
        'File_*_effis0.4_tph0.0_tpi1.0_n1_g0.0',
        'event_info_file_*_event_*.json'
    )

    for path in glob.glob(pattern, recursive=True):
        match = regex.search(path)
        if not match:
            continue
        file_id, event_id = match.group(1), match.group(2)
        if args.fileid is not None and int(file_id) != args.fileid:
            continue
        with open(path) as f:
            data = json.load(f)

        taus = data.get('Gen Info', [])
        row = {
            'file': int(file_id),
            'event': int(event_id)
        }

        # Itera sobre Gen Taus y asigna directamente al diccionario
        for i in range(1, 3):  # tau1, tau2
            if i <= len(taus):
                tau = taus[i-1]
                # Gen Tau info
                row[f'tau{i}_ID'] = tau.get('Decay Type', '')
                row[f'tau{i}_P'] = tau.get('P', '')
                row[f'tau{i}_Theta'] = tau.get('Theta', '')
                # Hija cargada
                d = charged_daughter(tau.get('Daughters', []))
                if d:
                    pt = math.hypot(d['Px'], d['Py'])
                    row[f'd{i}_PDG'] = d.get('PDGID', '')
                    row[f'd{i}_P'] = d.get('P', '')
                    row[f'd{i}_Pt'] = round(pt, 6)
                    row[f'd{i}_Pz'] = d.get('Pz', '')
                    row[f'd{i}_Theta'] = d.get('Theta', '')
                else:
                    # No hay hija cargada, dejamos campos vacíos
                    row[f'd{i}_PDG'] = ''
                    row[f'd{i}_P'] = ''
                    row[f'd{i}_Pt'] = ''
                    row[f'd{i}_Pz'] = ''
                    row[f'd{i}_Theta'] = ''
            else:
                # Gen Tau no existe, rellenar vacíos
                row[f'tau{i}_ID'] = ''
                row[f'tau{i}_P'] = ''
                row[f'tau{i}_Theta'] = ''
                row[f'd{i}_PDG'] = ''
                row[f'd{i}_P'] = ''
                row[f'd{i}_Pt'] = ''
                row[f'd{i}_Pz'] = ''
                row[f'd{i}_Theta'] = ''
        
        row["tau1_PFO_reco"] = 0
        row["tau2_PFO_reco"] = 0
        recotaus = data.get('Reco Info', [])
        if recotaus:
            print("File", file_id)
            print("Event", event_id)
            print("Encontrados RecoTaus")
            taup3 = []
            for gen_tau in taus:
                taup3.append((gen_tau.get('Px', ''),
                            gen_tau.get('Py', ''),
                            gen_tau.get('Pz', '')
                            ))
            print("Tau p3", taup3)
            recotaup3 = []
            for reco_tau in recotaus:
                recotaup3.append((reco_tau.get('Px', ''),
                            reco_tau.get('Py', ''),
                            reco_tau.get('Pz', '')
                            )                    
                )
            print("Taus Gen", taus)    
            for i, reco_tau in enumerate(recotaus):
                print("Reco Tau", reco_tau)
                dist_1 = math.dist(recotaup3[i], taup3[0])
                dist_2 = math.dist(recotaup3[i], taup3[1])
                print("Distancias", dist_1, dist_2)
                if dist_1 < dist_2 and abs(reco_tau["Decay Type"]) == abs(taus[0]["Decay Type"]):
                    row["tau1_PFO_reco"] = 1
                    print("Caso tau1")
                elif dist_2 < dist_1 and abs(reco_tau["Decay Type"]) == abs(taus[1]["Decay Type"]):
                    row["tau2_PFO_reco"] = 1
                    print("Caso tau2")
            print("\n")
        
          
          

        records.append(row)

    # DataFrame y guardado a CSV
    df = pd.DataFrame(records)
    # Especificar orden de columnas
    cols = [
        'file', 'event',
        'tau1_ID', 'tau1_P', 'tau1_Theta',
        'd1_PDG', 'd1_P', 'd1_Pt', 'd1_Pz', 'd1_Theta',
        'tau2_ID', 'tau2_P', 'tau2_Theta',
        'd2_PDG', 'd2_P', 'd2_Pt', 'd2_Pz', 'd2_Theta', "tau1_PFO_reco",
        "tau2_PFO_reco"
    ]
    df = df[cols]
    df.to_csv(args.output, index=False)
    print(f"CSV guardado en {args.output}")
    print(df.head())

if __name__ == '__main__':
    main()