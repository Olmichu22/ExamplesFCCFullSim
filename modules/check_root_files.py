import ROOT
from pathlib import Path
import argparse

def check_root_files(path, tree_name="events", expected_events=1000):
    path = Path(path)

    if not path.exists():
        print(f"[ERROR] La ruta no existe: {path}")
        return

    root_files = sorted(path.glob("*.root"))

    if not root_files:
        print("[INFO] No se encontraron archivos ROOT")
        return

    valid_files = 0
    wrong_event_files = []

    for f in root_files:
        try:
            root_file = ROOT.TFile.Open(str(f))
        except OSError:
            print(f"[ZOMBIE] {f.name} (no se pudo abrir)")
            continue

        if not root_file or root_file.IsZombie():
            print(f"[ZOMBIE] {f.name}")
            continue

        valid_files += 1

        tree = root_file.Get(tree_name)
        if not tree:
            wrong_event_files.append((f.name, "no tree"))
        else:
            n_events = tree.GetEntries()
            if n_events != expected_events:
                wrong_event_files.append((f.name, n_events))  

        root_file.Close()

    print("\n===== RESUMEN =====")
    print(f"Archivos ROOT totales : {len(root_files)}")
    print(f"Archivos NO zombies   : {valid_files}")

    if wrong_event_files:
        print("\nArchivos con número de eventos != 1000:")
        for name, info in wrong_event_files:
            print(f"  - {name}: {info}")
    else:
        print("\nTodos los archivos válidos tienen 1000 eventos ✔")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="Ruta a los archivos ROOT")
    parser.add_argument("--tree", default="events", help="Nombre del TTree")
    parser.add_argument("--expected-events", type=int, default=1000)
    args = parser.parse_args()

    check_root_files(
        args.path,
        tree_name=args.tree,
        expected_events=args.expected_events
    )
