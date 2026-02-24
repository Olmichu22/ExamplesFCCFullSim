#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script para fusionar múltiples archivos ROOT con TTrees.

Lee varios ficheros ROOT que contienen TTrees (e.g., outtree_original,
outtree_min_err, outtree_max_err) y los fusiona en un único archivo
de salida manteniendo la estructura de los árboles.

Uso típico:

  python merge_root_trees.py \
      --output merged_output.root \
      --input file1.root file2.root file3.root

  # O usando un fichero de texto con las rutas:
  python merge_root_trees.py \
      --output merged_output.root \
      --input-list files.txt

  # Con patrón glob:
  python merge_root_trees.py \
      --output merged_output.root \
      --glob "/path/to/files/*.root"
"""

import argparse
import os
import sys
import glob as glob_module
import ROOT
from ROOT import TFile, TChain


def get_tree_names(root_file_path):
    """
    Obtiene la lista de nombres de TTrees en un archivo ROOT.
    
    Parameters
    ----------
    root_file_path : str
        Ruta al archivo ROOT.
        
    Returns
    -------
    list
        Lista de nombres de TTrees encontrados.
    """
    tree_names = []
    
    tfile = TFile.Open(root_file_path, "READ")
    if not tfile or tfile.IsZombie():
        print(f"[ERROR] No se pudo abrir el archivo: {root_file_path}")
        return tree_names
    
    # Iterar sobre las claves del archivo
    for key in tfile.GetListOfKeys():
        obj = key.ReadObj()
        if isinstance(obj, ROOT.TTree):
            tree_names.append(key.GetName())
    
    tfile.Close()
    return tree_names


def merge_trees(input_files, output_file, tree_names=None, verbose=True):
    """
    Fusiona TTrees de múltiples archivos ROOT en un único archivo.
    
    Parameters
    ----------
    input_files : list
        Lista de rutas a los archivos ROOT de entrada.
    output_file : str
        Ruta al archivo ROOT de salida.
    tree_names : list, optional
        Lista de nombres de TTrees a fusionar. Si es None, se detectan
        automáticamente del primer archivo.
    verbose : bool
        Si True, imprime información de progreso.
        
    Returns
    -------
    dict
        Diccionario con estadísticas de la fusión.
    """
    if not input_files:
        print("[ERROR] No se proporcionaron archivos de entrada.")
        return None
    
    # Verificar que todos los archivos existen
    missing_files = [f for f in input_files if not os.path.isfile(f)]
    if missing_files:
        print("[ERROR] Los siguientes archivos no existen:")
        for f in missing_files:
            print(f"  - {f}")
        return None
    
    # Si no se especifican los nombres de árboles, detectarlos del primer archivo
    if tree_names is None:
        if verbose:
            print(f"[INFO] Detectando TTrees del primer archivo: {input_files[0]}")
        tree_names = get_tree_names(input_files[0])
        
        if not tree_names:
            print("[ERROR] No se encontraron TTrees en el primer archivo.")
            return None
        
        if verbose:
            print(f"[INFO] TTrees encontrados: {tree_names}")
    
    # Verificar que los árboles existen en todos los archivos
    if verbose:
        print("[INFO] Verificando consistencia de TTrees en todos los archivos...")
    
    for fpath in input_files:
        file_trees = get_tree_names(fpath)
        for tname in tree_names:
            if tname not in file_trees:
                print(f"[WARNING] El TTree '{tname}' no existe en {fpath}. Se omitirá.")
    
    # Crear el archivo de salida
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        if verbose:
            print(f"[INFO] Creado directorio de salida: {output_dir}")
    
    outfile = TFile(output_file, "RECREATE")
    if not outfile or outfile.IsZombie():
        print(f"[ERROR] No se pudo crear el archivo de salida: {output_file}")
        return None
    
    stats = {"trees": {}, "total_entries": 0, "input_files": len(input_files)}
    
    # Fusionar cada TTree usando TChain
    for tree_name in tree_names:
        if verbose:
            print(f"\n[INFO] Procesando TTree: '{tree_name}'")
        
        chain = TChain(tree_name)
        files_added = 0
        
        for fpath in input_files:
            # Verificar que el árbol existe en este archivo
            file_trees = get_tree_names(fpath)
            if tree_name in file_trees:
                chain.Add(fpath)
                files_added += 1
                if verbose:
                    # Obtener número de entradas de este archivo
                    temp_file = TFile.Open(fpath, "READ")
                    temp_tree = temp_file.Get(tree_name)
                    n_entries = temp_tree.GetEntries() if temp_tree else 0
                    temp_file.Close()
                    print(f"  + {fpath} ({n_entries} entries)")
        
        if files_added == 0:
            print(f"[WARNING] No se encontró el TTree '{tree_name}' en ningún archivo.")
            continue
        
        total_entries = chain.GetEntries()
        if verbose:
            print(f"[INFO] Total de entradas en '{tree_name}': {total_entries}")
        
        # Escribir el TTree fusionado al archivo de salida
        outfile.cd()
        merged_tree = chain.CloneTree(-1, "fast")
        merged_tree.Write()
        
        stats["trees"][tree_name] = {
            "entries": total_entries,
            "files_merged": files_added
        }
        stats["total_entries"] += total_entries
    
    outfile.Close()
    
    if verbose:
        print(f"\n[INFO] Fusión completada exitosamente.")
        print(f"[INFO] Archivo de salida: {output_file}")
        print(f"[INFO] Resumen:")
        print(f"  - Archivos procesados: {stats['input_files']}")
        print(f"  - TTrees fusionados: {len(stats['trees'])}")
        for tname, tinfo in stats['trees'].items():
            print(f"    * {tname}: {tinfo['entries']} entradas "
                  f"(de {tinfo['files_merged']} archivos)")
    
    return stats


def read_file_list(list_file):
    """
    Lee una lista de archivos desde un fichero de texto.
    
    Parameters
    ----------
    list_file : str
        Ruta al fichero con la lista de archivos (uno por línea).
        
    Returns
    -------
    list
        Lista de rutas a archivos.
    """
    files = []
    with open(list_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Ignorar líneas vacías y comentarios
            if line and not line.startswith('#'):
                files.append(line)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="Fusiona múltiples archivos ROOT con TTrees en un único archivo.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:

  # Fusionar archivos específicos:
  python merge_root_trees.py -o merged.root -i file1.root file2.root file3.root

  # Usar un fichero con la lista de archivos:
  python merge_root_trees.py -o merged.root --input-list files.txt

  # Usar un patrón glob:
  python merge_root_trees.py -o merged.root --glob "/path/to/results/**/outtree*.root"

  # Especificar qué TTrees fusionar:
  python merge_root_trees.py -o merged.root -i *.root --trees outtree_original outtree_min_err
        """
    )
    
    parser.add_argument(
        "-o", "--output",
        type=str,
        required=True,
        help="Ruta al archivo ROOT de salida."
    )
    
    parser.add_argument(
        "-i", "--input",
        type=str,
        nargs="*",
        default=[],
        help="Archivos ROOT de entrada."
    )
    
    parser.add_argument(
        "--input-list",
        type=str,
        help="Fichero de texto con lista de archivos ROOT (uno por línea)."
    )
    
    parser.add_argument(
        "--glob", "-g",
        type=str,
        help="Patrón glob para buscar archivos ROOT (e.g., '/path/**/*.root')."
    )
    
    parser.add_argument(
        "--trees",
        type=str,
        nargs="*",
        default=None,
        help="Nombres de los TTrees a fusionar. Si no se especifica, "
             "se detectan automáticamente del primer archivo."
    )
    
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Modo silencioso (menos información de progreso)."
    )
    
    args = parser.parse_args()
    
    # Recopilar todos los archivos de entrada
    input_files = list(args.input) if args.input else []
    
    # Añadir archivos desde fichero de lista
    if args.input_list:
        if not os.path.isfile(args.input_list):
            print(f"[ERROR] No se encontró el fichero de lista: {args.input_list}")
            sys.exit(1)
        input_files.extend(read_file_list(args.input_list))
    
    # Añadir archivos desde patrón glob
    if args.glob:
        glob_files = glob_module.glob(args.glob, recursive=True)
        if not glob_files:
            print(f"[WARNING] El patrón glob no encontró archivos: {args.glob}")
        input_files.extend(glob_files)
    
    # Eliminar duplicados manteniendo el orden
    seen = set()
    unique_files = []
    for f in input_files:
        abs_path = os.path.abspath(f)
        if abs_path not in seen:
            seen.add(abs_path)
            unique_files.append(abs_path)
    input_files = unique_files
    
    if not input_files:
        print("[ERROR] No se proporcionaron archivos de entrada.")
        print("Use --input, --input-list o --glob para especificar los archivos.")
        sys.exit(1)
    
    verbose = not args.quiet
    
    if verbose:
        print(f"[INFO] Archivos de entrada ({len(input_files)}):")
        for f in input_files:
            print(f"  - {f}")
        print()
    
    # Ejecutar la fusión
    stats = merge_trees(
        input_files=input_files,
        output_file=args.output,
        tree_names=args.trees,
        verbose=verbose
    )
    
    if stats is None:
        sys.exit(1)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
