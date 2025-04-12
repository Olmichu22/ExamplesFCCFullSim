import ROOT 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import os

def id_to_key(event_id, photons=False):
  pi0 = "π⁰"
  pi = "π"
  mu = "μ"
  e = "e"
  n = "n"
  neutrino = "ν"
  tau = "τ"
  gamma = "γ"
  
  if photons:
    if event_id < 0:
      if event_id == -13:
        key = f"{mu}"
      elif event_id == -11:
        key = f"{e}"
      elif event_id <= -20:
        key = f"h{n}"
      else:
        key = "Unknown"
    elif event_id == 0:
      key = f"h"
    elif event_id < 10:
      key = f"h{event_id}{gamma}"
    elif event_id == 10:
      key = f"3h"
    else:
      key = f"3h{event_id-10}{gamma}"

  else:
    if event_id < 0:
      if event_id == -13:
        key = f"{tau} → {mu}2{neutrino}"
      elif event_id == -11:
        key = f"{tau} → {e}2{neutrino}"
      elif event_id <= -20:
        key = f"{pi}{n}"
      else:
        key = "Unknown"
    elif event_id == 0:
      key = f"{tau} → {pi}{neutrino}"
    elif event_id < 10:
      key = f"{tau} → {pi}{event_id}{pi0}{neutrino}"
    elif event_id == 10:
      key = f"{tau} → 3{pi}{neutrino}"
    else:
      key = f"{tau} → {3}{pi}{event_id-10}{pi0}{neutrino}"
  return key

def plot_1D_hist(file, variabs, labels, outputpath, normalize):
    """
    Plots individual 1D histograms.
    Each histogram is retrieved from the ROOT file, formatted using the provided labels,
    and then saved as a PNG file in the "1D" subfolder of outputpath.
    """
    # Create output folder for 1D plots
    out_dir = os.path.join(outputpath, "1D")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Create a canvas for drawing
    c = ROOT.TCanvas("c_1D", "1D Histograms", 800, 600)
    # Loop over each variable
    for var in variabs:
        histo = file.Get(var)
        if not histo:
            print(f"Warning: Histogram '{var}' not found in the ROOT file.")
            continue
        # If label config exists, set axis titles and overall title
        if var in labels:
            cfg = labels[var]
            histo.SetXTitle(cfg.get("x", ""))
            histo.SetYTitle(cfg.get("y", ""))
            histo.SetTitle(cfg.get("title", ""))
        c.Clear()
        if normalize:
          histo.DrawNormalized()
        else:
          histo.Draw()
        out_file = os.path.join(out_dir, f"{var}.png")
        c.SaveAs(out_file)
        print(f"Saved histogram '{var}' as '{out_file}'")
    c.Close()


def plot_hist_together(file, together_config, outputpath):
    """
    Plots groups of 1D histograms together on a single canvas.
    For each group defined in the plot_together configuration, all histograms are drawn
    on the same canvas with different colors and a legend.
    The resulting canvas is saved in a "together" subfolder inside the "1D" folder.
    """
    # Create output folder for together plots under 1D/together
    out_dir = os.path.join(outputpath, "1D", "together")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    # Iterate over each group in the together configuration
    for group, cfg in together_config.items():
        c = ROOT.TCanvas(f"c_together_{group}", group, 800, 600)
        legend = ROOT.TLegend(0.65, 0.70, 0.90, 0.90)
        legend.SetBorderSize(0)
        first = True
        normalize = cfg.get("norm", False)
        # For each histogram name in the group configuration
        for i, var in enumerate(cfg["variabs"]):
            histo = file.Get(var)
            if not histo:
                print(f"Warning: Histogram '{var}' not found in the ROOT file.")
                continue
            # Set the common X and Y axis titles from the group config

            
            # Assign a distinct line color for each histogram (simple scheme)
            if i == 0:
                histo.SetLineColor(ROOT.kRed)
            elif i == 1:
                histo.SetLineStyle(2)
                histo.SetLineColor(ROOT.kBlue)
            elif i == 2:
                histo.SetLineStyle(3)
                histo.SetLineColor(ROOT.kGreen+2)
            else:
                histo.SetLineColor(ROOT.kMagenta)
            histo.SetLineWidth(2)
            
            # Draw the first histogram normally; then draw others with "same"
            if first:
              histo.SetXTitle(cfg.get("x", ""))
              histo.SetYTitle(cfg.get("y", ""))
              # Optionally, set the title of the histogram (or leave it empty)
              histo.SetTitle(cfg.get("title", ""))
              if normalize:
                max_val = histo.GetMaximum()
                print(f"Max value before scaling: {max_val}")
                if max_val != 0:
                # Escalamos para que el máximo sea 1.
                  histo.Scale(1.0 / max_val)
                histo.GetYaxis().SetRangeUser(0., 1.1)

              histo.Draw("HIST")
              first = False
            else:
              if normalize:
                max_val = histo.GetMaximum()
                if max_val != 0:
                # Escalamos para que el máximo sea 1.
                  histo.Scale(1.0 / max_val)
                # histo_norm.GetYaxis().SetRangeUser(0, 1)
                

              histo.Draw("HIST same")

                # Set axis from 0 to 1
                
            # Add legend entry using the corresponding label from config, if provided
            label = cfg["labels"][i] if i < len(cfg["labels"]) else var
            legend.AddEntry(histo, label, "l")
        
        legend.Draw()
        out_file = os.path.join(out_dir, f"{group}.png")
        c.SaveAs(out_file)
        print(f"Saved group '{group}' as '{out_file}'")
        c.Close()


def plot_2D_hist(file, variabs, labels, outputpath):
    """
    Plots 2D histograms.
    Each 2D histogram is drawn with the "COLZ" option and saved as a PNG file
    in the "2D" subfolder of outputpath.
    """
    # Create output folder for 2D plots
    out_dir = os.path.join(outputpath, "2D")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    c = ROOT.TCanvas("c_2D", "2D Histograms", 800, 600)
    for var in variabs:
        histo = file.Get(var)
        if not histo:
            print(f"Warning: 2D Histogram '{var}' not found in the ROOT file.")
            continue
        if var in labels:
            cfg = labels[var]
            histo.SetXTitle(cfg.get("x", ""))
            histo.SetYTitle(cfg.get("y", ""))
            histo.SetTitle(cfg.get("title", ""))
        c.Clear()
        histo.Draw("COLZ")
        out_file = os.path.join(out_dir, f"{var}.png")
        c.SaveAs(out_file)
        print(f"Saved 2D histogram '{var}' as '{out_file}'")
    c.Close()

def plot_cm(results_df, outputpath, plotphotons=False):
    """
    Generates and saves two confusion matrix plots:
      1. With absolute values.
      2. With normalized values (per actual class) expressed as percentages.
    
    If plotphotons is True, uses the 'PhotonPredicted' column instead of 'Predicted'.
    The resulting confusion matrix may be rectangular if the true and predicted classes differ.
    
    Matrices are saved in the same folder as before, with an added suffix.
    """
    # Extract true labels and predicted labels based on the flag
    y_true = results_df['True']
    if plotphotons:
        y_pred = results_df['PhotonPredicted']
        suffix = "_photons"
    else:
        y_pred = results_df['Predicted']
        suffix = ""
    
    # Compute the confusion matrix:
    if plotphotons:
        # Use pandas crosstab to allow a rectangular matrix
        cm_df = pd.crosstab(y_true, y_pred)
        cm = cm_df.values
        classes_true = cm_df.index.values
        classes_pred = cm_df.columns.values
        mapped_classes_true = [id_to_key(cls, photons=False) for cls in classes_true]
        # print(classes_pred[:10])
        mapped_classes_pred = [id_to_key(cls, photons=True) for cls in classes_pred]
        # print(mapped_classes_pred[:10])
    else:
        # Use sklearn's confusion_matrix for a square matrix
        classes = np.unique(np.concatenate((y_true.values, y_pred.values)))
        cm = confusion_matrix(y_true, y_pred, labels=classes)
        mapped_classes_true = [id_to_key(cls, photons=False) for cls in classes]
        mapped_classes_pred = mapped_classes_true  # Same for both axes
    
    # Create output directory if it doesn't exist
    cm_dir = os.path.join(outputpath, "CM")
    if not os.path.exists(cm_dir):
        os.makedirs(cm_dir)
    
    # --- Absolute values plot ---
    plt.figure(figsize=(12, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Absolute Values)")
    plt.colorbar()
    if plotphotons:
        xtick_marks = np.arange(len(mapped_classes_pred))
        ytick_marks = np.arange(len(mapped_classes_true))
        plt.xticks(xtick_marks, mapped_classes_pred, rotation=45)
        plt.yticks(ytick_marks, mapped_classes_true)
    else:
        tick_marks = np.arange(len(mapped_classes_true))
        plt.xticks(tick_marks, mapped_classes_true, rotation=45)
        plt.yticks(tick_marks, mapped_classes_true)
    
    thresh = cm.max() / 2.
    # Annotate each cell with the absolute value
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", fontsize=8)
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, "confusion_matrix_absolute" + suffix + ".png"))
    plt.close()
    
    # --- Normalized values plot (per actual label) ---
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)  # Replace NaN with 0 for rows with zero sum
    plt.figure(figsize=(12, 8))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion Matrix (Normalized)")
    plt.colorbar()
    if plotphotons:
        plt.xticks(np.arange(len(mapped_classes_pred)), mapped_classes_pred, rotation=45)
        plt.yticks(np.arange(len(mapped_classes_true)), mapped_classes_true)
    else:
        tick_marks = np.arange(len(mapped_classes_true))
        plt.xticks(tick_marks, mapped_classes_true, rotation=45)
        plt.yticks(tick_marks, mapped_classes_true)
    
    thresh_norm = cm_normalized.max() / 2.
    # Annotate each cell with the percentage
    for i in range(cm_normalized.shape[0]):
        for j in range(cm_normalized.shape[1]):
            percentage = cm_normalized[i, j] * 100
            plt.text(j, i, f"{percentage:.1f}%",
                     horizontalalignment="center",
                     color="white" if cm_normalized[i, j] > thresh_norm else "black", fontsize=8)
    
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(cm_dir, "confusion_matrix_normalized" + suffix + ".png"))
    plt.close()
    
    print(f"Saved confusion matrices to '{cm_dir}'")


import ROOT

def plot_absolute(canvas, graphs, absolute_keys, colors, xaxis,
                           min_absolute_value, max_absolute_value,
                           title, outputpath, mig_str):
    """
    Grafica los datos absolutos utilizando PyROOT.
    
    Parámetros:
      canvas            : objeto TCanvas donde se dibuja el gráfico.
      graphs            : diccionario con objetos TGraph.
      absolute_keys     : lista de claves (strings) para los gráficos.
      colors            : lista con los códigos de color (por ejemplo, [1,2,3,...]).
      xaxis             : etiqueta para el eje X.
      min_absolute_value: valor mínimo para definir el rango del eje Y.
      max_absolute_value: valor máximo para definir el rango del eje Y.
      title             : título a dibujar en la parte superior.
      outputpath        : ruta para guardar la imagen.
      mig_str           : sufijo para el nombre del archivo.
    """
    canvas.cd()
    # Dibujar cada gráfico
    for i, key in enumerate(absolute_keys):
        color = colors[i % len(colors)]
        graph = graphs[key]
        
        graph.SetTitle(key)
        graph.SetLineColor(color)
        graph.SetMarkerColor(color)
        graph.SetMarkerStyle(20)
        graph.SetLineWidth(2)
        
        if i == 0:
            graph.GetXaxis().SetTitle(xaxis)
            graph.GetYaxis().SetTitle("Counts")
            graph.GetYaxis().SetRangeUser(0.9 * min_absolute_value, 1.1 * max_absolute_value)
            graph.Draw("alp")  # Dibuja ejes, línea y puntos
        else:
            graph.Draw("lp")   # Dibuja línea y puntos sin reconfigurar ejes

    canvas.BuildLegend()
    canvas.Update()

    # Agrega un recuadro de texto (TPaveText) centrado en la parte superior
    t = ROOT.TPaveText(0.2, 0.95, 0.8, 0.99, "NDC")
    t.SetTextAlign(22)  # Centrado horizontal y vertical
    t.SetTextSize(0.04)
    t.SetFillStyle(0)   # Fondo transparente
    t.SetBorderSize(0)  # Sin borde
    t.AddText(title)
    t.Draw()

    # Guarda el canvas como imagen
    canvas.SaveAs(outputpath + f"graphs_plot_{mig_str}.png")


def plot_metric(canvas, graphs, metric_keys, colors, xaxis,
                       title, outputpath, metric, mig_str):
    """
    Grafica los datos métricos utilizando PyROOT.
    
    Parámetros:
      canvas       : objeto TCanvas para el gráfico.
      graphs       : diccionario con objetos TGraph.
      metric_keys  : lista de claves (strings) para iterar los gráficos.
      colors       : lista con los códigos de color.
      xaxis        : etiqueta del eje X.
      title        : título del recuadro de texto.
      outputpath   : ruta de salida para la imagen.
      metric       : cadena identificadora para la gráfica (parte del nombre del archivo).
      mig_str      : sufijo adicional para el nombre del archivo.
    """
    canvas.cd()
    for i, key in enumerate(metric_keys):
        color = colors[i % len(colors)]
        graph = graphs[key]
        
        graph.SetLineColor(color)
        graph.SetMarkerColor(color)
        graph.SetMarkerStyle(20)
        graph.SetLineWidth(2)
        graph.SetTitle(key)
        graph.SetName(key)
        
        if i == 0:
            graph.GetXaxis().SetTitle(xaxis)
            graph.GetYaxis().SetTitle("Normalized Counts")
            graph.GetYaxis().SetRangeUser(0, 1.2)
            graph.Draw("alp")
        else:
            graph.Draw("lp")
    
    canvas.BuildLegend()
    canvas.Update()
    
    t = ROOT.TPaveText(0.2, 0.95, 0.8, 0.99, "NDC")
    t.SetTextAlign(22)
    t.SetTextSize(0.04)
    t.SetFillStyle(0)
    t.SetBorderSize(0)
    t.AddText(title)
    t.Draw()

    canvas.Update()
    canvas.SaveAs(outputpath + f"graphs_plot_{metric}_{mig_str}.png")
