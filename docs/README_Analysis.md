# Documentation of TauAnalysis, RhoAnalysis and ZAnalysis scripts

Este documento resume el propósito y la forma de utilizar los diferentes scripts que se encuentran en los directorios `TauAnalysis`, `RhoAnalysis` y `ZAnalysis`. La mayoría de ellos emplea funciones definidas en el directorio `modules` para reconstruir partículas, acceder a información de eventos y realizar cálculos específicos.

## TauAnalysis

### AnalyzeUnmatchedEvents.py
Analiza eventos en los que no se pudo asociar un `GenTau` con un `RecoTau`. Se leen ficheros CSV con las etiquetas verdaderas y predichas y, para el modo de decaimiento indicado, se agrupan los eventos por número de piones, fotones y neutrones. Los resultados se guardan en un nuevo CSV.
Ejecución básica:
```bash
python AnalyzeUnmatchedEvents.py -i <ruta_csvs> [-d DECAY]
```

### cutExperiment.py
Realiza un barrido de distintos parámetros de selección (por ejemplo cortes en momento de fotones o piones) y genera los ficheros de salida necesarios para calcular matrices de confusión o eficiencias. El script admite un archivo YAML de configuración con `-c`.

El YAML contiene las claves:

- `experiment`: parámetro sobre el que se realizará el barrido.
- `cuts`: valores a estudiar para cada corte (`NeutronCut`, `TauPhotonPCut`, `TauPionPCut`, `dRMax`, `MatchedGenMinDR`, `generalPCut`).
- `general`: ajustes globales como `decay`, `outfile`, `sample`, `matchedCM` y `test`.
- `output`: ruta y nombres de los ficheros generados.

Ejecución simplificada (solo se indican argumentos no contemplados en el YAML):
```bash
python cutExperiment.py -c mi_config.yaml --range 0 2 0.1
```
En caso necesario puede añadirse `--gatr-result <csv>` para usar predicciones externas.

### loss_events_analysis_summary.py
Recorre directorios con ficheros JSON generados por `plotTausLongEvent.py`, extrae la información relevante de las partículas de cada evento (taus, hijas cargadas y posibles taus reconstruidos) y la vuelca en un único CSV para posteriores análisis.
Ejecución básica:
```bash
python loss_events_analysis_summary.py <directorio_input> [-o salida.csv] [--fileid N]
```

### plotCutExperiment.py
A partir de los CSV generados en `cutExperiment.py`, dibuja la evolución de las migraciones o métricas cuando se modifica un corte. El aspecto de las gráficas se define en un YAML pasado con `-p`.

El YAML de ploteo suele incluir:

- `migrations`: pares "decay->pred" que se representarán.
- `axis`: campos `xlabel` y `title` del gráfico.
- `metric`: tipo de métrica a representar (`recall` o `purity`).

Ejecución básica (indicando solo la ruta a procesar y el YAML de ploteo):
```bash
python plotCutExperiment.py -i <ruta_experimento> -p mi_plot.yaml
```

### plotTausLongEvent.py
Procesa un único evento de un fichero ROOT (o de las predicciones de GATr) y guarda en JSON la información detallada de las partículas reconstruidas y generadas. Su configuración se toma de un YAML pasado con `-c`.

El YAML `taurecolong.yaml` define:

- `cuts`: valores de `dRMax`, `TauPhotonPCut`, `TauPionPCut`, `NeutronCut`, `MatchedGenMinDR` y `generalPCut`.
- `general`: campos como `decay`, `outfile`, `sample`, `matchedCM` y `test`.
- `output`: directorio y nombre de los ROOT generados.

Ejecución básica indicando el archivo y el evento a mostrar:
```bash
python plotTausLongEvent.py -c config/taurecolong.yaml --nfile 1 --eventid 0
```

### plotTausLongResults.py
Analiza colecciones completas de eventos y produce histogramas ROOT junto con las matrices de confusión. Todos los parámetros se leen de un YAML idéntico al empleado en `plotTausLongEvent.py`.

Uso habitual (solo se añade la ruta de predicciones si existe):
```bash
python plotTausLongResults.py -c config/taurecolong.yaml --gatr-result preds.csv
```

## RhoAnalysis

### analysisRHOTree.py
Script avanzado para estudiar la polarización de taus a través del canal rho. Su configuración se toma del mismo YAML utilizado para la reconstrucción larga de taus (`taurecolong.yaml`).

Para lanzar el estudio (opcionalmente con predicciones de GATr):
```bash
python analysisRHOTree.py -c config/taurecolong.yaml --gatr-result preds.csv
```

## ZAnalysis

### EventDistribution.py
Cuenta el número de leptones reconstruidos por evento y guarda histogramas de sus momenta. Las opciones de cortes y el nombre de salida se definen en un archivo YAML especificado con `-c`.

El YAML (`eventdist.yaml`) contiene:
- `cuts`: valores mínimos de momento para leptones y neutrones (`ElectPCut`, `MuonPCut`, `TauPhotonPCut`, `TauPionPCut`, `dRMax`, `NeutronCut`).
- `general`: `decay`, `outfile`, `sample` y `test`.
- `output`: nombres de archivos con los histogramas.

Ejecución básica simplemente indicando la configuración:
```bash
python EventDistribution.py -c config/eventdist.yaml
```

### ZDistributionPlot.py
Abre los ficheros producidos por `EventDistribution.py` y genera gráficas con ROOT para las distribuciones de leptones y sus momenta. Permite representar, además, las distribuciones separadas por número de leptones en el evento.
Ejecución básica:
```bash
python ZDistributionPlot.py -i <directorio_resultados>
```

### ZGenAnalysis.py
Estudia los bosones Z a nivel generador. Extrae sus propiedades cinemáticas, así como los modos de decaimiento de los taus hijos, y guarda histogramas ROOT y un CSV con las fracciones de cada modo.
Ejecución básica:
```bash
python ZGenAnalysis.py -f <muestra> [-d DECAY]
```

### ZRecoAnalysis.py
Realiza la reconstrucción del bosón Z y de los taus hijos a partir de PFOs. Calcula eficiencias comparando con la información generadora y escribe histogramas y matrices de confusión. Admite cortes en momento mínimo para los leptones.
Ejecución básica:
```bash
python ZRecoAnalysis.py -f <muestra> [opciones de corte]
```

### ZRecoSimplePlot.py
Herramienta de postprocesado que abre el ROOT de `ZRecoAnalysis.py` y dibuja histogramas seleccionados junto con la matriz de confusión de tipos de decaimiento reconstruidos frente a los verdaderos.
Ejecución básica:
```bash
python ZRecoSimplePlot.py -f <muestra>
```

### ZSimplePlot.py
Script ligero que permite superponer histogramas de distintas muestras en un mismo lienzo usando ROOT. Se indican las muestras, etiquetas y variables a representar.
Ejecución básica:
```bash
python ZSimplePlot.py -f decayAll decay0 -l "all" "0-prong" -v histoGenZP
```

### findZ_test.py
Pequeño test para `ZReco`. Permite validar las funciones que localizan bosones Z en los eventos, ya sea buscando todos (`complete`) o deteniéndose tras encontrar el primero (`single`).
Ejecución básica:
```bash
python findZ_test.py [-t complete|single]
```

---
Todos estos scripts esperan que las rutas a los ficheros de entrada correspondan a los datasets de FCC-ee almacenados en `/pnfs/ciemat.es/...`. Para un correcto funcionamiento es necesario haber instalado `key4hep` y disponer de las dependencias indicadas en el repositorio.
