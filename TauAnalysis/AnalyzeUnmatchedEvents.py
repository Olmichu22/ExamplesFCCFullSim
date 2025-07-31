import argparse
import ROOT
import logging
import os
import numpy
import pandas

def setup_logging():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    return logger
  
def parse_arguments():
    parser = argparse.ArgumentParser(description="Analyze unmatched events in FCC data",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-i", "--input", required=True, help="Input ROOT file with unmatched events")
    parser.add_argument("-d", "--decay", default=-11, help="Decay type to analyze (default: -11 for tau decay)", type=int)
    return parser.parse_args()
  

def main():
  logger = setup_logging()
  args = parse_arguments()
  
  file = "true_predicted_label_decayAll.csv"
  input_path = args.input
  input_file = os.path.join(input_path, file)
  decay_type = args.decay
  
  logger.info(f"Reading input file: {input_file}")
  try:
    data = pandas.read_csv(input_file)
  except FileNotFoundError:
    logger.error(f"Input file '{input_file}' not found.")
    return
  except pandas.errors.EmptyDataError:
    logger.error(f"Input file '{input_file}' is empty.")
    return
  except Exception as e:
    logger.error(f"Error reading input file: {e}")
    return
  
  logger.info(f"Input file read successfully. Number of rows: {len(data)}")
  
  logger.info(f"Filtering data for decay type: {decay_type}")
  
  filtered_data = data[(data['True'] == decay_type) & (data["Predicted"] == -1)]
  
  # Agrupamos los resultados según las 3 últimas columnas
  columns = ["Countpions","Countphotons","Countneutrons"]
  grouped_data = filtered_data.groupby(columns).size().reset_index(name='Count')
  logger.info(f"Grouped data size: {len(grouped_data)}")
  # Mostramos los resultados
  logger.info("Grouped results:")
  for index, row in grouped_data.iterrows():
    logger.info(f"Pions: {row['Countpions']}, Photons: {row['Countphotons']}, Neutrons: {row['Countneutrons']}, Count: {row['Count']}")
  
  # Guardamos los resultados en un archivo CSV
  output_file = f"unmatched_events_summary_{decay_type}.csv"
  output_path = os.path.join(input_path, output_file)
  try:
    grouped_data.to_csv(output_path, index=False)
    logger.info(f"Results saved to {output_path}")
  except Exception as e:
    logger.error(f"Error saving results to CSV: {e}")
    return

if __name__ == "__main__":
    main()
  