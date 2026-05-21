from podio import root_io
from modules import myutils
import argparse
from tqdm import tqdm
from pathlib import Path
def check_root_files(file_list):
    """
    Check if the given list of ROOT files exists and are readable.

    :param file_list: List of ROOT file paths to check.
    :return: List of valid ROOT files.
    """
    invalid_files = []
    valid_files = []
    for file in tqdm(file_list):
        my_file = Path(file)
        if my_file.is_file():
            try:
              root_file = myutils.open_root_file(file)
              if not root_file or root_file.IsZombie():
                # print("File %s is a zombie or could not be opened.", file)
                invalid_files.append(file.split('/')[-1])
                continue
              valid_files.append(file)
            except Exception as e:
                # print(f"Error opening file {file}: {e}")
                invalid_files.append(file.split('/')[-1])
                continue
        else:
            # print(f"File not found or not readable: {file}")
            invalid_files.append(file.split('/')[-1])
            
    return valid_files, invalid_files


def main():
    """
    Main function to execute the check on ROOT files.
    """
    parser = argparse.ArgumentParser(description="Check ROOT files for validity.")
    parser.add_argument('files', nargs='+', help='List of ROOT files to check.')
    args = parser.parse_args()
    # Example usage
    root_files = args.files
    print("Checking ROOT files:", root_files)
    valid_files, invalid_files = check_root_files(root_files)
    # Check number of events in each file
    # for file in tqdm(valid_files):
    #     root_file = myutils.open_root_file(file)
        # if root_file:
        #     num_events = root_file.GetEntries()
        #     print(f"File: {file}, Number of events: {num_events}")
        # else:
        #     print(f"Could not open file: {file}")
    # print("Valid ROOT files:", valid_files)
    print("Invalid ROOT files:", invalid_files)
if __name__ == "__main__":
    main()