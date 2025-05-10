import argparse
import logging
import sys
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_argparse():
    """
    Sets up the argument parser for the command-line interface.

    Returns:
        argparse.ArgumentParser: The argument parser object.
    """
    parser = argparse.ArgumentParser(description='Adds Laplacian noise to numerical data for differential privacy.')

    parser.add_argument('data_file', type=str, help='Path to the numerical data file (CSV).')
    parser.add_argument('epsilon', type=float, help='Privacy parameter epsilon (higher value means less privacy). Must be a positive number.')
    parser.add_argument('output_file', type=str, help='Path to the output file with noisy data.')
    parser.add_argument('--sensitivity', type=float, default=1.0, help='Sensitivity of the data (default: 1.0).')
    parser.add_argument('--column', type=int, default=0, help='Column index to apply noise to (default: 0).')
    parser.add_argument('--delimiter', type=str, default=',', help='Delimiter used in the data file (default: comma).')

    return parser

def laplace_noise(sensitivity, epsilon):
    """
    Generates Laplacian noise.

    Args:
        sensitivity (float): The sensitivity of the data.
        epsilon (float): The privacy parameter epsilon.

    Returns:
        float: A single Laplacian noise value.
    """
    try:
        if epsilon <= 0:
            raise ValueError("Epsilon must be a positive number.")
        
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return noise
    except Exception as e:
        logging.error(f"Error generating Laplacian noise: {e}")
        raise

def apply_differential_privacy(data_file, epsilon, output_file, sensitivity=1.0, column=0, delimiter=','):
    """
    Applies differential privacy by adding Laplacian noise to the specified column in the data file.

    Args:
        data_file (str): Path to the numerical data file (CSV).
        epsilon (float): Privacy parameter epsilon.
        output_file (str): Path to the output file with noisy data.
        sensitivity (float, optional): Sensitivity of the data. Defaults to 1.0.
        column (int, optional): Column index to apply noise to. Defaults to 0.
        delimiter (str, optional): Delimiter used in the data file. Defaults to ','.
    """
    try:
        with open(data_file, 'r') as infile, open(output_file, 'w') as outfile:
            for i, line in enumerate(infile):
                line = line.strip()
                values = line.split(delimiter)

                # Input validation
                if len(values) <= column:
                    logging.error(f"Column index {column} is out of range for line {i+1}.  Skipping this line.")
                    outfile.write(line + '\n')  # Write original line to output
                    continue
                
                try:
                    numerical_value = float(values[column])
                except ValueError:
                    logging.error(f"Non-numerical value found in column {column} on line {i+1}. Skipping this line.")
                    outfile.write(line + '\n')
                    continue

                # Apply Laplacian noise
                noise = laplace_noise(sensitivity, epsilon)
                noisy_value = numerical_value + noise

                # Create noisy line
                noisy_values = values[:] # Make a copy of the list
                noisy_values[column] = str(noisy_value)
                noisy_line = delimiter.join(noisy_values)

                outfile.write(noisy_line + '\n')

        logging.info(f"Differential privacy applied successfully. Noisy data written to {output_file}")

    except FileNotFoundError:
        logging.error(f"Data file not found: {data_file}")
        sys.exit(1)
    except ValueError as ve:
         logging.error(f"Value Error: {ve}")
         sys.exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
        sys.exit(1)

def main():
    """
    Main function to parse arguments and apply differential privacy.
    """
    parser = setup_argparse()
    args = parser.parse_args()

    try:
        apply_differential_privacy(args.data_file, args.epsilon, args.output_file, args.sensitivity, args.column, args.delimiter)
    except Exception as e:
        logging.error(f"Error during differential privacy application: {e}")
        sys.exit(1)


if __name__ == "__main__":
    """
    Usage Examples:

    1. Basic usage:
       python main.py data.csv 1.0 noisy_data.csv

    2. Specifying sensitivity and column index:
       python main.py data.csv 0.5 noisy_data.csv --sensitivity 0.5 --column 2

    3. Using a different delimiter:
       python main.py data.csv 2.0 noisy_data.csv --delimiter ';'
    """
    main()