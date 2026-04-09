import primer3
import logging
import pandas as pd
import nupack
import os
import numpy as np
from binnd.core.utils.logger import setup_logger

logger = setup_logger(__name__)


def create_primer3_model(temperature: float, sodium_concentration: float, magnesium_concentration: float) -> primer3.thermoanalysis.ThermoAnalysis:
    """Create a primer3 ThermoAnalysis model with specified conditions.

    Args:
        temperature (float): The temperature in degrees Celsius.
        sodium_concentration (float): The sodium concentration in M.
        magnesium_concentration (float): The magnesium concentration in M.
    Returns:
        An instance of primer3.thermoanalysis.ThermoAnalysis
    """
    model = primer3.thermoanalysis.ThermoAnalysis(
        temp_c=temperature,
        mv_conc=sodium_concentration * 1000,  # Convert M to mM
        dv_conc=magnesium_concentration * 1000  # Convert M to mM
    )
    return model


def create_nupack_model(temperature: float, sodium_concentration: float, magnesium_concentration: float) -> nupack.Model:
    """Create a NUPACK model with specified conditions.

    Args:
        temperature (float): The temperature in degrees Celsius.
        sodium_concentration (float): The sodium concentration in M.
        magnesium_concentration (float): The magnesium concentration in M.
    Returns:
        An instance of nupack.Model
    """
    model = nupack.Model(material='dna', celsius=temperature,
                         sodium=sodium_concentration, magnesium=magnesium_concentration)
    return model


def calculate_deltag_primer3(seq1: str, seq2: str, primer3_model: primer3.thermoanalysis.ThermoAnalysis) -> float:
    """Calculate deltaG using primer3 library.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
        primer3_model: An instance of primer3.thermoanalysis.ThermoAnalysis
    Returns:
        float: The calculated deltaG value in kcal/mol.
    """

    result = primer3_model.calc_heterodimer(seq1, seq2)
    result = result.todict()

    dG = result['dg']
    # Convert from cal/mol to kcal/mol and round to 6 decimal places
    dG = round(dG / 1000, 6)
    return dG


def calculate_deltag_nupack(seq1: str, seq2: str, nupack_model: nupack.Model) -> float:
    """Calculate deltaG using NUPACK library.

    Args:
        seq1 (str): The first sequence.
        seq2 (str): The second sequence.
        nupack_model: An instance of nupack.Model
    Returns:
        float: The calculated deltaG value in kcal/mol.
    """

    seq_1 = nupack.Strand(seq1, name="s1")
    seq_2 = nupack.Strand(seq2, name="s2")

    # Create a complex set with the two strands excluding single strands and homodimers for computational efficiency
    set = nupack.ComplexSet(strands=[seq_1, seq_2], complexes=nupack.SetSpec(
        max_size=2, exclude=[[seq_1], [seq_2], [seq_1, seq_1], [seq_2, seq_2]]))

    # Create a complex for the heterodimer, the one that we want to calculate deltaG for
    complex = nupack.Complex([seq_1, seq_2], name="s1+s2")

    result = nupack.complex_analysis(
        complexes=set, model=nupack_model, compute=["pfunc"])

    dg = round(result[complex].free_energy, 6)

    return dg


def calculate_deltag_primer3_batch(input_csv_path: str,
                                   seq1_col_name: str,
                                   seq2_col_name: str,
                                   out_col_name: str,
                                   primer3_model: primer3.thermoanalysis.ThermoAnalysis,
                                   chunksize: int = 10000,
                                   verbose: bool = False) -> int:
    """Calculate deltaG for pairs of sequences in a CSV file and add results to a new column.
    Args:
        input_csv_path (str): Path to the input CSV file.
        seq1_col_name (str): Name of the column containing the first sequences (5` - 3`).
        seq2_col_name (str): Name of the column containing the second sequences (5` - 3`).
        out_col_name (str): Name of the output column to store deltaG values.
        primer3_model: An instance of primer3.thermoanalysis.ThermoAnalysis
        chunksize (int): Number of rows to process at a time.
        verbose (bool): Whether to log progress information.
    Returns:
        int: 0 on success.
    """
    if verbose:
        logger.info(
            f"Starting batch deltaG calculation for file: {input_csv_path}")

        total_rows = sum(1 for _ in open(input_csv_path)) - 1  # Exclude header
        total_chunks = (total_rows + chunksize - 1) // chunksize
        logger.info(f"Total rows: {total_rows}, Total chunks: {total_chunks}")

    # Temporary output file for safety
    output_temp_path = input_csv_path + ".tmp"
    with pd.read_csv(input_csv_path, chunksize=chunksize) as reader:
        for i, chunk in enumerate(reader):
            if verbose:
                if i % 10 == 0:
                    logger.info(f"Processing chunk {i}...")
            chunk[out_col_name] = [
                calculate_deltag_primer3(s1, s2, primer3_model)
                for s1, s2 in zip(chunk[seq1_col_name], chunk[seq2_col_name])
            ]

            write_header = (i == 0)
            chunk.to_csv(output_temp_path, mode='a',
                         index=False, header=write_header)

    # Replace original file with updated file
    os.replace(output_temp_path, input_csv_path)
    return 0


def calculate_deltag_primer3_files(file_list: list,
                                   seq1_col_name: str,
                                   seq2_col_name: str,
                                   out_col_name: str,
                                   primer3_model: primer3.thermoanalysis.ThermoAnalysis,
                                   chunksize: int = 10000,
                                   verbose: bool = False) -> int:
    """Calculate deltaG for files in a list of CSV files.

    Args:
        file_list (list): List of paths to CSV files.
        seq1_col_name (str): Name of the column containing the first sequences (5` - 3`).
        seq2_col_name (str): Name of the column containing the second sequences (5` - 3`).
        out_col_name (str): Name of the output column to store deltaG values.
        primer3_model: An instance of primer3.thermoanalysis.ThermoAnalysis
        chunksize (int): Number of rows to process at a time.
        verbose (bool): Whether to log progress information.

    Returns:
        int: 0 on success.
    """
    if verbose:
        logger.info(f"Starting deltaG calculation for {len(file_list)} files.")
        logger.info(f"File list: {file_list}")

    for file_path in file_list:
        calculate_deltag_primer3_batch(
            file_path,
            seq1_col_name,
            seq2_col_name,
            out_col_name,
            primer3_model,
            chunksize=chunksize,
            verbose=verbose
        )
    return 0


def generate_beadprimer_dg_distribution(seq_csv_path: str,
                                        seq_col_name: str,
                                        bead_primer: str,
                                        dg_model,
                                        dg_model_type: str,
                                        out_path: str,
                                        freq_col_name: str = None,
                                        chunk_size=10000,
                                        verbose=False) -> int:
    """Generate deltaG distribution for sequences against a bead primer and save to CSV.

    Args:
        seq_csv_path (str): Path to the CSV file containing sequences.
        seq_col_name (str): Name of the column containing sequences.
        bead_primer (str): The bead primer sequence.
        dg_model: The deltaG model (primer3 or nupack).
        dg_model_type (str): Type of deltaG model ("primer3" or "nupack").
        out_path (str): Path to save the output CSV file.
        chunk_size (int): Number of rows to process at a time.
        verbose (bool): Whether to log progress information.
    Returns:
        int: 0 on success.
    """

    if dg_model_type == "primer3":
        dg_func = calculate_deltag_primer3
        if verbose:
            logger.info("Using primer3 for deltaG calculations.")
    elif dg_model_type == "nupack":
        dg_func = calculate_deltag_nupack
        if verbose:
            logger.info("Using NUPACK for deltaG calculations.")
    else:
        logger.error(f"Unsupported dg_model_type: {dg_model_type}")
        return

    with pd.read_csv(seq_csv_path, chunksize=chunk_size) as reader:
        for i, chunk in enumerate(reader):
            if verbose and i % 10 == 0:
                logger.info(f"Processing chunk {i}...")

            dg_values = np.array([
                dg_func(seq, bead_primer, dg_model)
                for seq in chunk[seq_col_name]
            ])

            # Frequency Replication
            if freq_col_name is not None:
                dg_values = np.repeat(
                    dg_values, chunk[freq_col_name].astype(int).values)

            output_df = pd.DataFrame(
                dg_values, columns=[f'deltaG_{dg_model_type}'])

            write_header = (i == 0)
            output_df.to_csv(out_path, mode='a', index=False,
                             header=write_header)

    return 0


def test_primer3_deltag():
    seq1 = "TATGTTCACGGCGGGACTTG"
    seq2 = "GGACTCTCTGCGTTGATAGA"
    temperature = 25.0  # degrees Celsius
    sodium_concentration = 0.05  # 50 mM
    magnesium_concentration = 0.0

    model = create_primer3_model(
        temperature, sodium_concentration, magnesium_concentration)

    dG = calculate_deltag_primer3(
        seq1, seq2, model)
    print(f"Calculated deltaG: {dG} kcal/mol")


def test_nupack_deltag():
    seq1 = "ACGACTTTAGCCTAACCAGT"
    seq2 = "CGAGATGGGGGTGGTTAGGC"
    temperature = 25.0  # degrees Celsius
    sodium_concentration = 0.05  # 50 mM
    magnesium_concentration = 0.0

    model = create_nupack_model(
        temperature, sodium_concentration, magnesium_concentration)
    dG = calculate_deltag_nupack(
        seq1, seq2, model)
    print(f"Calculated deltaG (NUPACK): {dG} kcal/mol")


if __name__ == "__main__":
    pass
    test_primer3_deltag()
    # test_nupack_deltag()
