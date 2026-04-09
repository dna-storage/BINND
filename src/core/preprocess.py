import json
import os
import pandas as pd
import gzip
from Bio import SeqIO
from Bio import pairwise2
from binnd.core.utils.logger import setup_logger

logger = setup_logger(__name__)


class SequenceFilterStats:
    """ Class to store statistics about the sequence filtering process. It includes methods to increment various counts and save/load the stats to/from a JSON file.
    """

    def __init__(self):
        self.total_sequence_count = 0
        self.filtered_sequence_count = 0
        self.unique_sequence_count = 0
        self.exact_targetseq_match = 0
        self.insufficient_right_region = 0
        self.approximate_targetseq_match = 0
        self.reverse_complement_match = 0
        self.insertion_dict = {}
        self.left_trim_dict = {}
        self.alignment_score_dict = {}

    def increment_total_sequences(self):
        self.total_sequence_count += 1

    def increment_filtered_sequences(self):
        self.filtered_sequence_count += 1

    def increment_exact_targetseq_match(self):
        self.exact_targetseq_match += 1

    def increment_insufficient_right_region(self):
        self.insufficient_right_region += 1

    def increment_approximate_targetseq_match(self):
        self.approximate_targetseq_match += 1

    def increment_reverse_complement_match(self):
        self.reverse_complement_match += 1

    def increment_insertion_count(self, num_insertions):
        if num_insertions in self.insertion_dict:
            self.insertion_dict[num_insertions] += 1
        else:
            self.insertion_dict[num_insertions] = 1

    def increment_left_trim_count(self, num_left_trims):
        if num_left_trims in self.left_trim_dict:
            self.left_trim_dict[num_left_trims] += 1
        else:
            self.left_trim_dict[num_left_trims] = 1

    def increment_alignment_score(self, score):
        if score in self.alignment_score_dict:
            self.alignment_score_dict[score] += 1
        else:
            self.alignment_score_dict[score] = 1

    def sort_dict(self, dict):
        return {k: v for k, v in sorted(dict.items(), key=lambda item: item[0])}

    def save_to_json(self, file_path):
        """Saves stats to a JSON file instead of a Pickle."""
        with open(file_path, 'w') as f:
            # vars(self) returns the __dict__ of the object
            json.dump(vars(self), f, indent=4)

    @classmethod
    def load_from_json(cls, file_path):
        """Creates a class instance from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        instance = cls()
        instance.__dict__.update(data)
        return instance

    @staticmethod
    def combine_filter_stats(stats_dir: str, out_file_path: str) -> int:
        """Combine the stats from multiple JSON files in a directory into a single CSV file.
        

        Args:
            stats_dir (str): The directory containing the JSON stats files.
            out_file_path (str): The path where the combined CSV file will be saved.

        Returns:
            int: 0 if the operation was successful, 1 if there was an error (e.g., directory not found).
        """
        # Standard keys we want to extract for the summary CSV
        empty_dict = {
            "file": None,
            "total_sequence_count": 0,
            "filtered_sequence_count": 0,
            "unique_sequence_count": 0,
            "exact_targetseq_match": 0,
            "insufficient_right_region": 0,
            "approximate_targetseq_match": 0
        }

        stats_dict_list = []

        # Check if the directory exists
        if not os.path.exists(stats_dir):
            logger.warning(f"Directory not found: {stats_dir}")
            return 1
        files = os.listdir(stats_dir)
        files.sort()
        
        for file in files:
            if file.endswith(".json"):  # Switched from .pkl to .json
                stats = empty_dict.copy()
                file_path = os.path.join(stats_dir, file)

                with open(file_path, 'r') as f:
                    # This loads the data as a simple Python dictionary
                    loaded_stats = json.load(f)

                    stats["file"] = file.replace(".json", "")
                    # Access data using keys [] instead of dot notation .
                    stats["total_sequence_count"] = loaded_stats.get(
                        "total_sequence_count", 0)
                    stats["filtered_sequence_count"] = loaded_stats.get(
                        "filtered_sequence_count", 0)
                    stats["unique_sequence_count"] = loaded_stats.get(
                        "unique_sequence_count", 0)
                    stats["exact_targetseq_match"] = loaded_stats.get(
                        "exact_targetseq_match", 0)
                    stats["insufficient_right_region"] = loaded_stats.get(
                        "insufficient_right_region", 0)
                    stats["approximate_targetseq_match"] = loaded_stats.get(
                        "approximate_targetseq_match", 0)
                   
                    stats_dict_list.append(stats)

        # Convert to DataFrame and save
        if stats_dict_list:
            df = pd.DataFrame(stats_dict_list)
            df.to_csv(out_file_path, index=False)
            logger.info(f"Summary saved to {out_file_path}")

        return 0


def target_seq_based_filter(input_file: str,
                            target_sequence: str,
                            right_region_length: int,
                            output_dir: str,
                            is_bound: int,
                            exp_index: str,
                            is_perfect_match_only: bool = True,
                            approximate_match_edit_distance: int = 5,
                            output_file: str = None):
    """Filters sequences based on the presence of a target sequence (barcode) and the quality of the match.
    Args:
        input_file (str): Path to the input FASTQ file (can be gzipped).
        target_sequence (str): The target sequence to search for in the reads.
        right_region_length (int): The length of the region to extract after the target sequence.
        output_dir (str): Directory where the filtered sequences will be saved.
        is_bound (int): Indicator of whether the sequences are from bound (1) or unbound (0) or initial (-1) samples.
        exp_index (str): Experiment index to include in the output file name.
        is_perfect_match_only (bool): If True, only exact matches to the target sequence are considered. If False, approximate matches within a certain edit distance are also considered.
        approximate_match_edit_distance (int): The maximum edit distance allowed for approximate matches when is_perfect_match_only is False.
        output_file (str): Optional name for the output CSV file. If not provided, a default name will be generated based on is_bound and exp_index.
    """

    logger.info("Running sequence_align")
    logger.info(f"Input file: {input_file}")
    logger.info(f"Target sequence: {target_sequence}")
    logger.info(f"Is bound: {is_bound}")
    logger.info(f"Exp Index: {exp_index}")
    logger.info(f"Is perfect match only: {is_perfect_match_only}")
    logger.info(
        f"Approximate match edit distance: {approximate_match_edit_distance}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Output file: {output_file}")

    if input_file.endswith(".gz"):
        with gzip.open(input_file, "rt") as handle:
            records = list(SeqIO.parse(handle, "fastq"))
    else:
        records = list(SeqIO.parse(input_file, "fastq"))

    # stats object to store the statistics of the filtering process
    stats = SequenceFilterStats()

    columns = ["exp_index", "is_bound", "sequence"]
    default_value = None
    empty_dict = dict.fromkeys(columns, default_value)
    empty_dict['exp_index'] = exp_index
    empty_dict['is_bound'] = is_bound

    seq_dict_list = []
    thrown_out_seq_list = []
    seq_w_barcode_list = []
    progress_count = 0
    for record in records:
        if progress_count % 10000 == 0:
            logger.info(f"Processed {progress_count} sequences")
        progress_count += 1
        # if progress_count == 10000:
        #     break
        stats.increment_total_sequences()
        # initialize the dictionary that will be used to store the details of the sequence
        seq_dict = empty_dict.copy()

        seq = record.seq

        # find the target sequence
        index = seq.find(target_sequence)
        is_approximate_match = False
        if index != -1:  # exact match
            stats.increment_exact_targetseq_match()
            start_index = index + len(target_sequence)
            end_index = start_index + right_region_length
            seq_w_barcode_list.append(seq)

            if (end_index <= len(seq)):  # enough bases are there after the barcode for var primer seq
                stats.increment_filtered_sequences()
                primer = str(
                    seq[start_index:start_index + right_region_length])
                # logger.info(seq)
                if 'N' not in primer:
                    seq_dict['sequence'] = primer
                    seq_dict_list.append(seq_dict)
            else:
                stats.increment_insufficient_right_region()
                thrown_out_seq_list.append(seq)

        elif not is_perfect_match_only:  # check for approximate matches
            end_index = len(record.seq) - right_region_length
            # length of the remaining sequence should be atlest 30 (40 - 5 * 2)
            if (end_index >= len(target_sequence) - approximate_match_edit_distance * 2):
                # if the nucleotide matches, a score of +1. for mismatch or gap, a score of -1
                align = pairwise2.align.localms(
                    seq, target_sequence, 1, -1, -1, -1)
                # check if the score is above the threshold and we have enough sequences after the barcode region for var primer region
                if (align[0].score >= len(target_sequence) - approximate_match_edit_distance * 2):
                    stats.increment_approximate_targetseq_match()
                    if (align[0].end + right_region_length <= len(align[0].seqA)):
                        is_approximate_match = True
                        stats.increment_filtered_sequences()
                        primer = str(
                            align[0].seqA[align[0].end: align[0].end + right_region_length])
                        if 'N' not in primer:
                            seq_dict['sequence'] = primer
                            seq_dict_list.append(seq_dict)

                        # insertion stats
                        num_insertions = align[0].seqB[align[0].start:align[0].end].count(
                            '-')
                        if num_insertions > 0:
                            stats.increment_insertion_count(num_insertions)

                        # deletion stats
                        left_trim_count = 0
                        temp = align[0].seqA[0:align[0].end]
                        for i in range(0, align[0].end):
                            if temp[i] == '-':
                                left_trim_count += 1
                            else:
                                break
                        stats.increment_left_trim_count(left_trim_count)

                        stats.increment_alignment_score(int(align[0].score))

                    else:
                        stats.increment_insufficient_right_region()
                        thrown_out_seq_list.append(seq)
                else:
                    thrown_out_seq_list.append(seq)

            else:
                thrown_out_seq_list.append(seq)

        # Let's check reverse complement of the target sequence
        # if (index == -1 and is_approximate_match == 0): # make sure the sequence hasn't fall under any of the previous categories
        #     target_seq_rev_comp = Seq(target_sequence).reverse_complement()
        #     index = seq.find(target_seq_rev_comp)
        #     if (index != -1):  # exact match
        #         start_index = index + len(target_seq_rev_comp)
        #         end_index = start_index + right_region_length - config.VAR_PRIMER_ERROR
        #         # enough bases left for var primer seq
        #         if (end_index <= len(seq)):
        #             stats.increment_reverse_complement_match()

    if output_file is None:
        if is_bound == 1:
            output_file = f"bound_{exp_index}"
        elif is_bound == 0:
            output_file = f"unbound_{exp_index}"
        elif is_bound == -1:
            output_file = f"initial_{exp_index}"
        else:
            raise ValueError(
                "Invalid value for is_bound. Expected 1, 0, or -1.")

    df = pd.DataFrame(seq_dict_list, columns=columns)
    #calculate the unique sequence count
    stats.unique_sequence_count = df['sequence'].nunique()
    df.to_csv(os.path.join(output_dir, output_file + ".csv"), index=False)
    logger.info(f"Saved the filtered sequences to {output_file}.csv")

    # save the thrown out sequences
    # with open(os.path.join("/gpfs_backup/tuck_data/gbrihad/htp/data/htp/htp_thrownaway_data", output_file + "_thrown_away.csv"), "w") as handle:
    #     for seq in thrown_out_seq_list:
    #         handle.write(str(seq) + "\n")

    # df = pd.DataFrame(seq_w_barcode_list, columns=["sequence"])
    # df.to_csv(os.path.join("/gpfs_backup/tuck_data/gbrihad/htp/data/htp/seq_w_barcode", output_file+'.csv'), index=False)

    # sort the dictionaries
    stats.insertion_dict = stats.sort_dict(stats.insertion_dict)
    stats.left_trim_dict = stats.sort_dict(stats.left_trim_dict)
    stats.alignment_score_dict = stats.sort_dict(stats.alignment_score_dict)

    # save the stats object
    stats.save_to_json(os.path.join(output_dir, output_file + "_stats.json"))
    return 0


def CT_percent_based_filter(input_csv_path, CT_percent_threshold, output_directory):
    df = pd.read_csv(input_csv_path)
    sequences = df['sequence']
    filtered_sequences = []

    for seq in sequences:
        seq_half = seq[int(len(seq)/2):]
        seq_last_quarter = seq[int(3*len(seq)/4):]
        CT_percent_second_half = (seq_half.count(
            'C') + seq_half.count('T'))*100/len(seq_half)
        CT_percent_last_quarter = (seq_last_quarter.count(
            'C') + seq_last_quarter.count('T'))*100/len(seq_last_quarter)

        # print(f'seq: {seq}, CT_percent: {CT_percent}')
        if CT_percent_second_half <= CT_percent_threshold and CT_percent_last_quarter <= 80:
            filtered_sequences.append(seq)

    # print(f"Total sequences: {len(sequences)}")
    print(f"Extracted {len(filtered_sequences)} sequences")
    new_df = pd.DataFrame()
    new_df['sequence'] = filtered_sequences
    new_df.to_csv(os.path.join(output_directory,
                  os.path.basename(input_csv_path)), index=False)


def batch_CT_percent_based_filter(csv_directory, CT_percent_threshold, output_directory):
    for file in os.listdir(csv_directory):
        if file.endswith(".csv"):
            print(f"Processing {file}")
            CT_percent_based_filter(os.path.join(
                csv_directory, file), CT_percent_threshold, output_directory)


def fastp_adapter_trim_and_merge(input_file_1, input_file_2, merged_out_file):
    command = f'fastp --in1 {input_file_1} --in2 {input_file_2} --merge --merged_out {merged_out_file}'
    print(command)
    os.system(command)

def batch_fastp_adapter_trim_and_merge(input_folder, output_folder):
    
    for file in os.listdir(input_folder):
        if file.endswith('fastq.gz') and 'R1' in file:
            print(file, file.replace('R1', 'R2'))
            file_1 = os.path.join(input_folder, file)
            file_2 = os.path.join(input_folder, file.replace('R1', 'R2'))
            merged_file = os.path.join(output_folder, file.replace('R1', 'merged'))
            fastp_adapter_trim_and_merge(file_1, file_2, merged_file)