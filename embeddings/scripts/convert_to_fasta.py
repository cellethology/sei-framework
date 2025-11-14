"""Convert CSV file with sequences to FASTA format.

This script reads a CSV file containing sequence data and converts it to
FASTA format, using specified columns as the sequence identifier and
full_sequence as the sequence data.

# Single column (defaults to construct_name if not specified)
python embeddings/input_data/Feng_2023/convert_to_fasta.py \
    embeddings/input_data/Feng_2023/merged_data_dSort-Seq_results.csv \
    --id-columns construct_name --sequence-column full_sequence

# Multiple columns
python embeddings/input_data/Feng_2023/convert_to_fasta.py \
    embeddings/input_data/Feng_2023/merged_data_dSort-Seq_results.csv \
    --id-columns sequence_index construct_name --sequence-column full_sequence

# Use default (construct_name) - no need to specify
python embeddings/input_data/Feng_2023/convert_to_fasta.py \
    embeddings/input_data/Feng_2023/merged_data_dSort-Seq_results.csv --sequence-column full_sequence
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def csv_to_fasta(
    csv_path: str,
    output_path: Optional[str] = None,
    sequence_id_columns: Optional[list[str]] = None,
    sequence_column: str = "full_sequence",
    fallback_id_column: str = "sequence_index",
    header_delimiter: str = "|",
    include_column_names: bool = False,
) -> None:
    """Convert CSV file to FASTA format.

    Args:
        csv_path: Path to input CSV file.
        output_path: Path to output FASTA file. If None, uses CSV filename
            with .fasta extension.
        sequence_id_columns: List of column names to combine as sequence
            identifier in FASTA header. If None, defaults to ["construct_name"].
        sequence_column: Column name containing the sequence data.
            Defaults to "full_sequence".
        fallback_id_column: Column name to use as fallback identifier if
            sequence_id_columns are missing. Defaults to "sequence_index".
        header_delimiter: Delimiter to use when combining multiple columns.
            Defaults to "|".
        include_column_names: If True, include column names in header format
            like "col1=value1|col2=value2". If False, just use values like
            "value1|value2". Defaults to False.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        ValueError: If required columns are missing from the CSV file.
        OSError: If there's an error writing the output file.
    """
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    # Determine output path
    output_file = csv_file.with_suffix(".fasta") if output_path is None else Path(output_path)

    logger.info(f"Reading CSV file: {csv_path}")
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise OSError(f"Error reading CSV file: {e}") from e

    # Validate required columns
    if sequence_column not in df.columns:
        raise ValueError(
            f"Required column '{sequence_column}' not found in CSV. "
            f"Available columns: {list(df.columns)}"
        )

    # Determine which columns to use for sequence ID
    id_columns = sequence_id_columns if sequence_id_columns is not None else ["construct_name"]

    # Validate all ID columns exist
    missing_columns = [col for col in id_columns if col not in df.columns]
    if missing_columns:
        logger.warning(
            f"Columns {missing_columns} not found. " f"Using '{fallback_id_column}' as fallback."
        )
        if fallback_id_column not in df.columns:
            raise ValueError(
                f"ID columns {id_columns} not found and fallback "
                f"'{fallback_id_column}' also not found in CSV. "
                f"Available columns: {list(df.columns)}"
            )
        id_columns = [fallback_id_column]

    # Remove rows with missing sequences
    initial_count = len(df)
    df = df.dropna(subset=[sequence_column])
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} rows with missing sequences.")

    # Remove rows with empty sequences
    df = df[df[sequence_column].str.strip() != ""]
    if len(df) < initial_count:
        logger.warning(f"Removed {initial_count - len(df)} additional rows with empty sequences.")

    logger.info(f"Writing {len(df)} sequences to FASTA file: {output_file}")

    # Write FASTA file
    try:
        with open(output_file, "w") as f:
            for _, row in df.iterrows():
                # Build sequence ID from multiple columns
                id_parts = []
                for col in id_columns:
                    value = str(row[col])
                    # Clean value (replace spaces with underscores, but keep delimiter)
                    value = value.replace(" ", "_")
                    if include_column_names:
                        id_parts.append(f"{col}={value}")
                    else:
                        id_parts.append(value)

                seq_id = header_delimiter.join(id_parts)
                # Clean sequence ID (replace delimiter if it conflicts with FASTA format)
                # Note: We keep the delimiter, but ensure no other problematic chars
                seq_id = seq_id.replace("\n", "_").replace("\r", "_")

                # Get sequence
                sequence = str(row[sequence_column]).strip().upper()

                # Validate sequence contains only valid nucleotides
                valid_nucleotides = set("ATCGN")
                if not all(c in valid_nucleotides for c in sequence):
                    logger.warning(
                        f"Sequence {seq_id} contains invalid characters. " "Writing anyway."
                    )

                # Write FASTA entry
                f.write(f">{seq_id}\n")
                # Write sequence in 80-character lines (standard FASTA format)
                for i in range(0, len(sequence), 80):
                    f.write(f"{sequence[i:i+80]}\n")

        logger.info(f"Successfully wrote {len(df)} sequences to {output_file}")
    except Exception as e:
        raise OSError(f"Error writing FASTA file: {e}") from e


def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description="Convert CSV file with sequences to FASTA format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "csv_path",
        type=str,
        help="Path to input CSV file",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output FASTA file (default: same as input with .fasta extension)",
    )
    parser.add_argument(
        "--id-columns",
        type=str,
        nargs="+",
        default=None,
        help="One or more column names to combine as sequence identifier. "
        "Columns will be joined with the delimiter. If not specified, "
        "defaults to construct_name. Example: "
        "--id-columns sequence_index construct_name",
    )
    parser.add_argument(
        "--sequence-column",
        type=str,
        default="full_sequence",
        help="Column name containing sequence data (default: full_sequence)",
    )
    parser.add_argument(
        "--fallback-id-column",
        type=str,
        default="sequence_index",
        help="Fallback column for sequence ID if id-columns are missing "
        "(default: sequence_index)",
    )
    parser.add_argument(
        "--header-delimiter",
        type=str,
        default="|",
        help="Delimiter to use when combining multiple columns in header " "(default: |)",
    )
    parser.add_argument(
        "--include-column-names",
        action="store_true",
        help="Include column names in header (e.g., 'col1=value1|col2=value2') "
        "instead of just values (e.g., 'value1|value2')",
    )

    args = parser.parse_args()

    try:
        csv_to_fasta(
            csv_path=args.csv_path,
            output_path=args.output,
            sequence_id_columns=args.id_columns,
            sequence_column=args.sequence_column,
            fallback_id_column=args.fallback_id_column,
            header_delimiter=args.header_delimiter,
            include_column_names=args.include_column_names,
        )
    except (FileNotFoundError, ValueError, OSError) as e:
        logger.error(f"Error: {e}")
        raise SystemExit(1) from e


if __name__ == "__main__":
    main()
