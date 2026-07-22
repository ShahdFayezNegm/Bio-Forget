from typing import List
from tqdm import tqdm

from src.patient import PatientRecord


class FastaParser:

    def parse_file(self, filepath: str) -> List[PatientRecord]:

        records = []

        with open(filepath, "r") as f:
            content = f.read()

        fasta_blocks = content.split(">")[1:]

        # Iterate through each FASTA block
        for block in tqdm(fasta_blocks, desc="Parsing"):

            lines = block.strip().split("\n")

            if len(lines) < 2:
                continue

            header = lines[0]

            sequence = (
                "".join(lines[1:])
                .replace(" ", "")
                .replace("\r", "")
                .upper()
            )

            patient_id = header.split()[0].split("|")[0]

            gene_name = "Unknown"

            if "|" in header:
                parts = header.split("|")
                if len(parts) >= 3:
                    gene_name = parts[2]

            diagnosis = 1 if "CANCER" in header.upper() else 0

            records.append(
                PatientRecord(
                    patient_id=patient_id,
                    sequence=sequence[:1500],
                    diagnosis=diagnosis,
                    gene_name=gene_name,
                )
            )

        print(f"[FASTA Parser] Loaded {len(records)} records")

        return records