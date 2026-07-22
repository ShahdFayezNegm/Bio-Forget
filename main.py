from src.patient import PatientRecord
from src.data_loader import download_real_ncbi_data
from src.system import BioForgetSystem


def main():
    print("=" * 80)
    print("Bio-Forget")
    print("=" * 80)

    system = BioForgetSystem()
    results = system.run()

    print("\nFinished Successfully!")
    return results


if __name__ == "__main__":
    main()