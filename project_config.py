import os
from pathlib import Path

# Load environment variables if available, otherwise use defaults
BASE_DIR = Path(os.environ.get("RAAHUL_BASE_DIR", "/n/data1/hms/dbmi/zaklab/mli/rare_disease_diagnosis/test_camera_ready/data/"))
CURR_KG = os.environ.get("RAAHUL_CURR_KG", "8.9.21_kg")

# Paths
PROJECT_DIR = BASE_DIR
KG_DIR = PROJECT_DIR / "knowledge_graph" / CURR_KG
PREDICT_RESULTS_DIR = PROJECT_DIR / "results"
SEED = 33  # Set seed for reproducibility

# Dataset Paths
MY_DATA_DIR = Path("simulated_patients")
MY_TRAIN_DATA = MY_DATA_DIR / f"disease_split_train_sim_patients_{CURR_KG}.txt"
MY_VAL_DATA = MY_DATA_DIR / f"disease_split_val_sim_patients_{CURR_KG}.txt"
CORRUPT_TRAIN_DATA = MY_DATA_DIR / f"disease_split_train_sim_patients_{CURR_KG}_phencorrupt.txt"
CORRUPT_VAL_DATA = MY_DATA_DIR / f"disease_split_val_sim_patients_{CURR_KG}_phencorrupt.txt"

# Test Data Paths (Exomiser)
MY_TEST_DATA = Path(
    os.environ.get(
        "RAAHUL_TEST_DATA",
        "/home/ema30/zaklab/rare_disease_dx/formatted_patients/UDN_patients-2022-01-05/all_udn_patients_kg_8.9.21_kgsolved_exomiser_distractor_genes_5_candidates_mapped_only_genes.txt",
    )
)
MY_SPL_DATA = MY_TEST_DATA.with_suffix("_spl_matrix.npy")
MY_SPL_INDEX_DATA = MY_TEST_DATA.with_suffix("_spl_index_dict.pkl")

# Print paths to verify correctness
if __name__ == "__main__":
    print("Project Directory:", PROJECT_DIR)
    print("Knowledge Graph Directory:", KG_DIR)
    print("Train Data:", MY_TRAIN_DATA)
    print("Validation Data:", MY_VAL_DATA)
    print("Test Data:", MY_TEST_DATA)
