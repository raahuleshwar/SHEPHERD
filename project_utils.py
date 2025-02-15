import json
import jsonlines
import pickle
from pathlib import Path
import project_config


##############################################
# Read in/write patients

def read_patients(filename):
    with jsonlines.open(filename) as reader:
        return list(reader)  # More efficient than looping


def write_patients(patients, filename):
    with open(filename, "w") as output_file:
        json.dump(patients, output_file, separators=(",", ":"))  # Minify JSON


def read_dicts():
    kg_dir = Path(project_config.KG_DIR)
    project_dir = Path(project_config.PROJECT_DIR)

    file_map = {
        "hpo_to_idx_dict": kg_dir / f'hpo_to_idx_dict_{project_config.CURR_KG}.pkl',
        "ensembl_to_idx_dict": kg_dir / f'ensembl_to_idx_dict_{project_config.CURR_KG}.pkl',
        "disease_to_idx_dict": kg_dir / f'mondo_to_idx_dict_{project_config.CURR_KG}.pkl',
        "orpha_mondo_map": project_dir / 'preprocess' / 'orphanet' / 'orphanet_to_mondo_dict.pkl',
    }

    return {key: pickle.load(open(path, 'rb')) for key, path in file_map.items()}

