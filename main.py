import json
from pathlib import Path
import requests
import pandas as pd
from src.query import extract_bibliographic_data
from src.preprocessing import check_coverage, pick_random_publications
from src.plot import generate_plots


def main():
    query = '(DOCTYPE ( ar ) OR DOCTYPE ( re )) AND ("Mathematical Oncology" OR ( TITLE-ABS-KEY("Cancer*" OR "Tumor*" OR "Neoplasm*" OR "Neoplasm" OR "Carcino*" OR "Chemotherapy" OR "Radiotherapy" OR "Cancer Therapy" OR "Immunotherapy") AND ( TITLE-ABS ("Mathematical mode*" OR "Agent-based mode*" OR "Mechanistic mode*" OR "Stochastic mode*" OR "Numerical mode*" OR “Deterministic mode*” OR “Game theor*” OR “Game-theor*” OR (“Computational” AND “Systems Biology”) OR “data-driven” OR “mechanistic learning” OR “differential equation” OR “predator-prey” OR “predator prey”) OR AUTHKEY ("Mathematical mode*" OR "Agent-based mode*" OR "Mechanistic mode*" OR "Stochastic mode*" OR "Numerical mode*" OR “Deterministic mode*” OR “Game theor*” OR “Game-theor*” OR (“Computational” AND “Systems Biology”) OR “data-driven” OR “mechanistic learning” OR “differential equation” OR “predator-prey” OR “predator prey”) ) ) AND NOT(“Molecular Dynamics”))'
    out_folder = "out_Query21_3"

    # extract_bibliographic_data(
    #     out_folder_name=out_folder,
    #     scopus_csv="data/Query21_Scopus.csv",
    #     config_files=["config_franci.json", "config_alessio.json"],
    #     modeling_methods_file="data/methods.json",
    #     run_search=True
    # )

    generate_plots(data_folder=f"out/{out_folder}",
                   scopus_csv="data/Query21_Scopus.csv")


if __name__ == "__main__":
    main()
