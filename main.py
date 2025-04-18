import numpy as np
from pathlib import Path
from src.query import extract_bibliographic_data
from src.preprocessing import check_coverage, pick_random_publications, remove_incomplete_years, margin_of_error
from src.plot import generate_plots





def main():
    # ------ Init ---------------
    # set the query number
    query_number = '34'   

    # create output folder                                    
    out_folder = Path(f"out/out_Query{query_number}")
    Path(out_folder).mkdir(exist_ok=True)

    # load data
    scopus_csv = Path(f'data/Query{query_number}_Scopus.csv')
    bib_file = Path('data/MathOncoBibliograpy.bib')

    # init seed (used in preprocessing)
    np.random.seed(221194)

    # ------ Preprocessing ------
    # Check the query with sampling and coverage
    out_preprocessing = Path(f"{out_folder}/preprocessing")
    out_preprocessing.mkdir(exist_ok=True)
    remove_incomplete_years(scopus_csv)
    pick_random_publications(scopus_csv, out_preprocessing)
    check_coverage(scopus_csv, bib_file, out_preprocessing)

    # ------ Extraction ---------
    extract_bibliographic_data(
        out_folder_name=out_folder,
        scopus_csv=scopus_csv,
        config_file=Path("config/config_moffitt01.json"),
        modeling_methods_file="data/methods.json",
        run_search=True
    )

    # ------ Visualization ------
    generate_plots(data_folder=f"{out_folder}",
                   scopus_csv=f"data/Query{query_number}_Scopus.csv")


if __name__ == "__main__":
    main()

