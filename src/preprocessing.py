import pandas as pd
from pathlib import Path
from pybtex.database import parse_file


def check_coverage(scopus_csv: Path,
                   bib_file: Path,
                   out_folder: Path):
    """
    Check coverage of Scopus data in the bibliography file.
    """
    # Load Scopus data
    scopus = pd.read_csv(scopus_csv)
    scopus_titles = scopus["Title"].map(lambda x: x.replace(" ", "").casefold()).tolist()
    scopus_titles = set(scopus_titles)
    

    # Load bibliography data
    bib_data = parse_file(bib_file)

    # Get titles and DOIs for those which are not preprints
    preprint_publishers = ["Cold Spring Harbor Laboratory", "arXiv"]
    bib_titles = [e.fields["title"].replace(" ", "").casefold() for e in bib_data.entries.values() 
                  if e.fields['publisher'] not in preprint_publishers]
    bib_full_titles = [e.fields["title"] for e in bib_data.entries.values()
                       if e.fields['publisher'] not in preprint_publishers]
    bib_dois = [e.fields.get("doi", None) for e in bib_data.entries.values()
                if e.fields['publisher'] not in preprint_publishers]
    
    # Check coverage
    n_bib_titles = len(bib_titles)
    n_bib_titles_in_scopus = 0
    not_found = []
    for bib_title, bib_full_title, bib_doi in zip(bib_titles, bib_full_titles, bib_dois):
        if bib_title in scopus_titles:
            n_bib_titles_in_scopus += 1
        else:
            not_found.append({"title": bib_full_title, "doi": bib_doi})
    
    # Save elements not found
    out_csv_name = f"not_found_in_newsletter_{scopus_csv.name}"
    out_csv = out_folder / out_csv_name
    not_found_df = pd.DataFrame(not_found)
    not_found_df.to_csv(out_csv, index=False)

    # Pick them randomly to check what's excluded
    not_found_random = not_found_df.sample(20)
    not_found_random.to_csv(str(out_csv.resolve()).replace('.csv', '_random20.csv'), index=False)

    # Save coverage percentage
    tex_file = out_folder / Path('Newsletter_Coverage.tex')
    with open(tex_file, 'w') as outfile:
        outfile.write(f"Number of Scopus entries:                                  {len(scopus_titles)}\n")
        outfile.write(f"Number of Newsletter entries:                              {n_bib_titles}\n")
        outfile.write(f"Number of Newsletter entries included in scopus:           {n_bib_titles_in_scopus}\n")
        outfile.write(f"Coverage (fraction):                                       {n_bib_titles_in_scopus / n_bib_titles}\n")

    # Print file content
    with open(tex_file, 'r') as infile:
        print(infile.read())


def pick_random_publications(csv_path: Path, out_folder: Path, n_samples: int = 200):
    # get dataframe
    df = pd.read_csv(csv_path)

    # init out excel file
    out_file_name = csv_path.name.replace(".csv", "_random_samples.xlsx")
    out_file = out_folder / Path(out_file_name)

    # pick random publications 
    random_publications = df.sample(n=n_samples)

    # add empty columns for categories
    random_publications["Is MathOnco? (broadly)"] = ""
    random_publications["Category"] = ""
    random_publications["Comment"] = ""

    # rearrange the columns
    cols = random_publications.columns.tolist()
    # rearrange the columns to have the following order: Title, Authors, Is MathOnco? (broadly), Category, Comment, ...:
    for col in ["Comment", "Category", "Is MathOnco? (broadly)", "Authors", "Title"]:
        if col in cols:
            cols.remove(col)
            cols.insert(0, col)
    random_publications = random_publications[cols]

    # write to excel as different sheets
    writing_mode = 'a' if out_file.exists() else 'w'
    if_sheet_exists = 'replace' if out_file.exists() else None
    with pd.ExcelWriter(out_file, mode=writing_mode, if_sheet_exists=if_sheet_exists) as writer:
        sheet_name = f"random_{n_samples}"
        random_publications.to_excel(writer, sheet_name=sheet_name, index=False)


def get_scopus_string_for_mathonco_newsletter(bib_file: str):
    # Load bibliography data
    bib_data = parse_file(bib_file)
    bib_dois = [e.fields.get("doi", None) for e in bib_data.entries.values()]

    # Create Scopus string
    doi_search_string = [f"DOI({doi})" for doi in bib_dois]
    scopus_string = " OR ".join(doi_search_string)

    return scopus_string

def remove_incomplete_years(scopus_csv: Path):
    """
    Remove incomplete years from the Scopus data.
    """
    # Load Scopus data
    scopus = pd.read_csv(scopus_csv)
    scopus["Year"] = pd.to_datetime(scopus["Year"], format="%Y", errors="coerce").dt.year

    # inform the user
    print("Nullyears: ", scopus["Year"].isnull().sum())
    print("Published over 2024:" , scopus[scopus["Year"] > 2024].shape[0])

    # Remove incomplete years
    scopus = scopus[scopus["Year"].notnull()]
    scopus = scopus[scopus["Year"] < 2025]

    # Overwrite CSV
    scopus.to_csv(scopus_csv, index=False)


if __name__ == "__main__":
    with open("out/newsletter_string", "w") as outfile:
        outfile.write(get_scopus_string_for_mathonco_newsletter("data/MathOncoBibliograpy.bib"))