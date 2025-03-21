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


def pick_random_publications(csv_path: Path, out_folder: Path, n_samples: int = 20):
    # get dataframe
    df = pd.read_csv(csv_path)

    # pick 20 random publications
    random_publications = df.sample(n=n_samples)

    # write to csv
    out_file_name = csv_path.name.replace(".csv", "_random20.csv")
    out_file = out_folder / Path(out_file_name)
    random_publications.to_csv(out_file, index=False)


def get_scopus_string_for_mathonco_newsletter(bib_file: str):
    # Load bibliography data
    bib_data = parse_file(bib_file)
    bib_dois = [e.fields.get("doi", None) for e in bib_data.entries.values()]

    # Create Scopus string
    doi_search_string = [f"DOI({doi})" for doi in bib_dois]
    scopus_string = " OR ".join(doi_search_string)

    return scopus_string


if __name__ == "__main__":
    print(get_scopus_string_for_mathonco_newsletter("data/MathOncoBibliograpy.bib"))