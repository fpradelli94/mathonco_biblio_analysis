import pandas as pd
from pybtex.database import parse_file


def check_coverage(scopus_csv: str,
                   bib_file: str):
    """
    Check coverage of Scopus data in the bibliography file.
    """
    # Load Scopus data
    scopus = pd.read_csv(scopus_csv)
    scopus_titles = scopus["Title"].map(lambda x: x.replace(" ", "").casefold()).tolist()
    scopus_titles = set(scopus_titles)
    

    # Load bibliography data
    bib_data = parse_file(bib_file)
    bib_titles = [e.fields["title"].replace(" ", "").casefold() for e in bib_data.entries.values()]
    bib_full_titles = [e.fields["title"] for e in bib_data.entries.values()]
    bib_dois = [e.fields.get("doi", None) for e in bib_data.entries.values()]
    
    # Check coverage
    n_bib_titles = len(bib_titles)
    n_bib_titles_in_scopus = 0
    not_found = []
    for bib_title, bib_full_title, bib_doi in zip(bib_titles, bib_full_titles, bib_dois):
        if bib_title in scopus_titles:
            n_bib_titles_in_scopus += 1
        else:
            not_found.append({"title": bib_full_title, "doi": bib_doi})
    
    out_csv_name = f"data/not_found_{scopus_csv.split('/')[-1]}"
    pd.DataFrame(not_found).to_csv(out_csv_name, index=False)

    print(f"Number of titles in bibliography: {n_bib_titles_in_scopus / n_bib_titles}")


def pick_random_publications(csv_name: str, n_samples: int = 20):
    # get dataframe
    df = pd.read_csv(csv_name)

    # pick 20 random publications
    random_publications = df.sample(n=n_samples)

    # write to csv
    random_publications.to_csv(csv_name.replace(".csv", "_random20.csv"), index=False)


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