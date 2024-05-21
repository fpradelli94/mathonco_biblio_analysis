import re
import json
from pathlib import Path
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt


def plot_journals(journals_df: pd.DataFrame,
                  top_n: int = 20,
                  output_file: str = None):
    """
    Plot the top n journals with the most publications
    """
    # sort journals for number of publications
    journals_df = journals_df.sort_values(by='n_publications', ascending=False)

    # get top n journals
    top_journals = journals_df.head(top_n)

    # plot
    _, ax = plt.subplots()
    labels = [f"{journal} ({citescore})" for journal, citescore in zip(top_journals['title'], top_journals['journal_citescore'])]
    ax.barh(top_journals['title'], top_journals['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f'Top {top_n} journals with most publications')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_subject_areas(subject_areas_df: pd.DataFrame,
                       top_n: int = 20,
                       output_file: str = None):
    """
    Plot the top n journals with the most publications
    """
    # sort journals for number of publications
    subject_areas_df = subject_areas_df.sort_values(by='n_publications', ascending=False)

    # get top n journals
    top_subject_areas = subject_areas_df.head(top_n)
    top_subject_areas = top_subject_areas.sort_values(by='n_publications', ascending=True)

    # plot
    _, ax = plt.subplots()
    labels = [f"{subject} ({abbrev}; {code})" for subject, abbrev, code in zip(top_subject_areas['name'], top_subject_areas['abbrev'], top_subject_areas['code'])]
    ax.barh(top_subject_areas['name'], top_subject_areas['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f'Top {top_n} subjects with most publications')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_authors(authors_df: pd.DataFrame,
                 institiutions_df: pd.DataFrame,
                 top_n: int = 20,
                 output_file: str = None):
    """
    Plot the top n authors with the most publications
    """
    # sort authors for number of publications
    authors_df = authors_df.sort_values(by='n_publications', ascending=False)

    # get top n authors
    top_authors = authors_df.head(top_n)
    top_authors = top_authors.sort_values(by='n_publications', ascending=True)

    # get institution for each author
    top_authors_institutions_id = top_authors['affiliation_id(s)']

    # get institution names
    top_authors['affiliation'] = top_authors_institutions_id.apply(lambda x: [institiutions_df.loc[institiutions_df['scopus_id'] == int(i), 'institute_name'].values[0] for i in x.split(';')][0])

    # plot
    _, ax = plt.subplots()
    labels = [f"{author} ({aff})" for author, aff in zip(top_authors['author_indexed-name'], top_authors['affiliation'])]
    ax.barh(top_authors['author_indexed-name'], top_authors['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f'Top {top_n} authors with most publications')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_citations(scopus_df: pd.DataFrame,
                   output_file: str = None):
    """
    Plot histogram of number of citations
    """
    _, ax = plt.subplots()
    ax.hist(scopus_df['Cited by'], bins=25)
    ax.axvline(scopus_df['Cited by'].mean(), color='tab:orange', linestyle='dashed', linewidth=1, label='Mean')
    ax.legend()
    print(f"Mean number of citations: {scopus_df['Cited by'].mean()}")
    ax.set_yscale('log')
    ax.set_xlabel('Number of citations')
    ax.set_ylabel('Number of publications')
    ax.set_title('Number of citations per publication')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_institutions(institutions_df: pd.DataFrame,
                      top_n: int = 20,
                      output_file: str = None):
    """
    Get top n institutions for number of publications
    """
    # sort authors for number of publications
    institutions_df = institutions_df.sort_values(by='n_publications', ascending=False)

    # get top n authors
    top_institutions = institutions_df.head(top_n)

    # plot
    _, ax = plt.subplots()
    labels = [f"{institute} ({city}, {country})" for institute, city, country in zip(top_institutions['institute_name'], top_institutions["institute_city"], top_institutions['institute_country'])]
    ax.barh(top_institutions['institute_name'], top_institutions['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f'Top {top_n} institutions with most publications')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_countries(intitutions_df: pd.DataFrame,
                   top_n: int = 20,
                   output_file: str = None):
    """
    Get top n most cited publications
    """
    # get publications from countries
    countries = intitutions_df.groupby('institute_country').sum().sort_values(by='n_publications', ascending=False).drop(columns=['scopus_id', 'institute_name', 'institute_city'])

    # get top countries
    countries = countries.head(top_n)
    countries = countries.sort_values(by='n_publications', ascending=True)

    # plot
    _, ax = plt.subplots()
    ax.barh(countries.index, countries['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_title(f'Top {top_n} countries with most publications')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")

    # within the US, get publications for each state
    us_only = intitutions_df[intitutions_df['institute_country'] == 'United States']
    us_only = us_only.dropna()
    us_states = us_only.groupby('institute_state').agg({'n_publications': 'sum',
                                                        'institute_city': lambda x: "; ".join(x),
                                                        'institute_name': lambda x: "; ".join(x)}).sort_values(by='n_publications', ascending=False)

    # count number of institutes for each state
    us_states['n_institutes'] = us_states['institute_name'].apply(lambda x: len(set(x.split(';'))))

    # get top states
    top_us_states = us_states.head(20)

    # plot
    _, ax = plt.subplots()
    labels = [state for state in top_us_states.index]
    ax.barh(top_us_states.index, top_us_states['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f'Top {top_n} US states with most publications')
    ax.invert_yaxis()
    plt.savefig(output_file.replace(".png", "_US.png"), dpi=600, bbox_inches="tight")


def plot_funding_sponsors(funding_df: pd.DataFrame,
                          top_n: int = 20,
                          output_file: str = None):
    """
    Plot the top n funding sponsors (considdering the number of publications)
    """
    # sort sponsors for number of publications
    funding_df = funding_df.sort_values(by='n_publications', ascending=False).dropna(subset=['xocs:funding-agency', 'xocs:funding-agency-country'])

    # get top n sponsors
    top_sponsors = funding_df.head(top_n)


    # plot
    _, ax = plt.subplots()
    labels = [f"{s_name} ({s_country.split('/')[-2]})" for s_name, s_country in zip(top_sponsors['xocs:funding-agency'], top_sponsors["xocs:funding-agency-country"])]
    ax.barh(top_sponsors['xocs:funding-agency'], top_sponsors['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(f'Top {top_n} sponsors with most publications')
    ax.invert_yaxis()
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_keywords(author_keywords_df: pd.DataFrame,
                  indexed_keywords_df: pd.DataFrame,
                  top_n: int = 20,
                  output_file: str = None):
    """
    Plot the top n keywords
    """
    # get top n author keywords
    top_author_keywords = author_keywords_df.sort_values('n_publications', ascending=False).head(top_n)
    top_indexed_keywords = indexed_keywords_df.sort_values('n_publications', ascending=False).head(top_n)

    # plotkeywords
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for df, ax, keyword_type in zip([top_author_keywords, top_indexed_keywords], axes, ['author_keyword', 'Index Keywords']):
        ax.barh(df[keyword_type], df['n_publications'])
        ax.grid(axis='x')
        ax.set_xlabel('Number of publications')
        ax.set_yticklabels(df[keyword_type], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"Top {top_n} {keyword_type.replace('_', ' ')}")
    fig.tight_layout()
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_modeling_methods(modeling_approaches_df: pd.DataFrame,
                          modeling_approaches_dict: dict,
                          output_file: str = None):
    """
    Plot the modeling approaches
    """
    # generate dictionary for broad approches
    general_approaches = list(modeling_approaches_dict.keys())
    general_approaches_counts = Counter(general_approaches)
    for ga in general_approaches:
        for approach in modeling_approaches_dict[ga]:
            for _, row in modeling_approaches_df.iterrows():
                if re.search(approach, row["modeling_approach"], re.IGNORECASE):
                    general_approaches_counts[ga] += row["n_matches_tot"] 

    # plot
    _, ax = plt.subplots()
    ax.barh(list(general_approaches_counts.keys()), list(general_approaches_counts.values()))
    ax.set_xlabel('Number of matches (titles / abstracts)')
    ax.set_title('Modeling approaches')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")


def plot_quartiles(publications_quartile_df: pd.DataFrame,
                   output_file: str = None):
     """
     Plot the number of publications per quartile
     """
     # count number of publications per quartile
     n_publications = publications_quartile_df.groupby('quartile').count()["title"]

     _, ax = plt.subplots()
     ax.bar(n_publications.index, n_publications)
     ax.set_xlabel('Quartile')
     ax.set_ylabel('Number of publications')
     ax.set_title('Number of publications per quartile')
     plt.savefig(output_file, dpi=600, bbox_inches="tight")


def generate_plots(data_folder: str,
                   scopus_csv: str, 
                   output_folder: str = None):
    # Init oputput folder
    if output_folder is None:
        output_folder_plot = Path(f"{data_folder}/plot")
    else:
        output_folder_plot = Path(output_folder)
    output_folder_plot.mkdir(exist_ok=True, parents=True)
   
    # generate journals plot
    journals_df = pd.read_csv(f"{data_folder}/journals.csv")
    output_j_plot = f"{data_folder}/plot/journals_plot.png"
    plot_journals(journals_df, output_file=output_j_plot)

    # generate subject areas plot
    subject_areas_df = pd.read_csv(f"{data_folder}/subjects.csv")
    output_s_plot = f"{data_folder}/plot/subjects_plot.png"
    plot_subject_areas(subject_areas_df, output_file=output_s_plot)

    # generate authors plot
    authors_df = pd.read_csv(f"{data_folder}/authors.csv")
    institutions_df = pd.read_csv(f"{data_folder}/institutions.csv")
    output_a_plot = f"{data_folder}/plot/authors_plot.png"
    plot_authors(authors_df, institutions_df, output_file=output_a_plot)

    # generate citations histogram plot
    scopus_df = pd.read_csv(scopus_csv)
    output_c_plot = f"{data_folder}/plot/citations_histogram.png"
    plot_citations(scopus_df, output_file=output_c_plot)

    # generate institutions plot
    institutions_df = pd.read_csv(f"{data_folder}/institutions.csv")
    output_i_plot = f"{data_folder}/plot/institutions_plot.png"
    plot_institutions(institutions_df, output_file=output_i_plot)

    # generate countries and states plot
    output_co_plot = f"{data_folder}/plot/countries_plot.png"
    plot_countries(institutions_df, output_file=output_co_plot)

    # generate funding sponsors plot
    funding_df = pd.read_csv(f"{data_folder}/funding_sponsors.csv")
    output_f_plot = f"{data_folder}/plot/funding_sponsors_plot.png"
    plot_funding_sponsors(funding_df, output_file=output_f_plot)

    # generate keywords plot
    author_keywords_df = pd.read_csv(f"{data_folder}/author_keywords.csv")
    indexed_keywords_df = pd.read_csv(f"{data_folder}/indexed_keywords.csv")
    output_k_plot = f"{data_folder}/plot/keywords_plot.png"
    plot_keywords(author_keywords_df, indexed_keywords_df, output_file=output_k_plot)

    # generate modeling approaches plot
    modeling_approaches_df = pd.read_csv(f"{data_folder}/modeling_approaches.csv")
    with open(f"data/methods.json", 'r') as f:
        modeling_approaches_dict = json.load(f)
    output_modeling_plot = f"{data_folder}/plot/modeling_approaches_plot.png"
    plot_modeling_methods(modeling_approaches_df, modeling_approaches_dict, output_file=output_modeling_plot)

    # get quartiles plot
    publications_quartile_df = pd.read_csv(f"{data_folder}/publications_quartile.csv")
    output_q_plot = f"{data_folder}/plot/quartiles_plot.png"
    plot_quartiles(publications_quartile_df, output_file=output_q_plot)


if __name__ == "__main__":
    main()
    

    
