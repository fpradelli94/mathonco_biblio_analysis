import re
import json
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CMAP = plt.get_cmap('tab10')


def plot_journals(journals_df: pd.DataFrame,
                  scopus_df: pd.DataFrame,
                  top_n: int = 20,
                  output_file: str = None):
    """
    Plot the top n journals with the most publications. 
    For those journals, also plot the number of documents in time

    """
    # sort journals for number of publications
    journals_df = journals_df.sort_values(by='n_publications', ascending=False)

    # get top n journals
    top_journals = journals_df.head(top_n)

    # plot top journals
    _, ax = plt.subplots()
    labels = [f"{journal} ({citescore})" for journal, citescore in zip(top_journals['title'], top_journals['journal_citescore'])]
    ax.barh(top_journals['title'], top_journals['n_publications'])
    ax.set_xlabel('Number of publications')
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_title(f'Top {top_n} journals with most publications')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")

    # plot number of documents in time
    fig, ax = plt.subplots(top_n, 1, figsize=(10, 20), sharex=True, gridspec_kw={'hspace': 0.05})
    for i, journal in enumerate(top_journals['title']):
        if i == 0:
            ax[i].set_title('Number of documents in time')
        journal_df = scopus_df[scopus_df['Source title'] == journal]
        journal_df = journal_df.groupby('Year').count()['Title']
        ax[i].plot(journal_df.index, journal_df, label=journal, color=CMAP(0))
        ax[i].legend(loc='upper left', fontsize=8)
        ax[i].grid(axis='x', linestyle='--', alpha=0.7)
    ax[-1].set_xlabel('Year')
    fig.text(0.04, 0.5, 'Number of documents', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(output_file.replace(".png", "_time.png"), dpi=600, bbox_inches="tight")

    # plot number of documents in time accordingly to increasing or decreasing interest
    fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.05})
    for i, journal in enumerate([j for j in top_journals['title'] if j in ['Journal of Mathematical Biology', 'PLoS Computational Biology', 'Scientific Reports']]):
        if i == 0:
            ax[i].set_title('Number of documents in time')
        journal_df = scopus_df[scopus_df['Source title'] == journal]
        journal_df = journal_df.groupby('Year').count()['Title']
        ax[i].plot(journal_df.index, journal_df, label=journal, color=CMAP(0))
        ax[i].legend(loc='upper left', fontsize=8)
        ax[i].grid(axis='x', linestyle='--', alpha=0.7)
    ax[-1].set_xlabel('Year')
    fig.text(0.04, 0.5, 'Number of documents', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(output_file.replace(".png", "_time_increasing.png"), dpi=600, bbox_inches="tight")

    fig, ax = plt.subplots(3, 1, sharex=True, gridspec_kw={'hspace': 0.05})
    for i, journal in enumerate([j for j in top_journals['title'] if j in ['PLoS ONE', 'Cancer Research', 'Proceedings of the National Academy of Sciences of the United States of America']]):
        if i == 0:
            ax[i].set_title('Number of documents in time')
        journal_df = scopus_df[scopus_df['Source title'] == journal]
        journal_df = journal_df.groupby('Year').count()['Title']
        if journal == 'Proceedings of the National Academy of Sciences of the United States of America':
            journal = 'PNAS'
        ax[i].plot(journal_df.index, journal_df, label=journal, color=CMAP(1))
        ax[i].legend(loc='upper left', fontsize=8)
        ax[i].grid(axis='x', linestyle='--', alpha=0.7)
    ax[-1].set_xlabel('Year')
    fig.text(0.04, 0.5, 'Number of documents', va='center', rotation='vertical')
    fig.tight_layout(rect=[0.05, 0.05, 0.95, 0.95])
    plt.savefig(output_file.replace(".png", "_time_decreasing.png"), dpi=600, bbox_inches="tight")

    


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
    Plot citation counts. It generates three plots:
    - A histogram of the number of citations
    - The cumulative number of citations in time
    - The number of citations comparing the different open access modalities

    """
    # Plot histogram of number of citations
    _, ax = plt.subplots()
    ax.hist(scopus_df['Cited by'], bins=100, color=CMAP(0))
    ax.axvline(scopus_df['Cited by'].mean(), color=CMAP(1), linestyle='dashed', linewidth=1, label=f'Mean (= {scopus_df["Cited by"].mean():.2f})')
    ax.axvline(scopus_df['Cited by'].median(), color=CMAP(2), linestyle='dashed', linewidth=1, label=f'Median (= {scopus_df["Cited by"].median():.2f})')
    ax.legend()
    print(f"Mean number of citations: {scopus_df['Cited by'].mean()}")
    ax.set_yscale('log')
    ax.set_xlabel('Number of citations')
    ax.set_ylabel('Number of publications')
    ax.set_title('Number of citations per publication')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")

    # Plot the cumulative number of citations in time
    _, ax = plt.subplots()
    cumulative_citations = pd.DataFrame()
    cumulative_citations['Year'] = scopus_df['Year'].astype(int)
    cumulative_citations['Cited by'] = scopus_df['Cited by'].fillna(0).astype(int)
    cumulative_citations = cumulative_citations.sort_values(by='Year')
    cumulative_citations['Cumulative citations'] = cumulative_citations['Cited by'].cumsum()
    ax.plot(cumulative_citations['Year'], cumulative_citations['Cumulative citations'], color=CMAP(0))
    ax.set_xlabel('Year')
    ax.set_ylabel('Cumulative number of citations')
    ax.set_title('Cumulative number of citations in time')
    plt.savefig(output_file.replace(".png", "_cumulative.png"), dpi=600, bbox_inches="tight")

    # Complete the data about open access
    open_access_df = scopus_df[['Open Access', 'Cited by', 'Title']].copy()  # create a new dataframe
    open_access_df['Open Access'] = open_access_df['Open Access'].str.split('; ')  # split the open access categories
    open_access_df = open_access_df.explode('Open Access')  # expand the dataframe
    open_access_df['Open Access'] = open_access_df['Open Access'].fillna("No Open Access")  # Fill NaN values
    open_access_citations = open_access_df.groupby('Open Access').agg({'Cited by': ['mean', 'sum'], 'Title': 'count'}).reset_index()
    open_access_citations.columns = ['Open Access', 'Mean', 'Sum', 'Count']
    open_access_citations = open_access_citations.sort_values(by='Mean')
    open_access_citations['Open Access'] = open_access_citations['Open Access'].map(lambda x: x.replace("All Open Access; ", ""))

    # Plot citation histogram with std 
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    # Compare Open Access VS. No Open Access
    ax[0].bar(open_access_citations.loc[open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Open Access'], 
              open_access_citations.loc[open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Mean'], color=CMAP(0))
    ax[0].set_xlabel('Open Access VS. No Open Access')
    ax[0].set_ylabel('Mean number of citations')
    ax[0].set_xticks(range(len(open_access_citations.loc[open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Open Access'])))
    ax[0].set_xticklabels([label.replace("All", "").replace("No ", "No\n") for label in open_access_citations.loc[open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Open Access']], fontsize=8)

    # Compare different Open Access types
    ax[1].bar(open_access_citations.loc[~open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Open Access'], 
              open_access_citations.loc[~open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Mean'], color=CMAP(0))
    ax[1].set_xlabel('Different Open Access Types')
    ax[1].set_ylabel('Mean number of citations')
    ax[1].set_xticks(range(len(open_access_citations.loc[~open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Open Access'])))
    ax[1].set_xticklabels([label.replace("Open Access", "").replace(" ", "\n") for label in open_access_citations.loc[~open_access_citations['Open Access'].isin(['All Open Access', 'No Open Access'])]['Open Access']], fontsize=8)
    fig.tight_layout()
    plt.savefig(output_file.replace(".png", "_open_access.png"), dpi=600)


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
    labels = [f"{s_name}" for s_name in top_sponsors['xocs:funding-agency']]
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


def plot_regex_matches(scopus_df: pd.DataFrame,
                       expression_json: Path,
                       plot_title: str = "Title",
                       output_file: str = None):
    """
    Plot the number of matches for a given regular expression in documents titles, abstracts, and keywords
    """
    # load expression
    with open(expression_json, 'r') as f:
        expression = json.load(f)

    # create counter for general categories
    general_categories = list(expression.keys())
    general_categories_counts = Counter(general_categories)
    expression_counters = {gc: Counter(expression[gc]) for gc in general_categories}
    for gc in tqdm(general_categories):
        for exp in expression[gc]:
            for _, row in scopus_df.iterrows():
                if re.search(exp, str(row["Title"]), re.IGNORECASE) or \
                   re.search(exp, str(row["Abstract"]), re.IGNORECASE) or \
                   re.search(exp, str(row["Author Keywords"]), re.IGNORECASE):
                    general_categories_counts[gc] += 1
                    expression_counters[gc][exp] += 1
        expression_counters[gc]['total'] = general_categories_counts[gc]
    
    # plot
    _, ax = plt.subplots()
    ax.barh(list(general_categories_counts.keys()), list(general_categories_counts.values()))
    ax.set_xlabel('Number of matches (titles / abstracts / keywords)')
    ax.set_title(plot_title)
    plt.savefig(output_file, dpi=600, bbox_inches="tight")

    # save expression counters as json
    with open(output_file.replace(".png", ".json"), 'w') as f:
        json.dump(expression_counters, f, indent=4)


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


def plot_n_documents_in_time(scopus_df: pd.DataFrame,
                             output_file: str = None):
    """
    Plot the number of documents in time
    """
    # get number of documents per year
    documents_per_year = scopus_df.groupby('Year').count()['Title']

    # plot
    _, ax = plt.subplots()
    ax.plot(documents_per_year.index, documents_per_year, color=CMAP(0))
    ax.set_xlabel('Year')
    ax.set_ylabel('Number of documents')
    ax.set_title('Number of documents in time')
    plt.savefig(output_file, dpi=600, bbox_inches="tight")

    # compare to publications in general and cancer
    # load data
    general_n_papers_csv = Path("data/n_docs_per_year/n_docs_per_year_GENERAL.csv")
    cancer_n_papers_csv = Path("data/n_docs_per_year/n_docs_per_year_CANCER.csv")
    general_n_papers = pd.read_csv(general_n_papers_csv)
    cancer_n_papers = pd.read_csv(cancer_n_papers_csv)

    # Normalize the number of documents on the year 1970
    documents_per_year = documents_per_year.reset_index()
    normalizing_year = 1975
    documents_per_year['Normalized'] = documents_per_year['Title'] / documents_per_year.loc[documents_per_year['Year'] == normalizing_year, 'Title'].values[0]
    general_n_papers['Normalized'] = general_n_papers['N_DOCS'] / general_n_papers.loc[general_n_papers['YEAR'] == normalizing_year, 'N_DOCS'].values[0]
    cancer_n_papers['Normalized'] = cancer_n_papers['N_DOCS'] / cancer_n_papers.loc[cancer_n_papers['YEAR'] == normalizing_year, 'N_DOCS'].values[0]

    # Get only documents after the first MathOnco paper
    documents_per_year = documents_per_year.loc[documents_per_year['Year'] >= documents_per_year['Year'].min()]
    general_n_papers = general_n_papers.loc[general_n_papers['YEAR'] >= documents_per_year['Year'].min()]
    cancer_n_papers = cancer_n_papers.loc[cancer_n_papers['YEAR'] >= documents_per_year['Year'].min()]

    # Plot comparison
    _, ax = plt.subplots()
    ax.plot(documents_per_year['Year'], documents_per_year['Normalized'], color=CMAP(0), label='Mathematical Oncology')
    ax.plot(general_n_papers['YEAR'], general_n_papers['Normalized'], color=CMAP(1), label='General (Article + Reviews)')
    ax.plot(cancer_n_papers['YEAR'], cancer_n_papers['Normalized'], color=CMAP(2), label='Cancer (Article + Reviews)')
    ax.set_xlabel('Year')
    ax.set_ylabel(f'Number of documents (normalized at {normalizing_year})')
    ax.set_title('Number of documents in time')
    ax.legend()
    plt.savefig(output_file.replace(".png", "_comparison.png"), dpi=600, bbox_inches="tight")


def generate_plots(data_folder: str,
                   scopus_csv: str, 
                   output_folder: str = None):
    # Init oputput folder
    if output_folder is None:
        output_folder_plot = Path(f"{data_folder}/plot")
    else:
        output_folder_plot = Path(output_folder)
    output_folder_plot.mkdir(exist_ok=True, parents=True)

    # Load scorpus data
    scopus_df = pd.read_csv(scopus_csv)
   
    # generate journals plot
    journals_df = pd.read_csv(f"{data_folder}/journals.csv")
    output_j_plot = f"{data_folder}/plot/journals_plot.png"
    plot_journals(journals_df, scopus_df, output_file=output_j_plot)

    # generate subject areas plot
    subject_areas_df = pd.read_csv(f"{data_folder}/subjects.csv")
    output_s_plot = f"{data_folder}/plot/subjects_plot.png"
    plot_subject_areas(subject_areas_df, output_file=output_s_plot)

    # generate authors plot
    authors_df = pd.read_csv(f"{data_folder}/authors.csv")
    institutions_df = pd.read_csv(f"{data_folder}/institutions.csv")
    output_a_plot = f"{data_folder}/plot/authors_plot.png"
    plot_authors(authors_df, institutions_df, output_file=output_a_plot)

    # generate citations plots
    plot_citations(scopus_df, output_file=f"{data_folder}/plot/citations_histogram.png")

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

    # generate matches plot
    expression_json = Path("data/methods.json")
    plot_regex_matches(scopus_df, expression_json, plot_title="Modeling approaches", output_file=f"{data_folder}/plot/modeling_approaches_plot.png")
    expression_json = Path("data/cancer_types.json")
    plot_regex_matches(scopus_df, expression_json, plot_title="Cancer types", output_file=f"{data_folder}/plot/cancer_types_plot.png")

    # get quartiles plot
    publications_quartile_df = pd.read_csv(f"{data_folder}/publications_quartile.csv")
    output_q_plot = f"{data_folder}/plot/quartiles_plot.png"
    plot_quartiles(publications_quartile_df, output_file=output_q_plot)

    # Generate plot for documents in time
    plot_n_documents_in_time(scopus_df, 
                             output_file=f"{data_folder}/plot/n_documents_in_time.png")


def _generate_plots(data_folder: str,
                   scopus_csv: str, 
                   output_folder: str = None):
    # _generate_plots(data_folder, scopus_csv, output_folder)
    # generate citations histogram plot
    #with open

    # save top 20 papers by citations
    top_20_papers = scopus_df.sort_values(by='Cited by', ascending=False).head(20)
    top_20_papers.to_csv(f"{data_folder}/top_20_papers.csv", index=False)
    

if __name__ == "__main__":
    pass
    

    
