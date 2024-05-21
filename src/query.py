import re
import json
import logging
from typing import List
from pathlib import Path
from collections import Counter
import requests
import pandas as pd
from tqdm import tqdm
from elsapy.elsclient import ElsClient
from elsapy.elsprofile import ElsAuthor, ElsAffil
from elsapy.elsdoc import FullDoc, AbsDoc
from elsapy.elssearch import ElsSearch

# logging
logging.basicConfig(level=logging.INFO)

# folders
OUTPUT_FOLDER = Path("out")
OUTPUT_FOLDER.mkdir(exist_ok=True, parents=True)
 

# APIs
scoups_titles_base_url_issn = "https://api.elsevier.com/content/serial/title/issn/"  # necessary to get journal metadata
scopus_api_abstract_base_url = "https://api.elsevier.com/content/abstract/doi/"  # necessary to get abstract metadata
scival_metrics_url = "https://api.elsevier.com/analytics/scival/scopusSource/metrics"  # necessary to get journal metrics


def search_publications(search_string: str, 
                        config: dict, 
                        client: ElsClient,
                        publications_file: Path,
                        run_search: bool = False,
                        get_all: bool = False) -> pd.DataFrame:
    """
    Generate table with all MathOnco publications.
    """
    # check if is necessary to run search
    if publications_file.exists() and not run_search:
        return pd.read_csv(publications_file)
    
    # Search MathOnco publications
    doc_srch = ElsSearch(search_string, 'scopus')
    doc_srch.execute(client, get_all = get_all)

    # Iterate over publications
    output_table = []  # init output table as list
    for res_data in doc_srch.results:
        # init dict for the current record
        current_res_dict = {}  # init dict for the current record

        # get publication identifiers
        current_res_dict["scopus_id"] = res_data["dc:identifier"].replace("SCOPUS_ID:", "")
        current_res_dict["doi"] = res_data["prism:doi"] if "prism:doi" in res_data.keys() else None
        current_res_dict["title"] = res_data["dc:title"]

        # get number of citations
        current_res_dict["n_citations"] = res_data["citedby-count"]

        # get journal name
        current_res_dict["journal_name"] = res_data["prism:publicationName"]

        # get journal ISSN
        if "prism:issn" in res_data.keys():
            journal_issn = res_data['prism:issn']
        elif "prism:eIssn" in res_data.keys():
            journal_issn = res_data['prism:eIssn']
        else:
            journal_issn = None
        current_res_dict["journal_issn"] = journal_issn

        # append to table
        output_table.append(current_res_dict)
    
    # Save table
    output_df = pd.DataFrame(output_table)
    output_df.to_csv(publications_file, index=False)

    # return table
    return output_df


def generate_journals_csv(scopus_df: pd.DataFrame,
                          journals_file: Path,
                          config_files: List[dict],
                          run_search: bool = False,
                          req_limit: int = 19950) -> pd.DataFrame:
    """
    Get the journals (sources) where the publications were published.
    """
    # check if is necessary to run search
    if journals_file.exists() and not run_search:
        return pd.read_csv(journals_file)
    
    # get issn for each journal
    journals_issns = scopus_df["ISSN"]

    # get n publications for each journal
    journals_counts = journals_issns.value_counts().reset_index()
    journals_counts.columns = ["journal_issn", "n_publications"]

    # get unique values for journal issn
    journals_issns_unique = journals_counts["journal_issn"]

    # given that only a certain number of requests can be made, we will get the abstracts in chunks
    n_issns = len(journals_issns_unique)
    issns_chunks = [journals_issns_unique[i:i+req_limit] for i in range(0, n_issns, req_limit)]
    if len(issns_chunks) > len(config_files):
        logging.warning(f"Number of chunks ({len(issns_chunks)}) is greater than the number of available API keys ({len(config_files)}). Some chunks will not be processed")
        issns_chunks = issns_chunks[:len(config_files)]
    else:
        config_files = config_files[:len(issns_chunks)]

    # init output table
    output_df = pd.DataFrame()

    # get metadata for each journal
    for issns_chunk, config in zip(issns_chunks, config_files):
        for j_issn in tqdm(issns_chunk, desc="Getting journal metadata"):
            # query API for journal metadata
            journal_info = requests.get(url=f"{scoups_titles_base_url_issn}{j_issn}",
                                        params={"apiKey": config["apikey"]}).json()
            
            # If journal metadata not found, log and skip
            if "serial-metadata-response" not in journal_info.keys():
                logging.warning(f"Journal metadata not found for journal ISSN {j_issn}. I found the following keys: {journal_info.keys()}. Skipping.")
                continue
            journal_info = journal_info["serial-metadata-response"]["entry"][0]

            # init current record for journal
            current_res_dict = {}

            # get journal name
            current_res_dict["journal_issn"] = j_issn
            current_res_dict["scopus_id"] = journal_info["source-id"]
            current_res_dict["title"] = journal_info["dc:title"]

            # get journal citescore
            if "citeScoreYearInfoList" not in journal_info.keys():
                logging.warning(f'CiteScore information not found for journal ISSN {j_issn}. I found the following keys: {journal_info.keys()}. Skipping.')
                continue
            if "citeScoreCurrentMetric" not in journal_info["citeScoreYearInfoList"].keys():
                logging.warning(f'CiteScore information not found for journal ISSN {j_issn}. I found: {journal_info["citeScoreYearInfoList"].keys()}. Skipping.')
                continue
            current_res_dict["journal_citescore"] = journal_info["citeScoreYearInfoList"]["citeScoreCurrentMetric"]

            # get journal subjects
            current_res_dict[f"subject_code(s)"] = "; ".join([sa["@code"] for sa in journal_info["subject-area"]])
            current_res_dict[f"subject_abbrev(s)"] = "; ".join([sa["@abbrev"] for sa in journal_info["subject-area"]])
            current_res_dict[f"subject_name(s)"] = "; ".join([sa["$"] for sa in journal_info["subject-area"]])

            # get open access info
            current_res_dict["openaccess"] = journal_info["openaccess"]
            current_res_dict["openaccessType"] = journal_info["openaccessType"]

            # generate table
            current_res_df = pd.DataFrame([current_res_dict])

            # append to output table
            output_df = pd.concat([output_df, current_res_df], ignore_index=True)

    # join number of publications to output table
    output_df = output_df.merge(journals_counts, on="journal_issn", how="left")

    # save table
    output_df.to_csv(journals_file, index=False)

    # return table
    return output_df


def search_abstracts(config_files: List[dict],
                     scopus_csv: pd.DataFrame,
                     abstracts_file: Path,
                     run_search: bool = False,
                     req_limit: int = 9950) -> List[AbsDoc]:
    """
    Get and read AbsDocs.

    :param config: configuration dictionary [NOT USED].
    :param client: ElsClient instance.
    :param publications_df: DataFrame with publications metadata.
    :param abstracts_file: Path to save the abstracts.
    :param run_search: bool to run the search or not.
    """
    # check if is necessary to run search
    if not run_search and abstracts_file.exists():
        with open(abstracts_file, "r") as infile:
            return json.load(infile)
        
    # get list of scopus ids
    dois = scopus_csv["DOI"].dropna().tolist()

    # given that only a certain number of requests can be made, we will get the abstracts in chunks
    n_dois = len(dois)
    dois_chunks = [dois[i:i+req_limit] for i in range(0, n_dois, req_limit)]
    if len(dois_chunks) > len(config_files):
        logging.warning(f"Number of chunks ({len(dois_chunks)}) is greater than the number of available API keys ({len(config_files)}). Some chunks will not be processed")
        dois_chunks = dois_chunks[:len(config_files)]
    else:
        config_files = config_files[:len(dois_chunks)]

    # get AbsDocs
    output_list = []
    for doi_chunk, config in zip(dois_chunks, config_files):
        for doi in tqdm(doi_chunk, desc="Getting abstracts"):
            # get abstract metadata
            scp_abstract = requests.get(url=f"{scopus_api_abstract_base_url}{doi}",
                                        params={"apiKey": config["apikey"], 
                                                "httpAccept": "application/json", 
                                                "view": "FULL",
                                                "": ""})
            if scp_abstract.status_code != 200:
                logging.warning(f"Read document {doi} failed for reason: {scp_abstract.json()}. Headers are: {scp_abstract.headers}. Skipping.")
                continue
            
            scp_abstract = scp_abstract.json()  # get metadata

            # Append to output list
            output_list.append(scp_abstract)

    # Chache content to json
    with open(abstracts_file, "w") as outfile:
        json.dump(output_list, outfile, indent=4)

    # Return list of AbsDocs data
    return output_list


def get_authors(publications_list: List[dict],
                authors_file: Path) -> pd.DataFrame:
    """
    From the publications table, get institution metadata. Store the results in a table called out/institutions.csv.
    """
    # iterate over documents
    output_df = pd.DataFrame()  # init output table
    for scp_abstract in tqdm(publications_list, desc="Getting authors"):
        # init list of dicts for the current record
        current_res_list = []

        # get author information
        if ("authors" not in scp_abstract["abstracts-retrieval-response"].keys()) or (scp_abstract["abstracts-retrieval-response"]["authors"] is None):
            logging.warning("No author information found. Skipping.")
            continue
        
        for author in scp_abstract["abstracts-retrieval-response"]["authors"]["author"]:
            current_res_dict = {}  # init dict for the current record
            current_res_dict[f"scopus_id"] = author["@auid"]
            current_res_dict[f"author_indexed-name"] = author["ce:indexed-name"]
            if "affiliation" in author.keys():
                if isinstance(author["affiliation"], list):
                    current_res_dict[f"affiliation_id(s)"] = "; ".join([item["@id"] for item in author["affiliation"]])
                else:
                    current_res_dict[f"affiliation_id(s)"] = author["affiliation"]["@id"]
            current_res_list.append(current_res_dict)
        
        # generate table
        current_df = pd.DataFrame(current_res_list)

        # append to output table
        output_df = pd.concat([output_df, current_df], ignore_index=True)

    # count number of records for each author
    n_publications = output_df.groupby("scopus_id", as_index=False).size()
    n_publications = n_publications.rename(columns={"size": "n_publications"})

    # join number of publications to output table
    output_df = output_df.merge(n_publications, on="scopus_id", how="left").drop_duplicates(subset="scopus_id")

    # save table
    output_df.to_csv(authors_file, index=False)

    # return table
    return output_df


def get_affiliations(publications_list: List[dict],
                     institutions_file: Path,
                     clients_array: List[ElsClient],
                     run_search: bool = False,
                     req_limit=4950) -> pd.DataFrame:
    """
    From the publications table, get institution metadata. Store the results in a table called out/institutions.csv.
    """
    # iterate over documents
    raw_data_list = []          # init raw data list

    # check if is necessary to run search
    if institutions_file.exists() and not run_search:
        return pd.read_csv(institutions_file)

    for scp_abstract_info in tqdm(publications_list, desc="Getting institutions"):
        # extract information
        scp_abstract = scp_abstract_info["abstracts-retrieval-response"]

        # get affiliations info
        if "affiliation" not in scp_abstract.keys():
            logging.warning("No affiliation information found. Skipping.")
            continue

        # transform in list if not list
        if not isinstance(scp_abstract["affiliation"], list):
            scp_abstract["affiliation"] = [scp_abstract["affiliation"]]

        # get info for each affiliation
        for institute in scp_abstract["affiliation"]:
            # init dict for the current record
            current_res_dict = {}                        

            # get preliminary info
            institute_id = institute["@id"]
            current_res_dict[f"scopus_id"] = institute_id
            current_res_dict[f"institute_name"] = institute["affilname"]
            current_res_dict[f"institute_city"] = institute["affiliation-city"]
            current_res_dict[f"institute_country"] = institute["affiliation-country"]

            # append to list
            raw_data_list.append(current_res_dict)
                
    # generat raw data table
    raw_df = pd.DataFrame(raw_data_list)

    # count number of records for each author
    n_publications = raw_df.groupby("scopus_id", as_index=False).size()
    n_publications = n_publications.rename(columns={"size": "n_publications"})

    # join number of publications to output table
    raw_df = raw_df.merge(n_publications, on="scopus_id", how="left").drop_duplicates(subset="scopus_id")
    raw_df.to_csv(institutions_file.parent / Path("institutions_raw.csv"), index=False)

    # get more detailed information for each institution
    req_counter = Counter(clients_array)            # init counters
    current_client_i = 0                            # init current client
    output_data_list = []                           # init output list
    output_json = []                                # init output json
    for _, current_row in tqdm(raw_df.iterrows(), desc="Getting detailed information for institutions"):
        # get client
        client = clients_array[current_client_i]
        if req_counter[client] >= req_limit:
            current_client_i = current_client_i + 1
            if current_client_i >= len(clients_array):
                logging.warning("All clients have reached the request limit. Appending only the info available now.")
                output_data_list.append(current_row.to_dict())
                continue
            else:
                client = clients_array[current_client_i]   
            
        # get more detailed institute info with elsapy
        institution = ElsAffil(affil_id=current_row["scopus_id"])
        if not institution.read(client):
            logging.info(f"Could not retrive info for institution with id {institute_id}. Skipping.")
            output_data_list.append(current_row.to_dict())
            continue
        institute_data = institution.data

        # add to counter
        req_counter[client] += 1

        # extract data
        current_res_dict = current_row.to_dict()
        current_res_dict[f"institute_name"] = institute_data["institution-profile"]["preferred-name"]["$"]  # get prefferred name
        if isinstance(institute_data["institution-profile"]["address"], dict):
            # get detailed address
            institute_address = institute_data["institution-profile"]["address"]
            for key in ["city", "state", "country"]:
                if key in institute_address.keys():
                    current_res_dict[f"institute_{key}"] = institute_address[key]
        output_data_list.append(current_res_dict)

    # save table
    output_df = pd.DataFrame(output_data_list)
    output_df.to_csv(institutions_file, index=False)

    # save json
    with open(institutions_file.parent / Path("institutions.json"), "w") as outfile:
        json.dump(output_json, outfile, indent=4)

    # return table
    return output_df


def get_funding_sponsors(publications_list: List[dict],
                         funding_sponsors_file: str) -> pd.DataFrame:
    """
    From the publications list, get funding sponsors metadata. Store the results in a table called out/funding_sponsors.csv.
    """
    # iterate over documents
    output_df = pd.DataFrame()  # init output table
    for scp_abstract_info in tqdm(publications_list, desc="Getting funding sponsors"):
        # extract information
        scp_abstract = scp_abstract_info["abstracts-retrieval-response"]

        # init list for the current record
        current_res_list = []

        # check if funding information is available
        abstract_info = scp_abstract["item"]
        if "xocs:meta" not in abstract_info.keys():
            logging.warning("No funding information found. Skipping.")
            continue
        if "xocs:funding" not in abstract_info["xocs:meta"]["xocs:funding-list"].keys():
            logging.warning("No funding information found. Skipping.")
            continue

        # if available, get funding information
        abstract_funding_list = abstract_info["xocs:meta"]["xocs:funding-list"]["xocs:funding"]

        # if not list, transform in list
        if not isinstance(abstract_funding_list, list):
            abstract_funding_list = [abstract_funding_list]

        # for each sponsor, add to list
        for funding_sponsor in abstract_funding_list:
            current_res_dict = funding_sponsor
            if "xocs:funding-id" in current_res_dict.keys():
                if isinstance(current_res_dict["xocs:funding-id"], list):
                    funding_id_list = ", ".join([str(item["$"]) for item in current_res_dict["xocs:funding-id"] if item["$"] is not None])
                    current_res_dict["xocs:funding-id"] = funding_id_list
            current_res_list.append(current_res_dict)
                    
        # generate table
        current_df = pd.DataFrame(current_res_list).drop_duplicates()

        # append to output table
        output_df = pd.concat([output_df, current_df], ignore_index=True)

    # count number of publications for each sponsor
    n_publications = output_df.groupby("xocs:funding-agency-acronym", as_index=False).size()
    n_publications = n_publications.rename(columns={"size": "n_publications"})
    output_df = output_df.merge(n_publications, on="xocs:funding-agency-acronym", how="left").drop_duplicates(subset="xocs:funding-agency-acronym")

    # save table
    output_df.to_csv(funding_sponsors_file, index=False)

    # return table
    return output_df


def get_author_keywords(publications_list: List[dict],
                        author_keywords_file: Path) -> pd.DataFrame:
    """
    From the publications list, get keywords metadata. Store the results in a table called out/keywords.csv.
    """
    # iterate over documents to get the Author Keywords
    output_df = pd.DataFrame()  # init output table
    for scp_abstract_info in tqdm(publications_list, desc="Getting author keywords"):
        # extract information
        scp_abstract = scp_abstract_info["abstracts-retrieval-response"]

        # init list for the current record
        current_res_list = []

        # get keywords info
        if ("authkeywords" in scp_abstract.keys()) and (scp_abstract["authkeywords"] is not None):
            abstract_keywords = scp_abstract["authkeywords"]["author-keyword"]
        else:
            continue

        # get current record
        current_res_list = []
        for item in abstract_keywords:
            if isinstance(item, dict):
                current_res_list.append({"author_keyword": item["$"]})
            else:
                current_res_list.append({"author_keyword": item}) 

        # capitalize each keyword
        current_res_list = [{"author_keyword": item["author_keyword"].capitalize()} for item in current_res_list]
        
        # generate table
        current_df = pd.DataFrame(current_res_list).drop_duplicates()

        # append to output table
        output_df = pd.concat([output_df, current_df], ignore_index=True)

    # count number of publications for each keyword
    n_publications = output_df.groupby("author_keyword", as_index=False).size()
    n_publications = n_publications.rename(columns={"size": "n_publications"})

    # join number of publications to output table
    output_df = output_df.merge(n_publications, on="author_keyword", how="left").drop_duplicates(subset="author_keyword")

    # save table
    output_df.to_csv(author_keywords_file, index=False)

    # return table
    return output_df


def get_semicolumn_separated_filed_from_csv(scopus_df: pd.DataFrame,
                                            field_name: str,
                                            output_file: Path) -> pd.DataFrame:
    """
    From the publications table, get a semicolon separated field and count the number of publications for each value.
    """
    # get field
    field = scopus_df[field_name].copy()
    field = field.dropna().map(lambda x: x.split("; ")).explode().map(lambda x: x.capitalize()).reset_index(drop=True)

    # count number of publications for each value
    n_publications = field.value_counts().reset_index()
    n_publications.columns = [field_name, "n_publications"]

    # save table
    n_publications.to_csv(output_file, index=False)

    # return table
    return n_publications


def get_indexed_keywords_from_csv(scopus_df: pd.DataFrame,
                                  indexed_keywords_file: Path) -> pd.DataFrame:
    """
    From the publications table, get indexed keywords metadata. Store the results in a table called out/indexed_keywords.csv.
    """
    return get_semicolumn_separated_filed_from_csv(scopus_df, "Index Keywords", indexed_keywords_file)


def get_author_keywords_from_csv(scopus_df: pd.DataFrame,
                                 author_keywords_file: Path) -> pd.DataFrame:
    """
    From the publications table, get author keywords metadata. Store the results in a table called out/author_keywords.csv.
    """
    return get_semicolumn_separated_filed_from_csv(scopus_df, "Author Keywords", author_keywords_file)
    

def get_subjects_list(journals_file: Path,
                      subjects_file: Path) -> pd.DataFrame:
    """
    """
    # Load journals (with subjects)
    journals_df = pd.read_csv(journals_file)

    # Get the columns containing the subject abbreviations and names
    subject_code_column = "subject_code(s)"
    subject_abbrev_column = "subject_abbrev(s)"
    subject_name_column = "subject_name(s)"
    
    # Iterate over the rows and get the subjects and the number of publications
    records = []
    for _, row in tqdm(journals_df.iterrows(), desc="Getting Journal subjects"):
        # get data
        codes = row[subject_code_column]
        abbrev = row[subject_abbrev_column]
        names = row[subject_name_column]
        n_publications = row["n_publications"]

        # split data
        for code, abbrev, name in zip(codes.split("; "), abbrev.split("; "), names.split("; ")):
            records.append({"code": code, "abbrev": abbrev, "name": name, "n_publications": n_publications})
    
    # aggregate the same subjects
    output_df = pd.DataFrame(records)
    agg_function = {"abbrev": "first", "name": "first", "n_publications": "sum"}
    output_df = output_df.groupby(output_df['code']).aggregate(agg_function)
    
    # save output csv
    output_df.to_csv(subjects_file)
    return output_df


def get_modeling_approaches(scopus_df: pd.DataFrame,
                            modeling_approaches: List[str],
                            modeling_approaches_file: Path) -> pd.DataFrame:
    """
    Get the occcurrence of modeling approaches in the data.
    """
    # init counter object
    title_counter = Counter()
    abstract_counter = Counter()

    # for each title and each abstract in the document, check if the modeling approach is present
    for _, row in tqdm(scopus_df.iterrows(), desc="Getting modeling approaches"):
        # get titles and abstracts
        title = row["Title"]
        abstract = row["Abstract"]
        # get matches
        for approach in modeling_approaches:
            regex_string = f" {approach} "                                           # add spaces to avoid partial matches
            title_matches = re.findall(f" {approach} ", title, re.IGNORECASE)        # get title matches
            abstract_matches = re.findall(f" {approach} ", abstract, re.IGNORECASE)  # get abstract matches
            # lower all matches and remove dashes
            title_matches = [match.lower().replace("–", " ").replace("-", " ") for match in title_matches]
            abstract_matches = [match.lower().replace("–", " ").replace("-", " ") for match in abstract_matches]
            title_counter += Counter(title_matches)
            abstract_counter += Counter(abstract_matches)

    # convert results to dict 
    output_list = []
    tot_counter = title_counter + abstract_counter
    for match, count in tot_counter.items():
        output_list.append({"modeling_approach": match, 
                            "n_matches_tot": count, 
                            "n_matches_title": title_counter[match], 
                            "n_matches_abstract": abstract_counter[match]})

    # save dataframe to csv
    output_df = pd.DataFrame(output_list)
    output_df.to_csv(modeling_approaches_file, index=False)

    return output_df


def search_journals_metrics(config_files: List[dict],
                            journals_df: pd.DataFrame,
                            journals_metrics_JSON: Path,
                            run_search: bool = False,
                            req_limit: int = 4950) -> dict:
    """
    Download journal metrics from SciVal Scopus API.
    """
    # check if is necessary to run search
    if journals_metrics_JSON.exists() and not run_search:
        with open(journals_metrics_JSON, "r") as infile:
            return json.load(infile)
        
    # get list of scopus ids divided in sublists of 25 elements
    scopus_ids = [str(id) for id in journals_df["scopus_id"].tolist()]
    scopus_ids = [scopus_ids[i:i+25] for i in range(0, len(scopus_ids), 25)]

    # subdivide in chunks to meet thr requests limit
    scopus_ids_chunks = [scopus_ids[i:i+req_limit] for i in range(0, len(scopus_ids), req_limit)]
    if len(scopus_ids_chunks) > len(config_files):
        logging.warning(f"Number of chunks ({len(scopus_ids_chunks)}) is greater than the number of available API keys ({len(config_files)}). Some chunks will not be processed")
        scopus_ids_chunks = scopus_ids_chunks[:len(config_files)]
    else:
        config_files = config_files[:len(scopus_ids_chunks)]

    # iterate on sublists
    output_list = []
    for scopus_ids_chunk, config in zip(scopus_ids_chunks, config_files):
        for sublist in tqdm(scopus_ids_chunk, desc="Getting journal metrics"):
            # get journal metrics
            res = requests.get(scival_metrics_url,
                               params={"apiKey": config["apikey"],
                                       "metricTypes": "PublicationsInTopJournalPercentiles,OutputsInTopCitationPercentiles",
                                       "sourceIds": ",".join(sublist),
                                       "yearRange": "10yrs"})
            # append to list
            output_list += res.json()["results"]
    
    # save json
    with open(journals_metrics_JSON, "w") as outfile:
        json.dump(output_list, outfile, indent=4)

    # return dict
    return output_list


def extract_quartile_for_publications(publications_list: List[dict],
                                      journals_metrics_dict: Path,
                                      publications_quartile_file: Path) -> pd.DataFrame:
    """
    Extract the quartile of each publication.
    """
    # init output list
    output_list = []

    # iterate over publications
    for scp_abstract_info in tqdm(publications_list, desc="Getting quartiles"):
        # get information
        scp_abstract = scp_abstract_info["abstracts-retrieval-response"]

        # get bibliographic information
        scp_abstract_bib = scp_abstract["item"]["bibrecord"]

        # get publication title
        scp_abstract_title = scp_abstract_bib["head"]["citation-title"]

        # get publication source
        scp_abstract_source = scp_abstract_bib["head"]["source"]
        source_title = scp_abstract_source["sourcetitle"]
        source_id = int(scp_abstract_source["@srcid"])

        # get publication year
        publication_year = int(scp_abstract_source["publicationdate"]["year"])
        print(publication_year)

        # get quartile
        if publication_year < 2015:
            logging.info(f"Publication {scp_abstract_title} is older than 2014. Skipping.")
            quartile = pd.NA
        else:
            for jm in journals_metrics_dict:
                if jm["source"]["id"] == source_id:
                    metrics = jm["metrics"][0]  # get metrics based on citescore
                    values = metrics["values"]  # get values
                    # if no data are present, retunr NA
                    if 'percentageByYear' not in values[3].keys():
                        quartile = pd.NA
                        break
                    
                    # if data are present, get quartiles
                    values_q1 = values[3]['percentageByYear']       # get quartile 1
                    values_q2 = values[4]['percentageByYear']       # get quartile 2
                    values_q3 = values[5]['percentageByYear']       # get quartile 3
                    
                    # get correct quartile year
                    quartile_year = (publication_year - 1) if publication_year < 2025 else 2023 

                    # check if year is in the quartile
                    if str(quartile_year) not in values_q1.keys():
                        quartile = pd.NA
                        break

                    # if present, get quartile
                    if values_q1[str(quartile_year)] == 100:
                        quartile = "Q1"
                        break
                    if values_q2[str(quartile_year)] == 100:
                        quartile = "Q2"
                        break
                    if values_q3[str(quartile_year)] == 100:
                        quartile = "Q3"
                        break
                    quartile = "Q4"
                    break
                else:
                    quartile = pd.NA
        
        # append to output list
        output_list.append({"title": scp_abstract_title, "journal_title": source_title, "journal_scopus_id": source_id ,"quartile": quartile})
    
    # save to dataframe
    output_df = pd.DataFrame(output_list)

    # save to csv
    output_df.to_csv(publications_quartile_file, index=False)

    return output_df


def extract_bibliographic_data(
    out_folder_name: str,
    scopus_csv: str,
    config_files: List[str],
    modeling_methods_file: str = "data/methods.json",
    run_search: bool = False):
    """
    Extract bibliographic data from Scopus API AND from a Scopus CSV file (used for indexed keywords only).

    :param query: str, search query.
    :param scopus_csv: str, path to the scopus csv file.
    :param config_file: str, path to the configuration file.
    """
    # Generate output folder
    current_output_folder = OUTPUT_FOLDER / Path(out_folder_name)
    current_output_folder.mkdir(exist_ok=True, parents=True)
    scopus_csv = pd.read_csv(scopus_csv)

    # Initialize output files
    journals_file = current_output_folder / Path("journals.csv")
    abstracts_file = current_output_folder / Path("abstracts.json")
    authors_file = current_output_folder / Path("authors.csv")
    institutions_file = current_output_folder / Path("institutions.csv")
    funding_sponsors_file = current_output_folder / Path("funding_sponsors.csv")
    author_keywords_file = current_output_folder / Path("author_keywords.csv")
    indexed_keywords_file = current_output_folder / Path("indexed_keywords.csv")
    subjects_file = current_output_folder / Path("subjects.csv")
    modeling_approaches_file = current_output_folder / Path("modeling_approaches.csv")
    journals_metrics_json = current_output_folder / Path("journals_metrics.json")
    publications_quartile_file = current_output_folder / Path("publications_quartile.csv")

    # Load configuration
    config_array = []
    clients_array = []
    for config_file in config_files:
        with open(config_file, "r") as infile:
            config = json.load(infile)
            config_array.append(config)
            current_client = ElsClient(config['apikey'])
            current_client.inst_token = config['insttoken']
            clients_array.append(current_client)

    # Get list of abstracts
    publications_list = search_abstracts(config_array, scopus_csv, abstracts_file=abstracts_file, run_search=False)

    # Search journals
    _ = generate_journals_csv(scopus_csv, journals_file, config_array, run_search=False)

    # Search authors
    _ = get_authors(publications_list, authors_file=authors_file)

    # Search affiliations
    _ = get_affiliations(publications_list, institutions_file=institutions_file, clients_array=clients_array, run_search=False)

    # Searc funding sponsors
    _ = get_funding_sponsors(publications_list, funding_sponsors_file=funding_sponsors_file)

    # Search keywords
    _ = get_author_keywords(publications_list, author_keywords_file=author_keywords_file)
    _ = get_indexed_keywords_from_csv(scopus_csv, indexed_keywords_file=indexed_keywords_file)

    # Get subjects
    _ = get_subjects_list(journals_file, subjects_file)

    # Get modeling approaches thoruout regex search in titles and abstracts
    with open(modeling_methods_file, "r") as infile:
        modeling_approaches_dict = json.load(infile)
    modeling_approaches_list = sum(modeling_approaches_dict.values(), [])
    _ = get_modeling_approaches(scopus_csv, modeling_approaches=modeling_approaches_list, modeling_approaches_file=modeling_approaches_file)

    # Get journal metrics
    journal_metrics = search_journals_metrics(config_array, journals_df=pd.read_csv(journals_file), journals_metrics_JSON=journals_metrics_json, run_search=False)

    # Get quartile for each publication
    _ = extract_quartile_for_publications(publications_list, journal_metrics, publications_quartile_file)

 
if __name__ == "__main__":
    pass
