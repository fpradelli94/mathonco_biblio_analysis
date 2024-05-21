# Resources
- Elsevier Developer Portal: https://dev.elsevier.com/
- Scoups Search Tips: https://dev.elsevier.com/sc_search_tips.html
- `elsapy` (Python wrapper for Scopus API): https://github.com/ElsevierDev/elsapy

# Methods
The `main.py` script runs several API queries. 

## Publications
The first aims at finding all the publications regarding MathOnco and sotres the results in `out/publications.csv`. The important features are:
1. Scopus ID
2. DOI
3. Title
4. Number of Citations
5. Journal
6. Journal ISSN

## Journals
The second search aims at finding all journals where the MathOnco papers have been published. The result is stored in `out/journals.csv`. Important features are:
1. Journal ISSN
2. Journal Scopus Identifier
3. Journal Name
4. Citescore
5. Journal Subjects
6. Number of papers found on the journal

## Authors
The third search regards the authors working at MathOnco. The results are stored in `out/authors.csv`. Important features are:
1. Scopus ID
2. Name 
3. Surname
4. Number of papers found for authors

## Institutions
Institution publishing MathOnco. The results are stored in `out/institutions.csv`. Important features are:
1. Scopus ID
2. Affiliation Name
3. Affiliation Country
4. Affiliation City
5. Number of papers for each affiliation

## Funding sponsors
Funding sponsors for MathOnco research. The results are stored in `out/funding_sponsors.csv`. Important features are:
1. Funding sponsor name
2. Number of papers associated to the sponsor

## Citing Journals
Journals citing mathonco papers. The results are stored in `out/citing_journals.csv`. CURRENTLY NOT WOKING (Autentication Error).

## Keywords
Keywods selected by the authors writing MathOnco papers. The results are stored into `out/keywords.csv`.

## Text Mining


