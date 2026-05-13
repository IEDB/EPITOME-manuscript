Pipeline as below:

1. Run digest_papers.py on a directory containing PDFs. This writes to a separate directory, creating a new directory for each PDF name ("name.pdf" --> output/name) containing context.p.
2. Run entity_extraction.py to identify peptides, MHC molecules and assays from context.p, creating peptide_data, hla_data, and assay_data objects.
3. Run peptide_queries.py to ask questions about each peptide.
