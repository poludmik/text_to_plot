# from langchain.document_loaders import PyMuPDFLoader

# loader = PyMuPDFLoader("pdf_manuals/Porsche-Taycan-Manual.pdf")
# data = loader.load()
# print(data[0])

################

import camelot

tables = camelot.read_pdf("pdf_manuals/Porsche-Taycan-Manual.pdf")
# tables.export('foo.csv', f='csv', compress=True)
print(tables[0].df)
print(tables[0].parsing_report)

#####################

# import tabula

# Read pdf into list of DataFrame
# dfs = tabula.read_pdf("pdf_manuals/Porsche-Taycan-Manual_removed_avepdf_bb_v.pdf", pages='1') # page '271' is 269 on manual pdf pages (and apparently it indexes from 1...)
# # dfs = tabula.read_pdf("pdf_manuals/bobcat_removed.pdf", pages='19')
# print(dfs[0])
# print()
# print()
# print(dfs[1])

# convert PDF into CSV file
# tabula.convert_into("pdf_manuals/Porsche-Taycan-Manual.pdf", "pdf_manuals/output.csv", output_format="csv", pages='all')

# convert all PDFs in a directory
# tabula.convert_into_by_batch("input_directory", output_format='csv', pages='all')
