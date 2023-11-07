# from langchain.document_loaders import PyMuPDFLoader

# loader = PyMuPDFLoader("pdf_manuals/Porsche-Taycan-Manual.pdf")
# data = loader.load()
# print(data[0])

################

import camelot
import matplotlib.pyplot as plt

# tables = camelot.read_pdf("pdf_manuals/bobcat_removed.pdf", line_scale=100, pages='1', copy_text=['v']) # copy_text=['v'] - spans shared cells vertically
tables = camelot.read_pdf("pdf_manuals/Porsche-Taycan-Manual_removed_avepdf.pdf", 
                                # process_background=True, 
                                flavor='stream', 
                                # line_scale=40, 
                                pages='1',
                                table_regions=['40,409,562,75'], # for first page of porsche
                                # table_regions=['41,250,560,45'],
                                # edge_tol=50, 
                                # copy_text=['v'],
                            )

# tables.export('foo1.csv', f='csv', compress=False)

print(tables)
idx = 0
# camelot.plot(tables[idx], kind='text').show() # text, line, joint, contour etc.
# plt.pause(0)
print(tables[idx].df)
print(tables[idx].parsing_report)


# tables = camelot.read_pdf("pdf_manuals/Porsche-Taycan-Manual_removed_avepdf.pdf", flavor='stream', table_areas=['20,20,200,100'])


#####################

# import tabula

# # Read pdf into list of DataFrame
# # dfs = tabula.read_pdf("pdf_manuals/Porsche-Taycan-Manual_removed_avepdf_bb_v.pdf", pages='1') # page '271' is 269 on manual pdf pages (and apparently it indexes from 1...)
# dfs = tabula.read_pdf("pdf_manuals/bobcat_removed.pdf", pages='2')
# print(dfs[0])
# print()
# print()
# print(dfs[1])

# convert PDF into CSV file
# tabula.convert_into("pdf_manuals/Porsche-Taycan-Manual.pdf", "pdf_manuals/output.csv", output_format="csv", pages='all')

# convert all PDFs in a directory
# tabula.convert_into_by_batch("input_directory", output_format='csv', pages='all')
