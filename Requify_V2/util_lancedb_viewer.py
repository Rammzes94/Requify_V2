import lancedb

# Open your LanceDB
db = lancedb.connect("lancedb")

# List all tables
print("Tables:", db.table_names())

# Load a specific table
table = db.open_table("all_pdf_pages")
table2 = db.open_table("requirements")

# Convert both tables to pandas and save to Excel
df1 = table.to_pandas()
df2 = table2.to_pandas()

df1.to_excel("all_pdf_pages.xlsx", index=False)
df2.to_excel("requirements.xlsx", index=False)

