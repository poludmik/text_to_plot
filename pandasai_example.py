import pandas as pd
from pandasai import SmartDataframe
from pandasai import PandasAI
from pandasai.llm import OpenAI


llm = OpenAI()
llm.model = "gpt-4"

# Define a dataframe
# df = pd.DataFrame({
#     "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
#     "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
#     "happiness_index": [2, 1, 8, 3, 4, 4, 9, 10, 5, 30]
# })

# Or read a .csv file
df = pd.read_csv("data/weight_height.csv")

print(df)

# 'PandasAI()' will be deprecated
# pandas_ai = PandasAI(llm, verbose=True, enable_cache=False)
# res = pandas_ai(df, prompt='Which are the 4 happiest countries?')

df = SmartDataframe(df, config={"llm": llm, "enable_cache": False, "save_charts": True})

# request = "What are 2 least happy countries?"

# request = "Find the top 4 happiest countries. Plot them using different colors for each bar"

request = """Plot the average height and weight"""

# res = df.chat(request)
# print(res)

# df.chat()
print(df.chat(request))


# logs are in pandasai.log
# plots are saved to exports/charts
