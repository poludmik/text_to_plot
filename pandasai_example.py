import pandas as pd
from pandasai import SmartDataframe
from pandasai import PandasAI
from pandasai.llm import OpenAI

# Set 'OPENAI_API_KEY' variable first

llm = OpenAI()
# llm.model = "gpt-4"
print(llm.model)

df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [1, 2, 8, 3, 4, 4, 9, 10, 5, 11]
})

# 'PandasAI()' will be deprecated
# pandas_ai = PandasAI(llm, verbose=True, enable_cache=False)
# res = pandas_ai(df, prompt='Which are the 4 happiest countries?')

df = SmartDataframe(df, config={"llm": llm, "enable_cache": False, "save_charts": True}) # "save_charts_path": "/some_path"

res = df.chat('What are 2 least happy countries?')
print(res)

# df.chat("Plot the 4 happiest countries, using different colors for each bar",)

# logs are in pandasai.log
# plots are in exports/charts
