import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from langchain.agents import create_pandas_dataframe_agent 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Define a dataframe
# df = pd.DataFrame({
#     "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
#     "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
#     "happiness_index": [2, 1, 8, 3, 4, 4, 9, 10, 5, 30]
# })

# Or read a .csv file
df = pd.read_csv("data/weight_height.csv")

print(df)

print(df)

llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

agent = create_pandas_dataframe_agent(llm, df, verbose=True) 



# request = """What is the shape of the dataset?"""

# request = """Take all countires with happiness index more than 9 and output sum of their gdps. Take a square root of this sum."""

# request = """Plot the 3 countries with lowest gdps using continuous line graph. Place their happiness indexes on y axis. 
#               Enable grid and use red colour. Save it to 'plots/happiness_index.png'"""

request = """Find the average height and average weight. Then convert height to cm and weight to kg. Bar plot both averages. 
                Use blue color for height and yellow for weight. Save it to 'plots/average_h_w.png'"""

print(agent(request))