import os 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
from langchain.agents import create_pandas_dataframe_agent 
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI

# Set 'OPENAI_API_KEY' variable first

df = pd.DataFrame({
    "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
    "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
    "happiness_index": [2, 1, 8, 3, 4, 4, 9, 10, 5, 30]
})

print(df)

llm = ChatOpenAI(temperature=0.0, model='gpt-3.5-turbo-0613')

agent = create_pandas_dataframe_agent(llm, df, verbose=True) 



# print(agent("What is the shape of the dataset?"))
print(agent("Take all countires with happiness index more than 9 and output sum of their gdps. Take a square root of this sum."))
# print(agent("Plot the 2 countries with lowest happiness index using bar graph. Place their gdp on y axis. Enable grid and use red colour. Save it to 'agents_plots/plot.png'"))
