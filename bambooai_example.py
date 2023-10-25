import pandas as pd
from bambooai import BambooAI


df = pd.read_csv('data/country_happiness_gdp.csv')
# bamboo = BambooAI(df)

# exploratory=False automatically means Data Analyst DF agent

# bamboo = BambooAI(df, debug=False, vector_db=False, exploratory=False, llm_switch_plan=False, llm_switch_code=True, search_tool=False)
bamboo = BambooAI(df, debug=False, vector_db=False, exploratory=True, llm_switch_plan=False, search_tool=False, local_code_model='CodeLlama-7B-Python-fp16')

bamboo.pd_agent_converse("Find top 7 least happy countries.")