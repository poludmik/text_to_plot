import pandas as pd
from bambooai import BambooAI


df = pd.read_csv('data/country_happiness_gdp.csv')
# bamboo = BambooAI(df)
bamboo = BambooAI(df, debug=False, vector_db=False, exploratory=False, llm_switch_plan=False, llm_switch_code=True, search_tool=False)
bamboo.pd_agent_converse("Find top 7 least happy countries and barplot double of their gdps. Use red color for italy. Save the plot to 'agents_plots/bamboo_pie.png'. Don't use plt.show().")


