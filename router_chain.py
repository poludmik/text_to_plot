import os
import random
from typing import Any, Dict, List, Optional
from uuid import UUID
from langchain.chains.router import MultiPromptChain, MultiRouteChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

from langchain.agents import create_pandas_dataframe_agent
from langchain.schema.agent import AgentAction, AgentFinish
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain.chains import SimpleSequentialChain
import langchain.callbacks
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager


class Config(): 
    model = 'gpt-3.5-turbo-0613'
    llm = ChatOpenAI(model=model, temperature=0, callbacks=[])

cfg = Config()

class PromptFactory():
    plot_template = """You are the agent that can analize dataframes and create charts and plots based on them.

    Here is the task:
    {input}"""

    question_template = """You are the agent that can analize dataframes and answer questions on them.

    Here is the task:
    {input}"""


    prompt_infos = [
        {
            'name': 'Plot agent',
            'description': 'Good for creating plots and charts based on a provided dataframe',
            'prompt_template': plot_template
        },
        {
            'name': 'Question agent',
            'description': 'Good for answering math questions based on a provided dataframe',
            'prompt_template': question_template
        },
    ]


class MyCustomHandler(BaseCallbackHandler):
    # def on_llm_new_token(self, token: str, **kwargs) -> None:
    #     print(f"My custom handler, token: {token}")

    # def on_chain_start(self, serialized: Dict[str, Any], inputs: Dict[str, Any], *, run_id: UUID, parent_run_id: UUID | None = None, tags: List[str] | None = None, metadata: Dict[str, Any] | None = None, **kwargs: Any) -> Any:
    #     print("GODDAMNG", inputs, "*****")

    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        print(f"!!!!!!!!! My custom handler, agent finished: {finish} !!!!!")


def generate_destination_chains(df):
    """
    Creates a list of LLM chains with different prompt templates.
    """
    prompt_factory = PromptFactory()
    destination_chains = {}

    for p_info in prompt_factory.prompt_infos:
        name = p_info['name']
        # prompt_template = p_info['prompt_template']
        # chain = LLMChain(
        #     llm=cfg.llm, 
        #     prompt=PromptTemplate(template=prompt_template, input_variables=['input']))
        if name == "Plot agent":
            plotname = "agents_plots/" + str(random.randint(0, 999)) + ".png"
            # suffix = " Save the chart to " + "'" + plotname +"'." + ". Please use Action: python_repl_ast."
            prefix = "Please use Action: python_repl_ast. Save the resulting plot to " + "'" + plotname +"' and don't plt.show() it.\n"
        else:
            prefix = "Please use Action: python_repl_ast.\n"

        agent = create_pandas_dataframe_agent(cfg.llm, df, verbose=True, 
                                              agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                                              max_iterations=10, 
                                              prefix=prefix,
                                              callback_manager=BaseCallbackManager([MyCustomHandler()]))
        destination_chains[name] = agent

    # default_chain = ConversationChain(llm=cfg.llm, output_key="text")
    default_agent = create_pandas_dataframe_agent(cfg.llm, df, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, max_iterations=10, prefix="Please use Action: python_repl_ast.\n")
    return prompt_factory.prompt_infos, destination_chains, default_agent


def generate_router_chain(prompt_infos, destination_chains, default_chain):
    """
    Generats the router chains from the prompt infos.
    :param prompt_infos The prompt informations generated above.
    """
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = '\n'.join(destinations)
    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template,
        input_variables=['input'],
        output_parser=RouterOutputParser()
    )
    router_chain = LLMRouterChain.from_llm(cfg.llm, router_prompt)
    return MultiRouteChain(
        router_chain=router_chain,
        destination_chains=destination_chains,
        default_chain=default_chain,
        verbose=True,
        callbacks=[]
    )


if __name__ == "__main__":
    # Put here your API key or define it in your environment
    # os.environ["OPENAI_API_KEY"] = '<key>'

    df = pd.DataFrame({
        "country": ["United States", "United Kingdom", "France", "Germany", "Italy", "Spain", "Canada", "Australia", "Japan", "China"],
        "gdp": [19294482071552, 2891615567872, 2411255037952, 3435817336832, 1745433788416, 1181205135360, 1607402389504, 1490967855104, 4380756541440, 14631844184064],
        "happiness_index": [2, 1, 8, 3, 4, 4, 9, 10, 5, 30]
    })

    prompt_infos, destination_chains, default_chain = generate_destination_chains(df)
    chain = generate_router_chain(prompt_infos, destination_chains, default_chain)

    # question = "Plot a graph of three countries with lowest happiness_index."
    question = "How many rows are there?"

    result = chain(question)
    print(result)
    print()
