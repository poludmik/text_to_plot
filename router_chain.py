import os
import random
from uuid import UUID
from langchain.chains.router import MultiPromptChain, MultiRouteChain
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate

from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser
from langchain.chains.router.multi_prompt_prompt import MULTI_PROMPT_ROUTER_TEMPLATE

# from langchain.agents import create_pandas_dataframe_agent
from langchain.schema.agent import AgentAction, AgentFinish
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain.chains import SimpleSequentialChain
import langchain.callbacks
from langchain.callbacks.base import BaseCallbackHandler, BaseCallbackManager

from langchain.agents.agent_toolkits.pandas.base import _get_prompt_and_tools, _get_functions_prompt_and_tools
from typing import Any, Dict, List, Optional, Sequence, Mapping

from langchain.agents.agent import AgentExecutor, BaseSingleActionAgent
from langchain.agents.mrkl.base import ZeroShotAgent
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.types import AgentType
from langchain.schema.language_model import BaseLanguageModel
from langchain.tools import BaseTool

class Config(): 
    model = 'gpt-3.5-turbo-0613'
    llm = ChatOpenAI(model=model, temperature=0, callbacks=[])

cfg = Config()

class PromptFactory():
    plot_template = """Goddamn it"""
# {input}. Save the resulting plot to 'plotname.png' and don't plt.show() it. Use Action: 'python_repl_ast'. 

    question_template = """{input}"""

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


class PlotAgentCallback(BaseCallbackHandler):
    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        print(f"!!!!!!!!! Plot agent callback, agent finished: {finish} !!!!!")

class QuestionAgentCallback(BaseCallbackHandler):
    def on_agent_finish(self, finish: AgentFinish, *, run_id: UUID, parent_run_id: UUID | None = None, **kwargs: Any) -> Any:
        print(f"!!!!!!!!! Question agent callback, agent finished: {finish} !!!!!")

# Overwrite function to add a plotname to the prompt
def create_pandas_dataframe_agent(
    llm: BaseLanguageModel,
    df: Any,
    agent_type: AgentType = AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    callback_manager: Optional[BaseCallbackManager] = None,
    prefix: Optional[str] = None,
    suffix: Optional[str] = None,
    input_variables: Optional[List[str]] = None,
    verbose: bool = False,
    return_intermediate_steps: bool = False,
    max_iterations: Optional[int] = 15,
    max_execution_time: Optional[float] = None,
    early_stopping_method: str = "force",
    agent_executor_kwargs: Optional[Dict[str, Any]] = None,
    include_df_in_prompt: Optional[bool] = True,
    number_of_head_rows: int = 5,
    extra_tools: Sequence[BaseTool] = (),
    plotname: str|None = None,
    **kwargs: Dict[str, Any],
) -> AgentExecutor:
    """Construct a pandas agent from an LLM and dataframe."""
    agent: BaseSingleActionAgent
    if agent_type == AgentType.ZERO_SHOT_REACT_DESCRIPTION:
        prompt, base_tools = _get_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        if plotname is not None:
            prompt.template = prompt.template.replace("{input}", "{input}" + ". Save the resulting plot to " + "'" + plotname +"' and don't plt.show() it. Use Action: python_repl_ast.")
        else:
            prompt.template = prompt.template.replace("{input}", "{input}" + ". Use Action: python_repl_ast.")
        # print("START")
        # print(prompt.template)
        # print("END")
        tools = base_tools + list(extra_tools)
        llm_chain = LLMChain(
            llm=llm,
            prompt=prompt,
            callback_manager=callback_manager,
        )
        tool_names = [tool.name for tool in tools]
        agent = ZeroShotAgent(
            llm_chain=llm_chain,
            allowed_tools=tool_names,
            callback_manager=callback_manager,
            **kwargs,
        )
    elif agent_type == AgentType.OPENAI_FUNCTIONS:
        _prompt, base_tools = _get_functions_prompt_and_tools(
            df,
            prefix=prefix,
            suffix=suffix,
            input_variables=input_variables,
            include_df_in_prompt=include_df_in_prompt,
            number_of_head_rows=number_of_head_rows,
        )
        tools = base_tools + list(extra_tools)
        agent = OpenAIFunctionsAgent(
            llm=llm,
            prompt=_prompt,
            tools=tools,
            callback_manager=callback_manager,
            **kwargs,
        )
    else:
        raise ValueError(f"Agent type {agent_type} not supported at the moment.")
    return AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        callback_manager=callback_manager,
        verbose=verbose,
        return_intermediate_steps=return_intermediate_steps,
        max_iterations=max_iterations,
        max_execution_time=max_execution_time,
        early_stopping_method=early_stopping_method,
        **(agent_executor_kwargs or {}),
    )

def generate_destination_chains(df):
    """
    Creates a list of LLM chains with different prompt templates.
    """
    prompt_factory = PromptFactory()
    destination_chains = {}
    plotname = "agents_plots/" + str(random.randint(0, 999)) + ".png"

    for p_info in prompt_factory.prompt_infos:
        name = p_info['name']

        if name == "Plot agent":
            plotname = plotname
            callback = PlotAgentCallback()
        else:
            plotname = None
            callback = QuestionAgentCallback()
            
        agent = create_pandas_dataframe_agent(cfg.llm, df, verbose=True,
                        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, 
                        max_iterations=10,
                        #   prefix=prefix,
                        callback_manager=BaseCallbackManager([callback]),
                        plotname=plotname,
                        )

        destination_chains[name] = agent

    # default_chain = ConversationChain(llm=cfg.llm, output_key="text")
    default_agent = create_pandas_dataframe_agent(cfg.llm, df, verbose=True, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, max_iterations=10)
    return prompt_factory.prompt_infos, destination_chains, default_agent


class MultitypeDestRouteChain(MultiPromptChain) :
    """A multi-route chain that uses an LLM router chain to choose amongst prompts."""

    # router_chain: LLMRouterChain
    # """Chain for deciding a destination chain and the input to it."""

    destination_chains: Mapping[str, langchain.agents.agent.AgentExecutor]
    """Map of name to candidate chains that inputs can be routed to."""

    default_chain: langchain.agents.agent.AgentExecutor
    """Default chain to use when router doesn't map input to one of the destinations."""

    @property
    def output_keys(self) -> List[str]:
        return ["output"]


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
    return MultitypeDestRouteChain(
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

    question = "Pie plot a graph of three countries with lowest happiness_index. Make the pie plot 3D."
    # question = "Pie plot me GDPs of UK and Italy, but multiply Italy's GDP by 2. Use red color for UK and blue for Italy. Title the figure 'kek' and add legend. "
    # question = "Plot all countries gdps."
    # question = "How many rows are there?"

    try:
          result = chain(question)
    except Exception as e:
             print("EXCEPTION")
             result = str(e)
             if not result.startswith("Could not parse LLM output: `"):
                 raise e
             result = result.removeprefix("Could not parse LLM output: `").removesuffix("`")
    print(result)
