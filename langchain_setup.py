import openai
from langchain.chains.llm import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import OpenAI

def setup_langchain(api_key):
    openai.api_key = api_key

    prompt_template = PromptTemplate(
        input_variables=["sequence"],
        template="Given the following sequence of stock prices, predict the next price: {sequence}"
    )

    llm = OpenAI(temperature=0.5, model="gpt-4", openai_api_key = api_key)
    llm_chain = LLMChain(llm=llm, prompt=prompt_template)
    return llm_chain
