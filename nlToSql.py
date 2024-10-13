from typing import List
import streamlit as st
from sqlalchemy import create_engine
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI, OpenAI
from langchain.chains.sql_database.query import create_sql_query_chain 
from langchain.memory import ChatMessageHistory
import os
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains.openai_tools import create_extraction_chain_pydantic
from operator import itemgetter
import pandas as pd

server = 'varun'
database = 'loan_database'
username = 'sa'
password = '***********'
driver = 'ODBC Driver 17 for SQL Server'

OPENAI_API_KEY = 'api key'
os.environ["LANGCHAIN_TRACING_V2"] = 'true'
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
os.environ["LANGCHAIN_API_KEY"] = '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'


# Pick up only the tables info we need not all of them

tables =  pd.read_csv('table_details.csv')
tables = tables.dropna()
details = ''
for index,row in tables.iterrows():
    details = details + "Table Name:" + row["Table"] + "\n" + "Description:" + row["Description"] +"\n\n"

class Table(BaseModel):
    name:str = Field(description="Name of table in SQL database.")

def get_tables(tables: List[Table]) ->List[str]:
    tables  = [table.name for table in tables]
    return tables


table_details_prompt = f"""Return the names of ALL the SQL tables that MIGHT be relevant to the user question. \
The tables are:

{details}

Remember to include ALL POTENTIALLY RELEVANT tables, even if you're not sure that they're needed."""

table_chain = {"input":itemgetter("question")} | create_extraction_chain_pydantic(Table,llm=ChatOpenAI(temperature=0,model='gpt-4',api_key=OPENAI_API_KEY),system_message=table_details_prompt) | get_tables

def get_chain():
    
    db = SQLDatabase.from_uri(f'mssql://{username}:{password}@{server}/{database}?driver={driver}')

    llm = ChatOpenAI(temperature=0,model='gpt-4',api_key=OPENAI_API_KEY)

    generate_query = create_sql_query_chain(llm,db)
    run = QuerySQLDataBaseTool(db=db)
    ans_prompt = PromptTemplate.from_template("""
            Given the following user question, corresponding SQL query, and the SQL result answer the user question.
            QUESTION:{question}
            SQL QUERY :{query}
            SQL RESULT:{result}
            Answer:                               
    """)
    answer = ans_prompt | llm | StrOutputParser()

    chain = (
        RunnablePassthrough.assign(tabble_name_to_use = table_chain) |
        RunnablePassthrough.assign(query=generate_query).assign(
            result = itemgetter("query") | run
        )
        | answer
    )
    return chain

def create_history(messages):
    history = ChatMessageHistory()
    for message in messages:
        if message["role"] == 'user':
            history.add_user_message(message['content'])
        else:
            history.add_ai_message(message['content'])
    
    return history

def invoke_chain(question,messages):

    chain = get_chain()
    history = create_history(messages)
    response = chain.invoke({"question":question,"messages":history.messages})
    history.add_user_message(question)
    history.add_ai_message(response)
    return response


if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("Generating response..."):
        with st.chat_message("assistant"):
            response = invoke_chain(prompt,st.session_state.messages)
            st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})






