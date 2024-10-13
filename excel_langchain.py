from langchain_openai import OpenAI, ChatOpenAI
from langchain_experimental.agents.agent_toolkits import create_csv_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import streamlit as st
from io import BytesIO
import pandas as pd
from langchain_experimental.agents.agent_toolkits.python.base import create_python_agent
from langchain.prompts import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.sequential import SequentialChain,SimpleSequentialChain
import matplotlib.pyplot as plt
from langchain_experimental.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain_experimental.agents import create_pandas_dataframe_agent,create_csv_agent

st.set_page_config(layout='wide')


OPENAI_API_KEY = '<OPENAI_API_KEY>'
llm = ChatOpenAI(temperature=0,model='gpt-4',api_key=OPENAI_API_KEY)
file = st.file_uploader('choose a file to upload',type='csv')


def agent():
    pagent = create_python_agent(
        llm=llm,
        tool=PythonREPLTool(),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
       handle_parsing_errors=True
    )
    return pagent
     

def get_value(algorithm,df,answer):
    prompt = f"""
        Generate a Python script to solve the following problem:
        Problem: {answer}
        Use the following algorithm: {algorithm}
        Use the following dataframe {df}. Use the pandas DataFrame provided as the variable `df`. The dataframe is already loaded and available, so do not load any other dataset or create new data.
        Only return the Python code, without explanations, comments, or markdown formatting.
        When printing the output do it in streamlit so that it could be displayed after execution
        """
    solution = agent().run(prompt)    
    return solution

memory = ConversationBufferMemory(return_messages=True)

def create_history(messages):
    for message in messages:
        if message["role"] == "user":
            memory.chat_memory.add_user_message(message['content'])
        else:
            memory.chat_memory.add_ai_message(message['content'])
    return memory


# Function to detect whether the user is requesting a graph
def is_graph_request(query):
    questionPrompt = PromptTemplate(
        input_variables=["query"],
        template="""
            You are a helpful assistant that interprets user queries and decides whether the user is asking for plotting a chart or asking for information.
            Answer with one word based on the question. For example, if the user asks for a 'line graph of the volume column', answer 'plot'. Otherwise, answer 'information'.
        User query: {query}
        Answer:"""
    )
    chain = LLMChain(llm=llm, prompt=questionPrompt, memory=memory)  # Ensure memory is passed
    answer = chain.run({"query": query})
    return answer.strip().lower()

# Generate and display the graph
def generate_and_display_graph():
    graph_prompt = PromptTemplate(
        input_variables=["input"],
        template="""
        You are a helpful assistant that generates Python code to create the requested graph using matplotlib.
        The user will specify the graph type and the columns to plot. Respond with the correct Python code to plot the graph using Streamlit's `st.pyplot()`.
        Use the dataframe provided to generate the correct code. For example, if the user asks for a 'line graph of the volume column', generate that graph.

        INPUT: {input}
        Python code:"""
    )
    return graph_prompt

# Function to automatically generate insights based on the dataset
def auto_generate_question(df):
    with st.chat_message("assistant"):
        st.write("**Data Overview**")
        st.write(df.head())
        # st.session_state.messages.append({"role": "assistant", "content": df.head()})
        
        insights = [
            "What are the meanings of the columns?",
            "How many missing values does this dataset contain? Answer in a full sentence.",
            "Are there any duplicate values? If none, say there are no duplicate values.",
            "What new features would be interesting to create?"
        ]
        
    for insight in insights:
        agent = create_pandas_dataframe_agent(llm, df, memory=memory, allow_dangerous_code=True, verbose=True)
        result = agent.run(insight)
        with st.chat_message("assistant"):
            st.write(result)
            st.session_state.messages.append({"role": "assistant", "content": result})

# Initialize session state for messages and initialization status
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'initialized' not in st.session_state:
    st.session_state.initialized = False

# File uploader and main logic
if file is not None:
    df = pd.read_csv(file)
    tab1,tab2 = st.tabs(["Q/A with the dataset","Data Analysis"])
    with tab1:

        file_data = BytesIO(file.getvalue())

        # Auto-generate insights if not initialized
        if not st.session_state.initialized:
            # auto_generate_question(df)
            st.session_state.initialized = True
        else:
            # Display previous messages from the chat
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.write(message["content"])

        # Capture new user input
        q = st.chat_input("choose a file")
        if q is not None:
            with st.chat_message('user'):
                st.session_state.messages.append({"role": "user", "content": q})
                st.markdown(q)
                a = is_graph_request(q)

            # Handle plot requests
            if a == 'plot':
                input_text = f"QUESTION: {q}\nDATAFRAME:\n{df}"
                prompt = generate_and_display_graph()
                chain = LLMChain(llm=llm, prompt=prompt, memory=memory, verbose=True)
                pcode = chain.run({"input":input_text})
                with st.chat_message("assistant"):
                    if "plt.show()" in pcode:
                        pcode = pcode.replace("plt.show()", "st.pyplot(plt)")

                    st.code(pcode, language="python")
                    st.session_state.messages.append({"role": "assistant", "content": pcode})

                    exec_locals = {"df": df, "plt": plt, "st": st}
                    pcode = pcode.replace("```python", "")
                    pcode = pcode.replace("```", "")
                    try:
                        exec(pcode, {}, exec_locals)
                    except Exception as e:
                        st.error(f"Error executing the code: {e}")

            else:
                ptemplate = PromptTemplate(
                    input_variables=["q","answer"],
                    template="""
                        You are a helpful assistant that answers questions based on the provided information.
                        Analyze the question and answer carefully and try to frame the answer with respect to the question provided.
                        Do realize that you are speaking to non techicle audience and be as detailed as possible.
                        Do not inculde any additional information other than framing the response from the question and answer provided.
                        List out the answer if its containes numerous names.
                        For example: 
                        Questiion: 'what is the attack of pikachu?'
                        ANSWER: 'According to the information, the attack power of pikachu is 55.This number is a measure of the damage Pikachu can inflict on its opponents during battles'
            
                        Now, given the question & answer:
                        Question: {q}
                        ANSWER:{answer}
                        """
                )
                agent = create_csv_agent(
                        ChatOpenAI(temperature=0,model='gpt-4',api_key=OPENAI_API_KEY),
                        verbose = True,
                        allow_dangerous_code = True,
                        path=file_data
                    )
                chain = LLMChain(llm=llm,prompt=ptemplate,verbose=True)
                with st.spinner("Generating response..."):
                    answer = agent.run(q)
                    results = chain.invoke({"q":q,"answer":answer})
                    
                    if results["text"] is not None:
                        with st.chat_message("assistant"):
                            st.write(results["text"])
                            st.session_state.messages.append({"role": "assistant", "content": results["text"]})

    with tab2:
        if 'algorithm' not in st.session_state:
            st.session_state.algorithm = None
        if 'answer' not in st.session_state:
            st.session_state.answer = None
        
        if 'k' not in st.session_state:
            st.session_state.k = []

        st.subheader("Frame your problem to meaninful business problem and know what machine learning models can be useful in solving them")
        input = st.chat_input("")
        buisnessPrompt = PromptTemplate(
                input_variables=["query"],
                template="""
                    Convert the following business problem into a data science problem into a single sentance answer.
                    User query: {query}
                    Answer:"""
                 )
        
        modelPromt = PromptTemplate(
                input_variables=["dsQuery"],
                template="""
                    interpret the user queries list out the machine learning algorithms which you deem fit to solve the problem one line after another with proper numerics.
                    list out only the top 10.
                    Below is the problem which needs to be solved.
                    User query: {dsQuery}
                Answer:"""
            )
        
        if input is not None:
            
            chain = LLMChain(llm=llm, prompt=buisnessPrompt,verbose=True)
            answer = chain.run({"query": input})
            if answer:
                st.session_state.answer = answer

            chain1 = LLMChain(llm=llm, prompt=modelPromt,verbose=True)
            models = chain1.run({"dsQuery": answer})
           
            with st.chat_message("ai"):
                    st.markdown(answer)
            with st.chat_message("ai"):
                st.markdown(models)

            st.session_state.k.append("Select an algo")
            a = str(models)
            a = a.split("\n")
            for i in a:
                st.session_state.k.append(i[3:])
            
        algorithm = st.selectbox("Choose an algorithm", st.session_state.k)
        if algorithm is not None:
            st.session_state.algorithm = algorithm
            final = get_value(algorithm=st.session_state.algorithm,df=df,answer=st.session_state.answer)
            with st.spinner("Generating response..."):
                
                exec_locals = {"df": df, "pd": pd, "st":st}
                final = final.replace("```python", "").replace("```", "")
                
                st.code(final, language="python")
                try:
                    # Execute the Python code
                    # st.write("executing")
                    exec(final, {}, exec_locals)
                    # st.write("done")
                    # Check if the code produces any variables, like predictions
                    # if "predictions" in exec_locals:
                    #     result = exec_locals["predictions"]
                    #     st.write("Predictions:")
                    #     st.write(result)


                except Exception as e:
                    st.error(f"Error executing the generated code: {e}")
                


