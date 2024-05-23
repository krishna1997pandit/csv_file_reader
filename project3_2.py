import streamlit as st
import pandas as pd
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Function to set up the OpenAI API key
def set_openai_api_key(api_key: str):
    openai.api_key = api_key

# Streamlit app title
st.title("CSV File Analyzer with LLM")

# Input for OpenAI API Key
api_key = st.text_input("Enter your OpenAI API Key", type="password")

if api_key:
    # Set the OpenAI API key
    set_openai_api_key(api_key)

    # Initialize the OpenAI LLM through LangChain
    llm = OpenAI(api_key=api_key)

    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)

        # Display the dataframe
        st.write("Uploaded CSV file:")
        st.dataframe(df)
        
        # Ask user for the type of analysis they want
        user_query = st.text_input("Enter your query about the data:")

        if user_query:
            # Create the prompt template
            prompt_template = PromptTemplate(
                input_variables=["data", "query"],
                template="Given the following data:\n\n{data}\n\nAnswer the following question:\n\n{query}"
            )

            # Create the LLMChain
            chain = LLMChain(llm=llm, prompt=prompt_template)

            # Use the LLMChain to get the answer
            response = chain.run(data=df.to_string(index=False), query=user_query)

            # Display the result
            st.write("Result of your query:")
            st.write(response)
