# -------------------------------------------------------
# Libraries
# -------------------------------------------------------
import os
import openai
import json

from azure.core.credentials import AzureKeyCredential
from azure.ai.textanalytics import TextAnalyticsClient
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizableTextQuery, RawVectorQuery

from tenacity import retry, wait_random_exponential, stop_after_attempt
import warnings
warnings.filterwarnings("ignore")

from utils.credentials import (
    AZURE_OPENAI_SERVICE,
    AZURE_OPENAI_KEY,
    AZURE_OPENAI_GPT,
    AZURE_OPENAI_GPT_EMBED,
    AZURE_OPENAI_VERSION,
    AZURE_SEARCH_SERVICE,
    AZURE_SEARCH_SERVICE_KEY,
    AZURE_SEARCH_INDEX,
    INDEX_FIELD_CONTENT,
    INDEX_FIELD_METADATA,
    AZURE_LANGUAGE_ENDPOINT,
    AZURE_LANGUAGE_KEY
)

# -------------------------------------------------------
# OpenAI credentials
# -------------------------------------------------------
openai.api_base = AZURE_OPENAI_SERVICE
openai.api_key = AZURE_OPENAI_KEY
openai.api_version = AZURE_OPENAI_VERSION
openai.api_type = "azure"
os.environ["OPENAI_API_KEY"] = AZURE_OPENAI_KEY

# -------------------------------------------------------
# Classes
# -------------------------------------------------------
class QnA:
    """
    A class for:
        - Extract keywords from the user's question using OpenAI.
        - Retrieve documents from Azure Cognitive Search using keywords.
        - Generate a response to the user's question using OpenAI.
        - Filter PII from the generated answer using Azure Text Analytics.
        - Additional filtering of PII from the masked data using OpenAI.
    """

    def get_keywords(self, in_question: str, verbose: bool = False) -> str:
        """
        Extract keywords from the user's question using OpenAI.


        Args:
            in_question (str): The input question to extract keywords from.
            verbose (bool): Whether to print verbose output. Default is False.

        Returns:
            str: The extracted keywords.
        """
        # Method 1: using openai.ChatCompletion
        keywords_prompt = """You task is to only extract keywords from the user's message.
Examples:
User: What is the phone number of John?
Assistant: John, phone number.

User: Give me summary of introduction.pdf
Assistant: intruduction.pdf, summary.

Respond only with extracted keywords from the message.
"""
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_GPT,
            messages=[
                {"role": "system", "content": keywords_prompt},
                {"role": "user", "content": in_question}
            ]
        )
        if verbose:
            print(response)
        keywords = response.choices[0]['message']['content']
        return keywords

        # Method 2: using openai.Completion
        keywords_prompt = f"""<|im_start|>system
Your task is to only extract keywords from the user's message.

Examples:
User: What is the phone number of John?
Assistant: John, phone number.

User: Give me summary of introduction.pdf
Assistant: intruduction.pdf, summary.

Respond only with extracted keywords from the message.

Message:
{in_question}
<|im_end|>

<|im_start|>assistant
"""
        response = openai.Completion.create(engine=AZURE_OPENAI_GPT,
                                            prompt = keywords_prompt,
                                            temperature=0.7,
                                            max_tokens=2048,
                                            stop=["<|im_end|>", "<|im_start|>"])
        keywords = response.choices[0]['text']
        return keywords

    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    # Function to generate embeddings from text
    def generate_embeddings(self, text):
        response = openai.Embedding.create(
            input=text, engine=AZURE_OPENAI_GPT_EMBED)
        embeddings = response['data'][0]['embedding']
        return embeddings

    def retrieve_documents(self, keywords: str, top_k: int = 3) -> str:
        """
        Retrieve documents from Azure Cognitive Search using keywords.
  
        Args:
            keywords (str): The keywords to search for.
            metadata_field (str): The name of the metadata field to retrieve.
            content_field (str): The name of the content field to retrieve.
            top_k (int): The number of documents to retrieve. Default is 3.

        Returns:
            str: The retrieved documents as a string.
        """
        search_cred = AzureKeyCredential(AZURE_SEARCH_SERVICE_KEY)
        search_client = SearchClient(
            endpoint=AZURE_SEARCH_SERVICE,
            index_name=AZURE_SEARCH_INDEX,
            credential=search_cred
            )

        # Perform Hybrid Search
        vector_query = RawVectorQuery(vector=self.generate_embeddings(keywords), k=top_k, fields="content_vector")
        r = search_client.search(search_text = keywords,
                                 vector_queries=[vector_query],
                                 #  filter=filter, 
                                 top=top_k)

        ### TODO: Understand this and other parameters from below. Find configuration needed for a better retrieveal.
            # r = search_client.search(q, 
    #                                 filter=filter,
    #                                 query_type=QueryType.SEMANTIC, 
    #                                 query_language="it-it", 
    #                                 query_speller="lexicon", 
    #                                 semantic_configuration_name="default", 
    #                                 top=top, 
    #                                 query_caption="extractive|highlight-false" if use_semantic_captions else None)

        results = [f"File - {json.loads(doc[INDEX_FIELD_METADATA])['source']}, Page - {int(json.loads(doc[INDEX_FIELD_METADATA])['page']) + 1} : " + self.nonewlines(doc[INDEX_FIELD_CONTENT]) for doc in r]
        ### TODO: Check if it is better to use it:
        # if use_semantic_captions: 
        #     results = [doc[self.sourcepage_field] + ": " + nonewlines(" . ".join([c.text for c in doc['@search.captions']])) for doc in r]
        content = "\n".join(results)
        return content
    

    def llm(self, in_question, content, history, verbose=False):
        """
        Generate a response to the user's question using OpenAI.

        Args:
            in_question (str): The question to be answered.
            content (str): The source content to be used for answering the question.
            history (str): The chat history to be used for answering the question.
            verbose (bool, optional): Whether to print the OpenAI response. Defaults to False.

        Returns:
            str: The response generated by OpenAI's GPT-3 API.
        """

        ragPrompt = f"""You are an assistant chatbot for question-answering tasks. Use only information from Source or Chat History to answer the question.
If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
If you didn't receive the question but some random message, tell 'I didn't get your request, how can I assist you?'.
If you received the question but there is nothing in Source or Chat History to answer it, respond with 'I don't know the answer, sorry.'.

Source: 
{content}

Chat History:
{history}
"""
        response = openai.ChatCompletion.create(
            engine=AZURE_OPENAI_GPT,
            messages=[
                {"role": "system", "content": ragPrompt},
                {"role": "user", "content": in_question}
            ]
        )
        if verbose:
            print(response)
        llm_response = response.choices[0]['message']['content']
        return llm_response


    def language_pii_filter(self, text:str):
        """
        Filter PII from the generated answer using Azure Text Analytics.

        Returns:
            None
        """
        text_analytics_client = TextAnalyticsClient(endpoint=AZURE_LANGUAGE_ENDPOINT, 
                                                    credential=AzureKeyCredential(AZURE_LANGUAGE_KEY)
                                                   )
        result = text_analytics_client.recognize_pii_entities([text])
        docs = [doc for doc in result if not doc.is_error]
        for doc in docs:
            redacted_text = doc.redacted_text
        return redacted_text


    def openai_pii_filter(self, text:str):
        """
        Filters PII from the input text using OpenAI.

        Returns:
            None
        """
#         # Method 1: using openai.ChatCompletion
#         system_msg = f"""You are an assistant to filter PII from the message.
# Your task is to replace sensitive words with ****. to receive the message that can be masked. Your task is to check and replace unmasked sensitive words with ****. This includes personal information, credit card numbers, social security numbers, and other personally identifiable information.
# If there is no sensitive data to mask, return the original message received.
# If there is sensitive data to mask, respond only with the masked message.
# Do not add any new words or signs, only replace the sensitive ones.
# Do not provide any additional information or comment."""
#         messages= [
#             {"role": "system", "content": system_msg},
#             {"role": "user", "content": text}
#         ]
#         response = openai.ChatCompletion.create(engine=AZURE_OPENAI_GPT,
#                                                 messages = messages,
#                                                 temperature=0,
#                                                 # max_tokens=2048
#                                                 )
#         response = response.choices[0]['message']['content']
        # return response

        # Method 2: using openai.Completion
        system_msg = f"""<|im_start|>system
Your task is to replace sensitive words with ****.
This includes personal information, credit card numbers, social security numbers, and other personally identifiable information.
Do not remove or add any new words or signs.
Do not provide any additional information or comment. 
If there is no sensitive data to mask, respond only with the original message received.
If there is sensitive data to mask, respond only with the masked message.

Message:
{text}
<|im_end|>

<|im_start|>assistant
"""
        response = openai.Completion.create(engine=AZURE_OPENAI_GPT,
                                            prompt = system_msg,
                                            temperature=0.7,
                                            max_tokens=2048,
                                            stop=["<|im_end|>", "<|im_start|>"])
        response = response.choices[0]['text']
        return response

    def generate_response(self, messages):
        # Get last message and history of 10 messages
        in_question = messages[-1]['message']
        last_10_messages = messages[-10:]
        # Get keywords from the user's question
        keywords = self.get_keywords(in_question)
        # Retrieve documents from the indexer
        content = self.retrieve_documents(keywords)
        # Get the response from the LLM
        llm_response = self.llm(in_question, content, last_10_messages)
        # Mask PII with Azure Language
        masked_language = self.language_pii_filter(llm_response)
        # Mask PII with Azure OpenAI
        masked_openai = self.openai_pii_filter(masked_language)
        return masked_openai


    @staticmethod
    def nonewlines(s: str) -> str:
        """
        Replaces newline characters in a string with spaces.

        Args:
            s (str): The input string.

        Returns:
            str: The input string with newline characters replaced by spaces.
        """
        return s.replace('\n', ' ').replace('\r', ' ')

    # def openai_pii_filter_function(self, text:str):
    #     """
    #     Filters PII from the input text using OpenAI.

    #     Returns:
    #         None
    #     """
    #     messages= [
    #         {"role": "user", "content": text}
    #     ]

    #     functions= [  
    #         {
    #             "name": "censor_sensitive_data",
    #             "description": "Censors not-masked sensitive data from the text. This includes credit card numbers, social security numbers, and other personally identifiable information.",
    #             "parameters": {
    #                 "type": "object",
    #                 "properties": {
    #                     "toFilter": {
    #                         "type": "boolean",
    #                         "description": "False if there isn't not-masked sensitive data to be censored. True otherwise."
    #                     },
    #                     "censored": {
    #                         "type": "string",
    #                         "description": "The words to be censored"
    #                     }
    #                 },
    #                 "required": ["toFilter"]
    #             }
    #         }
    #     ]

    #     response = openai.ChatCompletion.create(
    #         engine=AZURE_OPENAI_GPT,
    #         messages=messages,
    #         functions=functions,
    #         function_call={"name":"censor_sensitive_data"}
    #     )
    #     response = json.loads(response['choices'][0]['message']['function_call']['arguments'])
        
    #     to_filter = response['toFilter']

    #     # If the filter is True and there is no censored node, set to_filter to False
    #     if to_filter == True and 'censored' not in response:
    #         to_filter = False

    #     # Replace the words to be censored with XXXXX
    #     if to_filter == True:
    #         to_censor = response['censored'].split(',')
    #         to_censor = [x.strip() for x in to_censor]

    #         censored_text = text
    #         for word in to_censor:
    #             censored_text = censored_text.replace(word, 'XXXXX')
    #         return censored_text
    #     else:
    #         return text