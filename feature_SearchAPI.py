import sys
import os
import uuid
import datetime
from utils.QnA import *


# ------------------------------------------
# Create a new conversation
# ------------------------------------------
user_id = "ba152960ef224191b0578855639ddb09"

def newConversation(user_id):
    id = uuid.uuid4().hex
    conv_id = user_id + id
    timestamp_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    docType = "conversation"

    document = {
            "userID": user_id
            ,"id": id
            ,"convID": conv_id
            ,"timestamp": timestamp_utc
            ,"messages":[]
            ,"docType": docType
        }
    return document

document = newConversation(user_id)
document


# ------------------------------------------
# User Asked a Question
# ------------------------------------------
# Append new question
def append_new_message(document, msg, msg_type):
    """
    Appends the given question and type to the messages node of the given document dictionary.

    Args:
    document (dict): The document dictionary to append the question and type to.
    question (str): The question to append.
    type (str): The type of the question (either "User" or "Assistant").

    Returns:
    None
    """
    document['messages'].append({
        "message": msg,
        "type": msg_type
    })

# msg = "Hello, how can I help you?"
# msg_type = "bot"
# append_new_message(document, msg, msg_type)


msg = "Can you tell me more?"
msg_type = "user"
append_new_message(document, msg, msg_type)
document


in_question = document['messages'][-1]['message']
in_type = document['messages'][-1]['type']
last_10_messages = document['messages'][-10:]

qna = QnA()
# Get keywords from the user's question
keywords = qna.get_keywords(in_question)
print(keywords)
# Retrieve documents from the indexer
content = qna.retrieve_documents(keywords)
print(content)
# Get the response from the LLM
llm_response = qna.llm(in_question, content, last_10_messages)
print(llm_response)

# Mask PII with Azure Language
masked_language = qna.language_pii_filter(llm_response)
print(masked_language)
# Mask PII with Azure OpenAI
response = qna.openai_pii_filter(masked_language)
print(response)