import pandas as pd
import os
import json
from dotenv import load_dotenv
from openai import AzureOpenAI
from prompt.cpaUserPrompt import cpa_prompt
import asyncio
from roundRobin.azureOpenAIClientRoundRobin import AzureOpenAIClientsRoundRobin
import logging
import re
from cache.cacheConfig import cache,async_diskcache
from promptflow.tracing import start_trace,trace
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)

ERROR_CODE = "ERR_001"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

azureOpenAIClient = AzureOpenAI(
  azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"), 
  api_key=os.getenv("AZURE_OPENAI_KEY"),  
  api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

o1_client_manager = AzureOpenAIClientsRoundRobin(os.getenv("AZURE_OPENAI_O1_DEPLOYMENT_NAME","o1-preview"))
o1_mini_client_manager = AzureOpenAIClientsRoundRobin(os.getenv("AZURE_OPENAI_O1_MINI_DEPLOYMENT_NAME","o1-mini"))

def sanitize_json_string(json_str: str) -> str:
    # Remove all control characters (ASCII 0-31)
    return re.sub(r'[\x00-\x1F]', '', json_str)

def extract_json_content(text: str) -> str:
    # Remove control characters first
    cleaned_text = sanitize_json_string(text)
    start = cleaned_text.find('{')
    end = cleaned_text.rfind('}')
    if start == -1 or end == -1:
        raise ValueError("No valid JSON object found")
    return cleaned_text[start : end + 1]

@async_diskcache("answer_cpa_by_o1_preivew_model")
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def answer_cpa_by_o1_preivew_moddel(question:str,index:int):
    user_cpa_question = cpa_prompt.format(cpa_question=question)
    aAzureOpenclient = await o1_client_manager.get_next_client()
    # "Unsupported value: 'messages[0].role' does not support 'system' with this model.
    try:
        response = await aAzureOpenclient.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_O1_DEPLOYMENT_NAME","o1-preview"),
            messages=[
            {
                "role": "user",
                "content": str(user_cpa_question)
            }],
        )
        return response.choices[0].message.content
    except Exception as e:
            logging.error(f"{ERROR_CODE} >> API request failed for row {index + 1}: {e}. Retrying...")
            raise e


@async_diskcache("answer_cpa_by_o1_mini_model")
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def answer_cpa_by_o1_mini_model(question:str,index:int):
    user_cpa_question = cpa_prompt.format(cpa_question=question)
    aAzureOpenclient = await o1_mini_client_manager.get_next_client()
    try:
        response = await aAzureOpenclient.chat.completions.create(
            model=os.getenv("AZURE_OPENAI_O1_MINI_DEPLOYMENT_NAME","o1-mini"),
            messages=[
            {
                "role": "user",
                "content": str(user_cpa_question)
            }]
        )
        return response.choices[0].message.content
    except Exception as e:
        logging.error(f"{ERROR_CODE} >> API request failed for row {index + 1}: {e}. Retrying...")
        raise e

# Function to process a single row
async def process_row(index, row, semaphore):
    async with semaphore:
        logging.info(f"Processing row {index + 1}")
        question = row['question']
        try:
            o1PreivewModifyResult = await answer_cpa_by_o1_preivew_moddel(question,index)

        except Exception as e:
            logging.error(f"finlly failed to answer_cpa_by_o1_preivew_model for row {index + 1}: {e}")
            o1PreivewModifyResult = f"{ERROR_CODE} row No. : {index + 1} >> {e}"
        
        try:
            o1miniModifyResult = await answer_cpa_by_o1_mini_model(question,index)
        except Exception as e:
            logging.error(f"finlly failed to answer_cpa_by_o1_mini_moddl for row {index + 1}: {e}")
            o1miniModifyResult = f"{ERROR_CODE} row No. : {index + 1} >> {e}"
        
        return index, o1PreivewModifyResult, o1miniModifyResult

async def read_cpa_excel_file():
    # Read the Excel file
    df = pd.read_excel(os.getenv("CPA_FILE_PATH"))
    
    # Keep only the "ID", "question", and "answer" columns
    df = df[['ID', 'question', 'answer','difficulitiy']]

    o1PreivewOutputs = [None] * len(df)
    o1PreivewProcessOutputs = [None] * len(df)
    o1PreivewAnswerOutputs = [None] * len(df)
    o1miniOutputs = [None] * len(df)
    o1miniProcessOutputs = [None] * len(df)
    o1miniAnswerOutputs = [None] * len(df)

    semaphore = asyncio.Semaphore(int(os.getenv("CONCURRENT_TASK_SEMAPHORE_COUNT",40)))  # Limit to 40 concurrent tasks
    tasks = []
    for index, row in df.iterrows():
        tasks.append(process_row(index, row, semaphore))
    
    # Run tasks concurrently
    results = await asyncio.gather(*tasks)

    for index, o1PreivewModifyResult, o1miniModifyResult in results:
        o1PreivewOutputs[index] = o1PreivewModifyResult
        o1miniOutputs[index] = o1miniModifyResult

        try:
            cleanedO1PreivewModifyResult = extract_json_content(o1PreivewModifyResult)
            o1previewData = json.loads(cleanedO1PreivewModifyResult)
            o1PreivewProcessOutputs[index] = o1previewData.get("process")
            o1PreivewAnswerOutputs[index] = o1previewData.get("answer")
        except Exception as e:
            logging.error(f"Failed to parse O1 Preview response for row {index + 1}: {e}")

        try:
            cleanedO1miniModifyResult = extract_json_content(o1miniModifyResult)
            o1miniData = json.loads(cleanedO1miniModifyResult)
            o1miniProcessOutputs[index] = o1miniData.get("process")
            o1miniAnswerOutputs[index] = o1miniData.get("answer")
        except Exception as e:
            logging.error(f"Failed to parse O1 Mini response for row {index + 1}: {e}")

    df['o1-preview-result'] = o1PreivewOutputs
    df['o1-preview-process'] = o1PreivewProcessOutputs
    df['o1-preview-answer'] = o1PreivewAnswerOutputs
    df['o1-mini-result'] = o1miniOutputs
    df['o1-mini-process'] = o1miniProcessOutputs
    df['o1-mini-answer'] = o1miniAnswerOutputs
    output_file_path = os.getenv("RESULT_OUTPUT_DIR_PATH") + "/" + "azure_O1_preview_and_mini_result.xlsx"
    logging.info(f"All rows processed. Saving to Excel file, {output_file_path}")
    df.to_excel(output_file_path, index=False)

    return output_file_path
        
if __name__ == "__main__":
   start_trace()
   result = asyncio.run(read_cpa_excel_file())

   # Access cache statistics
   hits = cache.hits
   misses = cache.misses
   total_requests = hits + misses

   # Calculate hit and miss rates
   if total_requests > 0:
        hit_rate = hits / total_requests
        miss_rate = misses / total_requests
        print(f"Cache Hits: {hits}")
        print(f"Cache Misses: {misses}")
        print(f"Cache Hit Rate: {hit_rate * 100:.2f}%")
        print(f"Cache Miss Rate: {miss_rate * 100:.2f}%")
   else:
        print("No cache requests have been made.")

   print("save to :",result)
