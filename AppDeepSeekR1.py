import pandas as pd
import os
import json
from dotenv import load_dotenv
from prompt.cpaDeekSeekPrompt import cpa_deep_seek_system_prompt, cpa_deep_seek_user_prompt
from azure.ai.inference.models import SystemMessage, UserMessage
import asyncio
import logging
import re
from cache.cacheConfig import cache,async_diskcache
from promptflow.tracing import start_trace,trace
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from roundRobin.azureInferenceClientRoundRobin import azure_ai_inference_client_manager


ERROR_CODE = "ERR_001"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

DEEPSEEK_R1_MODE_NAME="DeepSeek-R1"

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

@async_diskcache("answer_cpa_by_deepseek_r1_model")
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def answer_cpa_by_deepseek_r1_model(question:str,index:int):
    azureDeepSeekClient = await azure_ai_inference_client_manager.get_next_client()
    user_cpa_question = cpa_deep_seek_user_prompt.format(cpa_question=question)
    try:
        response = await azureDeepSeekClient.complete(
            messages=[
                SystemMessage(content=str(cpa_deep_seek_system_prompt)),
                UserMessage(content=str(user_cpa_question))
            ],
            model=DEEPSEEK_R1_MODE_NAME, # when use the github deepseekR1,the parameter is necessary, if use the azure serverless deepseek R1,it is not necessary
            temperature=0.6,
            top_p=1
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
            deepSeekR1Result = await answer_cpa_by_deepseek_r1_model(question,index)

        except Exception as e:
            logging.error(f"finlly failed to answer_cpa_by_deepseek_r1_model for row {index + 1}: {e}")
            deepSeekR1Result = f"{ERROR_CODE} row No. : {index + 1} >> {e}"
        
        return index, deepSeekR1Result

async def read_cpa_excel_file():
    # Read the Excel file
    df = pd.read_excel(os.getenv("CPA_FILE_PATH"))
    
    # Keep only the "ID", "question", and "answer" columns
    df = df[['ID', 'question', 'answer','difficulitiy']]

    #TODO Limit to the first 60 rows
    df = df.head(40)

    deepSeekOutputs = [None] * len(df)
    deepSeekProcessOutputs = [None] * len(df)
    deepSeekAnswerOutputs = [None] * len(df)

    semaphore = asyncio.Semaphore(int(os.getenv("CONCURRENT_TASK_SEMAPHORE_COUNT",1)))  # Limit to 40 concurrent tasks
    tasks = []
    for index, row in df.iterrows():
        tasks.append(process_row(index, row, semaphore))
    
    # Run tasks concurrently
    results = await asyncio.gather(*tasks)

    for index, deepSeekR1Result in results:
        deepSeekOutputs[index] = deepSeekR1Result

        try:
            cleanedDeepSeekR1ModifyResult = extract_json_content(deepSeekR1Result)
            deepSeekData = json.loads(cleanedDeepSeekR1ModifyResult)
            deepSeekProcessOutputs[index] = deepSeekData.get("process")
            deepSeekAnswerOutputs[index] = deepSeekData.get("answer")
        except Exception as e:
            logging.error(f"Failed to parse deepseek R1 response for row {index + 1}: {e}")

    df['deepseek-R1-result'] = deepSeekOutputs
    df['deepseek-R1-process'] = deepSeekProcessOutputs
    df['deepseek-R1-answer'] = deepSeekAnswerOutputs
    output_file_path = os.getenv("RESULT_OUTPUT_DIR_PATH") + "/" + "azure_deepseek_R1_result[0:60].xlsx"
    logging.info(f"All rows processed. Saving to Excel file, {output_file_path}")
    df.to_excel(output_file_path, index=False)

    await azure_ai_inference_client_manager.close_all_clients()

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
