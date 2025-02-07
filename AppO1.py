import pandas as pd
import os
import json
from dotenv import load_dotenv
from openai import AsyncAzureOpenAI
from prompt.cpaUserPrompt import cpa_prompt
import asyncio
import logging
from prompt.cpaO1PlusPrompt import cpa_o1_plus_system_prompt, cpa_o1_plus_user_prompt
from cache.cacheConfig import cache,async_diskcache
from promptflow.tracing import start_trace,trace
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)
from pydantic import BaseModel

class ProcessAndAnswer(BaseModel):
    process: str
    answer: str

ERROR_CODE = "ERR_001"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv()

aAzureOpenAIClient = AsyncAzureOpenAI(
  azure_endpoint = os.getenv("AZURE_O1_AND_O3_INFERENCE_ENDPOINT"), 
  api_key=os.getenv("AZURE_O1_AND_O3_INFERENCE_CREDENTIAL"),  
  api_version=os.getenv("AZURE_O1_AND_O3_INFERENCE_API_VERSION")
)


@async_diskcache("answer_cpa_by_o1_medium_effort_model")
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def answer_cpa_by_o1_medium_effort_model(question:str,index:int)-> ProcessAndAnswer:
    user_cpa_question = cpa_o1_plus_user_prompt.format(cpa_question=question)
    try:
        # !!use .beta.chat.completions to parse, the .chat.completions has no parese method
        response = await aAzureOpenAIClient.beta.chat.completions.parse(
            model=os.getenv("AZURE_O1_DEPLOYMENT_NAME","o1"),
            messages=[
            {"role": "developer",
             "content": str(cpa_o1_plus_system_prompt)}, # optional equivalent to a system message for reasoning models     
            {
                "role": "user",
                "content": str(user_cpa_question)
            }],
            reasoning_effort="medium",
            response_format=ProcessAndAnswer
        )
        return response.choices[0].message.parsed
    except Exception as e:
            logging.error(f"{ERROR_CODE} >> API request failed for row {index + 1}: {e}. Retrying...")
            raise e

@async_diskcache("answer_cpa_by_o1_high_effort_model")
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def answer_cpa_by_o1_high_effort_model(question:str,index:int)-> ProcessAndAnswer:
    user_cpa_question = cpa_o1_plus_user_prompt.format(cpa_question=question)
    try:
        # !!use .beta.chat.completions to parse, the .chat.completions has no parese method
        response = await aAzureOpenAIClient.beta.chat.completions.parse(
            model=os.getenv("AZURE_O1_DEPLOYMENT_NAME","o1"),
            messages=[
            {"role": "developer",
             "content": str(cpa_o1_plus_system_prompt)}, # optional equivalent to a system message for reasoning models     
            {
                "role": "user",
                "content": str(user_cpa_question)
            }],
            reasoning_effort="high",
            response_format=ProcessAndAnswer
        )
        return response.choices[0].message.parsed
    except Exception as e:
            logging.error(f"{ERROR_CODE} >> API request failed for row {index + 1}: {e}. Retrying...")
            raise e

@async_diskcache("answer_cpa_by_o1_low_effort_model")
@retry(wait=wait_random_exponential(multiplier=1, max=60), stop=stop_after_attempt(3))
async def answer_cpa_by_o1_low_effort_model(question:str,index:int)-> ProcessAndAnswer:
    user_cpa_question = cpa_o1_plus_user_prompt.format(cpa_question=question)
    try:
        # !!use .beta.chat.completions to parse, the .chat.completions has no parese method
        response = await aAzureOpenAIClient.beta.chat.completions.parse(
            model=os.getenv("AZURE_O1_DEPLOYMENT_NAME","o1"),
            messages=[
            {"role": "developer",
             "content": str(cpa_o1_plus_system_prompt)}, # optional equivalent to a system message for reasoning models     
            {
                "role": "user",
                "content": str(user_cpa_question)
            }],
            reasoning_effort="low",
            response_format=ProcessAndAnswer
        )
        return response.choices[0].message.parsed
    except Exception as e:
            logging.error(f"{ERROR_CODE} >> API request failed for row {index + 1}: {e}. Retrying...")
            raise e


# Function to process a single row
async def process_row(index, row, semaphore):
    async with semaphore:
        logging.info(f"Processing row {index + 1}")
        question = row['question']
        o1MediumEffortResult = None
        o1LowEffortResult = None
        o1HighEffortResult = None

        # Start tasks concurrently
        medium_task = asyncio.create_task(answer_cpa_by_o1_medium_effort_model(question, index))
        low_task = asyncio.create_task(answer_cpa_by_o1_low_effort_model(question, index))
        high_task = asyncio.create_task(answer_cpa_by_o1_high_effort_model(question, index))

        # Wait for all tasks to complete, capturing exceptions if any
        results = await asyncio.gather(medium_task, low_task, high_task, return_exceptions=True)

        # Process Medium Effort result
        if isinstance(results[0], Exception):
            logging.error(f"finlly failed to answer_cpa_by_o1_medium_effort_model for row {index + 1}: {results[0]}")
            o1MediumEffortResult = f"{ERROR_CODE} row No. : {index + 1} >> {results[0]}"
        else:
            o1MediumEffortResult = results[0]
        
        # Process Low Effort result
        if isinstance(results[1], Exception):
            logging.error(f"finlly failed to answer_cpa_by_o1_low_effort_model for row {index + 1}: {results[1]}")
            o1LowEffortResult = f"{ERROR_CODE} row No. : {index + 1} >> {results[1]}"
        else:
            o1LowEffortResult = results[1]

        # Process High Effort result
        if isinstance(results[2], Exception):
            logging.error(f"finlly failed to answer_cpa_by_o1_high_effort_model for row {index + 1}: {results[2]}")
            o1HighEffortResult = f"{ERROR_CODE} row No. : {index + 1} >> {results[2]}"
        else:
            o1HighEffortResult = results[2]        

        return index, o1MediumEffortResult,o1LowEffortResult,o1HighEffortResult

async def read_cpa_excel_file():
    # Read the Excel file
    df = pd.read_excel(os.getenv("CPA_FILE_PATH"))
    
    # Keep only the "ID", "question", and "answer" columns
    df = df[['ID', 'question', 'answer','difficulitiy']]

    o1Outputs = [None] * len(df)
    o1ProcessOutputs = [None] * len(df)
    o1AnswerOutputs = [None] * len(df)

    o1LowOutputs = [None] * len(df)
    o1LowProcessOutputs = [None] * len(df)
    o1LowAnswerOutputs = [None] * len(df)

    o1HighOutputs = [None] * len(df)
    o1HighProcessOutputs = [None] * len(df)
    o1HighAnswerOutputs = [None] * len(df)

    semaphore = asyncio.Semaphore(int(os.getenv("CONCURRENT_TASK_SEMAPHORE_COUNT")))  # Limit to 40 concurrent tasks
    tasks = []
    for index, row in df.iterrows():
        tasks.append(process_row(index, row, semaphore))
    
    # Run tasks concurrently
    results = await asyncio.gather(*tasks)

    for index, o1MediumEffortResult,o1LowEffortResult,o1HighEffortResult in results:
        try:
            o1Outputs[index] = json.dumps(o1MediumEffortResult.model_dump(), indent=4, ensure_ascii=False)
            o1ProcessOutputs[index] = o1MediumEffortResult.process
            o1AnswerOutputs[index] = o1MediumEffortResult.answer
        except Exception as e:
            logging.error(f"Failed to parse o1 medium effort response for row {index + 1}: {e}")
        
        try:
            o1LowOutputs[index] = json.dumps(o1LowEffortResult.model_dump(), indent=4, ensure_ascii=False)
            o1LowProcessOutputs[index] = o1LowEffortResult.process
            o1LowAnswerOutputs[index] = o1LowEffortResult.answer
        except Exception as e:
            logging.error(f"Failed to parse o1 low effort response for row {index + 1}: {e}")
        
        try:
            o1HighOutputs[index] = json.dumps(o1HighEffortResult.model_dump(), indent=4, ensure_ascii=False)
            o1HighProcessOutputs[index] = o1HighEffortResult.process
            o1HighAnswerOutputs[index] = o1HighEffortResult.answer
        except Exception as e:
            logging.error(f"Failed to parse o1 high effort response for row {index + 1}: {e}")

    df['o1-low-result'] = o1LowOutputs
    df['o1-low-process'] = o1LowProcessOutputs
    df['o1-low-answer'] = o1LowAnswerOutputs

    df['o1-meduium-result'] = o1Outputs
    df['o1-meduiumi-process'] = o1ProcessOutputs
    df['o1-meduium-answer'] = o1AnswerOutputs

    df['o1-high-result'] = o1HighOutputs
    df['o1-high-process'] = o1HighProcessOutputs
    df['o1-high-answer'] = o1HighAnswerOutputs

    output_file_path = os.getenv("RESULT_OUTPUT_DIR_PATH") + "/" + "azure_o1_result.xlsx"
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