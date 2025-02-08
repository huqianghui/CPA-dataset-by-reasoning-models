import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
from azure.ai.inference.aio import ChatCompletionsClient
from azure.core.credentials import AzureKeyCredential


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
load_dotenv(verbose=True)
azureInferenceRoundRobinConnection= os.getenv("AZURE_INFERENCE_ROUND_ROBIN_CONNETION","[]")

@dataclass
class AzureAIInferenceConnection:
    endpoint: str
    apiKey: str

class AzureAIInferenceClientsRoundRobin:
    def __init__(self):
        self.clients = _build_azure_AI_inference_async_clients(azureInferenceRoundRobinConnection)
        self.client_count = len(self.clients)
        self.index = 0  # init
        self.lock = asyncio.Lock()
    
    async def get_next_client(self):
        async with self.lock: 
            # get client
            client = self.clients[self.index]
            # update index 
            self.index = (self.index + 1) % self.client_count
            return client
    
    async def close_all_clients(self):
        for client in self.clients:
            await client.close()

def _load_connections(azureAIInferenceRoundRobinConnection:str)->list[AzureAIInferenceConnection]:
    data_list = json.loads(azureAIInferenceRoundRobinConnection)
    return [
        AzureAIInferenceConnection(
            endpoint=item["AZURE_AI_INFERENCE_ENDPOINT"],
            apiKey=item["AZURE_AI_INFERENCE_API_KEY"],
        )
        for item in data_list
    ]
    
def _build_azure_AI_inference_async_clients(azureAIInferenceRoundRobinConnection:str)->list[ChatCompletionsClient]:
    azureAIInferenceConnectionList = _load_connections(azureInferenceRoundRobinConnection) 
    clients = []
    for connection in azureAIInferenceConnectionList:
        client = ChatCompletionsClient(
            endpoint=connection.endpoint,
            credential=AzureKeyCredential(connection.apiKey))
        clients.append(client)
    return clients

# init client
azure_ai_inference_client_manager = AzureAIInferenceClientsRoundRobin()