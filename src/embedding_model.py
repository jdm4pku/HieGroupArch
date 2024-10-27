from abc import ABC, abstractmethod
from openai import OpenAI
from sentence_transformers import SentenceTransformer
import time
from logger import get_logger
from tenacity import retry, stop_after_attempt,wait_random_exponential

logger = get_logger(__name__)

class BaseEmbeddingModel(ABC):
    @abstractmethod
    def create_embedding(self,text):
        pass

class OpenAIEmbeddingModel(BaseEmbeddingModel):
    def __init__(self,model="text-embedding-ada-002"):
        self.client = OpenAI(
            api_key = "", # 替换成你的api-key
            base_url="http://15.204.101.64:4000/v1"
        )
        self.model = model
    
    # @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def create_embedding(self, text):
        text =  text.replace("\n"," ")
        flag = False
        while not flag:
            try:
                embed = self.client.embeddings.create(input=[text],model=self.model).data[0].embedding
                flag=True
            except Exception as e:
                print(e)
                logger.info("retry")
                time.sleep(0.5)
        return embed
        # return (
        #     self.client.embeddings.create(input=[text],model=self.model)
        #     .data[0]
        #     .embedding
        # )

class SBertEmbeddingModel(BaseEmbeddingModel):
    def __init__(self,model_name="sentence-transformers/multi-qa-mpnet-base-cos-v1"):
        self.model = SentenceTransformer(model_name)
    
    def create_embedding(self, text):
        return self.model.encode(text)