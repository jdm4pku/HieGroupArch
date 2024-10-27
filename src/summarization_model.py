from abc import ABC, abstractmethod
from openai import OpenAI
import time
from tenacity import retry, stop_after_attempt, wait_random_exponential

class BaseSummarizationModel(ABC):
    @abstractmethod
    def summarize(self,context,max_tokens=150):
        pass

class GPT3TurboSummarizationModel(BaseSummarizationModel):
    def __init__(self,model="gpt-4o"):
        self.model = model
    @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self,context,max_tokens=500,stop_sequence=None):
        try:
            client = OpenAI(api_key = "", base_url="http://15.204.101.64:4000/v1") # 替换成你的api-key
            response = client.chat.completions.create(
                model = self.model,
                messages=[
                    {"role": "system", "content": "You are an expert in Linux functional analysis and feature modeling."},
                    {
                        "role": "user",
                        "content": f"Based on the following sub-features, please generate a parent feature that can cover these sub-features. The sub-features are: {context}: \n Please only output parent feature in the format of 'feature name:\n feature description:'.",
                    },
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        
        except Exception as e:
            print(e)
            return e

class GPT3SummarizationModel(BaseSummarizationModel):
    def __init__(self, model="text-davinci-003"):
        self.model = model

    # @retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6))
    def summarize(self, context, max_tokens=500, stop_sequence=None):
        client = OpenAI(api_key="", base_url="http://15.204.101.64:4000/v1") # 替换成你的api-key
        flag = False
        while not flag:
            try:
                response = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {
                                "role": "user",
                                "content": f"Based on the following sub-features, please generate a parent feature that can cover these sub-features. The sub-features are: {context}: \n Please only output parent feature in the format of 'feature name:\n feature description:'.",
                            },
                        ],
                        model="gpt-3.5-turbo",
                        max_tokens=max_tokens,   
                )
                flag=True
            except Exception as e:
                print(e)
                time.sleep(0.5)
        return response.choices[0].message.content
        # try:
        #     client = OpenAI()
        #     response = client.chat.completions.create(
        #         model=self.model,
        #         messages=[
        #             {"role": "system", "content": "You are a helpful assistant."},
        #             {
        #                 "role": "user",
        #                 "content": f"Write a summary of the following, including as many key details as possible: {context}:",
        #             },
        #         ],
        #         max_tokens=max_tokens,
        #     )
        #     return response.choices[0].message.content
        # except Exception as e:
        #     print(e)
        #     return e
