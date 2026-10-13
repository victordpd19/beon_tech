import os
from dotenv import load_dotenv
from openai import OpenAI
from anthropic import Anthropic, AsyncAnthropic
import services.utils as util


load_dotenv()
# openai_client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])
claude_client = AsyncAnthropic(api_key=os.environ['CLAUDE_API_KEY'])

def openai_complete(prompt:str, system: str='You are a helpful assistant' ) -> str:
    resp = openai_client.chat.completions().create(

        model = 'gpt-4o-mini', 
        messages = [{"role":"system", "content":system}, 
                    {"role":"user", "content": prompt},]
    )
    return resp.choices[0].message.content

@util.retry(2)
async def claude_complete(prompt:str, system_msg: str='You are a helpful assistant' ) -> tuple:
    resp = await claude_client.messages.create(

        model = 'claude-haiku-5-5',
        system= system_msg, 
        messages = [{"role":"user", "content": prompt}],
        max_tokens= 1024
    )
    return resp.content[0].text, resp.usage.output_tokens




# print(claude_complete(prompt="Who are you?"))