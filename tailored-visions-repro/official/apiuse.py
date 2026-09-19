# Code by zzjchen
# This code includes functions for ChatGPT API. Requires your OpenAI api key for usage.
# Login and visit https://platform.openai.com/account/api-keys to see your OpenAI api key
import openai
import os
#Replace the following openai key, organization with your openai key or organization
#If you have special api_base for api.openai.com, replace My_Base with your correct api_base
#
# [2026 patch] Also readable from the environment, so the repo runs without
# editing source. Point TV_OPENAI_BASE at serve_local_llm.py to run with a
# local model instead of ChatGPT:
#     python serve_local_llm.py &
#     export TV_OPENAI_BASE=http://127.0.0.1:8000/v1 TV_OPENAI_KEY=local
MY_OPENAI_KEY=os.environ.get('TV_OPENAI_KEY') or os.environ.get('OPENAI_API_KEY') or ""
MY_ORG=os.environ.get('TV_OPENAI_ORG') or ""
MY_BASE=os.environ.get('TV_OPENAI_BASE') or 'https://api.openai.com/v1/'
MY_MODEL=os.environ.get('TV_OPENAI_MODEL') or 'gpt-3.5-turbo'
import time
def stable_call_api(model_name=None,messages=[{'role':'user','content':'hello'}],return_dict=False,max_tokens=500,t_sleep=0.2,max_retry=5):
    '''
    A simple & stable wrapper for calling ChatGPT API, which keeps retrying after failure.
    Make sure you're able to connect the internet.

    Args:
        model_name: OpenAI model name. ChatGPT by default
        messages: Input message histories to ChatGPT
        return_dict: False if you just want to receive the string ChatGPT responds.
        max_tokens: Maximum of tokens ChatGPT is going to respond
        t_sleep: seconds until next retry in case of failure

    Returns:
        Chatgpt response message
    '''
    # [2026 patch] Upstream retries forever, so a bad key or an unreachable
    # endpoint hangs the run silently instead of failing. Bounded now.
    model_name = model_name or MY_MODEL
    skip=t_sleep
    for k in range(1, max_retry+1):
        try:
            return call_api(model_name, messages, return_dict, max_tokens)
        except Exception as e:
            print('  ',e,'Retrying', k, 'of', max_retry, 'sleep', skip)
            time.sleep(skip)
            skip+=t_sleep
    raise RuntimeError(
        'API call failed %d times. Check TV_OPENAI_KEY / TV_OPENAI_BASE, or start '
        'serve_local_llm.py to run without an OpenAI key.' % max_retry)
def call_api(model_name=None,messages=[{'role':'user','content':'hello'}],return_dict=False,max_tokens=3500):
    '''
        A simple wrapper for calling ChatGPT API.

        Args:
            model_name: OpenAI model name. ChatGPT by default
            messages: Input message histories to ChatGPT
            return_dict: False if you just want to receive the string ChatGPT responds.
            max_tokens: Maximum of tokens ChatGPT is going to respond

        Returns:
            Chatgpt response message
    '''
    completion = openai.ChatCompletion.create(
      model=model_name or MY_MODEL,
      messages=messages,
      max_tokens = max_tokens
    )

    ans=completion.choices[0].message
    if return_dict:
        return ans
    else:
        return ans['content']

if __name__ == '__main__':
    openai.organization = MY_ORG
    openai.api_key = MY_OPENAI_KEY
    mes=[{'role': 'user',
          'content' :"hello"
          }]
    print('User:',mes[0]['content'])
    message=call_api(messages=mes,return_dict=True)
    print('ChatGPT:',message)
