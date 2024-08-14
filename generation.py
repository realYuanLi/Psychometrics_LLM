import openai
from zhipuai import ZhipuAI
from http import HTTPStatus
import dashscope
import math
import traceback
import replicate
from openai import AzureOpenAI, OpenAI
import json
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import concurrent.futures
import os, time
from tenacity import retry, wait_random_exponential, stop_after_attempt
import urllib3
import yaml
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def load_api():
    with open("api.yaml", 'r') as stream:
        config = yaml.safe_load(stream)
        return config

api_keys = load_api()

qwen_api_key = api_keys['api_keys']['qwen']
openai_api = api_keys['api_keys']['openai']
deepinfra_api = api_keys['api_keys']['deepinfra']
zhipu_api = api_keys['api_keys']['zhipu']

supported_models = ['gpt-4', 'gpt-3.5', 'llama3-8b', 'llama3-70b', 'mixtral-8*7b', 'mistral-7b', 'mixtral-8*22b', 'glm4', 'qwen-turbo']
supported_dimensions = ['personality', 'emotion', 'values', 'ToM', 'motivation']
supported_sub_tasks = ['EA', 'EU', 'self-efficacy', 'big_five_inventory', 'dark_traits', 'vignette_test', 'false_belief', 'imposing_memory', 'strange_stories', 'culture_orientation', 'human-centered_values', 'moral_belief']
supported_reliability = ['position_bias', 'parallel_forms', 'internal_consistency', 'inter-rater']

tasks_config = {
    'emotion': {
        'EA': ['position_bias'],
        'EU': ['position_bias']
    },
    'motivation': {
        'self-efficacy': ['parallel_forms']
    },
    'personality': {
        'big_five_inventory': ['internal_consistency'],
        'dark_traits': ['internal_consistency'],
        'vignette_test': ['inter-rater']
    },
    'ToM': {
        'false_belief': ['parallel_forms', 'position_bias'],
        'imposing_memory': ['parallel_forms'],
        'strange_stories': ['inter-rater']
    },
    'values': {
        'culture_orientation': ['internal_consistency'],
        'human-centered_values': ['position_bias'],
        'moral_belief': ['parallel_forms']
    }
}

deepinfra_model_mapping = {'llama2-70b': 'meta-llama/Llama-2-70b-chat-hf',
                           'llama2-7b': 'meta-llama/Llama-2-7b-chat-hf',
                           'llama3-8b': 'meta-llama/Meta-Llama-3-8B-Instruct',
                           'llama3-70b': 'meta-llama/Meta-Llama-3-70B-Instruct',
                           'mistral-7b': 'mistralai/Mistral-7B-Instruct-v0.2',
                           'mixtral-8*7b': 'mistralai/Mixtral-8x7B-Instruct-v0.1',
                           'mixtral-8*22b': 'mistralai/Mixtral-8x22B-Instruct-v0.1'}

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(5))
def get_res(string, model, temperature=0.5):
    model_mapping = {'gpt-4': 'XXX', 'gpt-3.5': 'XXX'}
    client = AzureOpenAI(
        api_key=openai_api,
        api_version="XXX",
        azure_endpoint="https://XXX"
    )
    try:
        chat_completion = client.chat.completions.create(
            model=model_mapping[model],
            messages=[
                {"role": "user", "content": string}
            ],
            temperature=0.5
        )
        print(chat_completion.choices[0].message.content)
        return chat_completion.choices[0].message.content
    except Exception as e:
        print(e)
        return None

def qwen_res(string, temperature=0.5):
    dashscope.api_key=qwen_api_key
    messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': string}]
    try:
        response = dashscope.Generation.call(
            dashscope.Generation.Models.qwen_turbo,
            messages=messages,
            result_format='message',  # set the result to be "message" format.
            temperature=0.5
        )
        if response.status_code == HTTPStatus.OK:
            print(response)
            return response['output']['choices'][0].message.content
        else:
            print(response)
            return None
    except:
        return None

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def deepinfra_res(string, model, temperature=0.5):
    client = OpenAI(api_key=deepinfra_api,
                    base_url='https://api.deepinfra.com/v1/openai')
    top_p = 1 if temperature <= 1e-5 else 0.9
    # temperature=0.0001 if temperature<=1e-5 else temperature
    chat_completion = client.chat.completions.create(
        model=deepinfra_model_mapping[model],
        messages=[{"role": "user", "content": string}],
        max_tokens=2500,
        temperature=temperature,
        top_p=top_p,
    )
    print(chat_completion.choices[0].message.content)
    return chat_completion.choices[0].message.content

@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(6))
def zhipu_res(string, model, temperature=0.5):
    model_mapping = {'glm4': 'GLM-4'}
    client = ZhipuAI(api_key=zhipu_api)
    if temperature == 0:
        temperature = 0.01
    else:
        temperature = 0.5
    response = client.chat.completions.create(
        model=model_mapping[model],
        messages=[
            {"role": "user", "content": string},
        ],
        temperature=temperature
    )
    print(response.choices[0].message.content)
    return response.choices[0].message.content


def process_prompt(el, model):
    def model_generate(prompt, model):
        if model in ['gpt-3.5', 'gpt-4']:
            return get_res(prompt, model)
        elif model in ['llama3-8b', 'llama3-70b', 'mistral-7b', 'mixtral-8*7b', 'mixtral-8*22b']:
            return deepinfra_res(prompt, model)
        elif model in ['glm4']:
            return zhipu_res(prompt, model)
        elif model in ['qwen-turbo']:
            return qwen_res(prompt)
        else:
            raise ValueError('No model')

    if 'prompt' in el:
        el['answer'] = model_generate(el['prompt'], model)

    elif 'prompt1' in el and 'prompt2' in el:
        answer1 = model_generate(el['prompt1'], model)
        answer2 = model_generate(el['prompt2'], model)
        el['answer1'] = answer1
        el['answer2'] = answer2 
    else:
        raise KeyError('Prompt missing')
    return el

def process_file(model, eval_type, sub_task, file):
    base_path = os.path.join('result', model, sub_task)

    if not os.path.exists(base_path):
        os.makedirs(base_path)

    result_file_path = os.path.join(base_path, file.replace('.json', '_res.json'))

    if os.path.exists(result_file_path):
        with open(result_file_path, 'r') as f:
            save_data = json.load(f)
    else:
        save_data = []

    source_file_path = os.path.join(eval_type, file)
    if os.path.exists(source_file_path):
        with open(source_file_path, 'r') as f:
            test_data = json.load(f)
    else:
        test_data = []
        print(f"Warning: Source file {source_file_path} does not exist.")

    if model in ['gpt-3.5', 'gpt-4']:
        max_worker = 5
    else:
        max_worker = 1


    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        futures = {executor.submit(process_prompt, el, model): el for el in test_data if
                   el['prompt'] not in [k['prompt'] for k in save_data]}
        for future in tqdm(futures):
            el = futures[future]
            try:
                result = future.result()
                save_data.append(result)
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    with open(os.path.join('result', model, file.replace('.json', '_res.json')), 'w') as f:
        json.dump(save_data, f, indent=4, ensure_ascii=False)

    print(f'Finish {file}')

def load_config(config_path):
    try:
        with open(config_path, 'r') as file:
            loaded_yaml = yaml.safe_load(file)
            return loaded_yaml if loaded_yaml is not None else {}
    except FileNotFoundError:
        logging.error(f"The file {config_path} does not exist.")
        return {}
    except yaml.YAMLError as exc:
        logging.error(f"Error parsing YAML file: {exc}")
        return {}

def validate_and_get_config(config, category, supported_list):
    category_items = config.get(category, supported_list)
    if category_items is None:
        logging.warning(f"No items specified for {category}, defaulting to all supported items.")
        return supported_list
 
    invalid_items = [item for item in category_items if item not in supported_list]
    for item in invalid_items:
        logging.warning(f"{item} not supported")
    return [item for item in category_items if item in supported_list]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process models, dimensions, sub-tasks, and reliability from YAML.")
    parser.add_argument('config_file', nargs='?', default='', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config_file) if args.config_file else {}

    models = validate_and_get_config(config, 'models', supported_models)
    dimensions = validate_and_get_config(config, 'dimensions', supported_dimensions)
    sub_tasks = validate_and_get_config(config, 'sub-tasks', supported_sub_tasks)
    reliability_measures = validate_and_get_config(config, 'reliability', supported_reliability)

    for model in models:
        for dimension in dimensions:
            dimension_path = f"dataset/{dimension}"
            directory_names = [name for name in os.listdir(dimension_path) if os.path.isdir(os.path.join(dimension_path, name))]
            
            for sub_task in sub_tasks:
                for directory_name in directory_names:
                    sub_task_path = os.path.join(dimension_path, directory_name, sub_task)
                    if os.path.exists(sub_task_path):
                        file_list = []
                        for filename in os.listdir(sub_task_path):
                            # Check if the filename contains parentheses.
                            if '(' in filename and ')' in filename:
                                start = filename.find('(') + 1
                                end = filename.find(')')
                                content = filename[start:end]
                                # Add file if the content within parentheses is in reliability_measures.
                                if content in reliability_measures:
                                    file_list.append(filename)
                            elif '(' not in filename and ')' not in filename:
                                # Add file if there are no parentheses.
                                file_list.append(filename)

                        for file in file_list:
                            process_file(model, dimension, sub_task, file)