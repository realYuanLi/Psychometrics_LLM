import numpy as np
import pandas as pd
import os
import json
import re
import math
import string
import argparse
import csv
from generation import load_config, validate_and_get_config
from utils import parse_emotion_output, find_first_number, parse_score
from sklearn.metrics import cohen_kappa_score

BIG_FIVE_REFERENCE = "reference/raw_big_five.json"
DARK_TRAITS_REFERENCE = "reference/dark_traits_raw.json"

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


def big_five_eval(models: list, save_dir='result'):
    with open(BIG_FIVE_REFERENCE, 'r') as f:
        reference_data = json.load(f)
        reverse_question = reference_data['reverse']
        Extraversion_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Extraversion'][0]
        Agreeableness_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Agreeableness'][0]
        Conscientiousness_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Conscientiousness'][0]
        Neuroticism_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Neuroticism'][0]
        Openness_question = [el['cat_questions'] for el in reference_data['categories'] if el['cat_name'] == 'Openness'][0]

    model_score_dict = {}
    for model in models:
        model_score_dict[model] = {"Extraversion": [], "Agreeableness": [], "Conscientiousness": [], "Neuroticism": [], "Openness": []}
        directory_path = os.path.join("answer", model, "personality", "big_five_inventory")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        file_path = os.path.join(directory_path, json_files[0])
        with open(file_path, 'r') as f:
            data = json.load(f)
        for el in data:
            number = find_first_number(el['answer'])
            if number != 'No numbers found':
                el['index'] = int(el['index'])
                number = int(number)
                if el['index'] in reverse_question:
                    number = 6 - number
                if el['index'] in Extraversion_question:
                    model_score_dict[model]['Extraversion'].append(number)
                elif el['index'] in Agreeableness_question:
                    model_score_dict[model]['Agreeableness'].append(number)
                elif el['index'] in Conscientiousness_question:
                    model_score_dict[model]['Conscientiousness'].append(number)
                elif el['index'] in Neuroticism_question:
                    model_score_dict[model]['Neuroticism'].append(number)
                elif el['index'] in Openness_question:
                    model_score_dict[model]['Openness'].append(number)
                else:
                    print(el['index'])
                    raise ValueError('No Dimension!')

    # Calculate avg and std for each dimension
    model_results = {}
    for model in models:
        eval_type_dir = os.path.join(save_dir, model, "personality", "big_five_inventory")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        with open(json_filename, 'w') as jsonfile:
            json.dump(model_results[model], jsonfile, indent=4)

    return model_score_dict, model_results


def dark_traits_eval(models: list, save_dir='result'):
    with open(DARK_TRAITS_REFERENCE, 'r') as f:
        reference_data = json.load(f)
        Machiavellianism_question = [el.strip('(R)') for el in reference_data['Machiavellianism']]
        Narcissism_question = [el.strip('(R)') for el in reference_data['Narcissism']]
        Psychopathy_question = [el.strip('(R)') for el in reference_data['Psychopathy']]

    model_score_dict = {}
    for model in models:
        model_score_dict[model] = {"Machiavellianism": [], "Narcissism": [], "Psychopathy": []}
        directory_path = os.path.join("answer", model, "personality", "dark_traits")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        file_path = os.path.join(directory_path, json_files[0])
        with open(file_path, 'r') as f:
            data = json.load(f)
        for el in data:
            number = find_first_number(el['answer'])
            if number != 'No numbers found':
                number = int(number)
                if el['reverse']:
                    number = 6 - number
                if el['question'] in Machiavellianism_question:
                    model_score_dict[model]['Machiavellianism'].append(number)
                elif el['question'] in Narcissism_question:
                    model_score_dict[model]['Narcissism'].append(number)
                elif el['question'] in Psychopathy_question:
                    model_score_dict[model]['Psychopathy'].append(number)

    model_results = {}
    for model in models:
        eval_type_dir = os.path.join(save_dir, model, "personality", "dark_traits")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        with open(json_filename, 'w') as jsonfile:
            json.dump(model_results[model], jsonfile, indent=4)

    return model_score_dict, model_results


def emotion_EA_eval(models: list, save_dir='result'):
    model_score_dict = {}
    for model in models:
        model_score_dict[model] = []
        directory_path = os.path.join("answer", model, "emotion", "EA")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        file_path = os.path.join(directory_path, json_files[0])
        with open(file_path, 'r') as f:
            data = json.load(f)
            for el in data:
                answer = parse_emotion_output(el['answer'], el['choices'], 'EA')
                model_score_dict[model].append(1 if answer == el['label'] else 0)

    avg_dict = {}
    for model in models:
        avg_dict[model] = sum(model_score_dict[model]) / len(model_score_dict[model])
    print(avg_dict)
    
    for model in models:
        eval_type_dir = os.path.join(save_dir, model, "emotion", "EA")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        with open(json_filename, 'w') as jsonfile:
            json.dump(avg_dict, jsonfile, indent=4)

    return model_score_dict, avg_dict


def emotion_EU_eval(models: list, save_dir='result'):
    model_score_dict = {}
    for model in models:
        model_score_dict[model] = []
        directory_path = os.path.join("answer", model, "emotion", "EU")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        file_path = os.path.join(directory_path, json_files[0])
        with open(file_path, 'r') as f:
            data = json.load(f)
            for el in data:
                answer = parse_emotion_output(el['res'], el['choices'], 'EU')
                if el['emotion_label'] in el['choices']:
                    question_type = 'emotion_label'
                else:
                    question_type = 'cause_label'
                model_score_dict[model].append(1 if answer == el['choices'].index(el[question_type]) else 0)

    avg_dict = {}
    for model in models:
        avg_dict[model] = sum(model_score_dict[model]) / len(model_score_dict[model])
    print(avg_dict)
    
    for model in models:
        eval_type_dir = os.path.join(save_dir, model, "emotion", "EU")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        with open(json_filename, 'w') as jsonfile:
            json.dump(avg_dict, jsonfile, indent=4)

    return model_score_dict, avg_dict


def culture_eval(models: list, file: str, save_dir='result'):
    model_score_dict = {}
    for model in models:
        model_score_dict[model] = {}
        directory_path = os.path.join("answer", model, "values", "culture_orientation")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        file_path = os.path.join(directory_path, json_files[0])
        with open(file_path, 'r') as f:
            data = json.load(f)
            for el in data:
                if el['dimension'] not in model_score_dict[model]:
                    model_score_dict[model][el['dimension']] = []
                res_number = find_first_number(el['res'])
                if res_number != 'No numbers found':
                    model_score_dict[model][el['dimension']].append(float(res_number))

    # Calculate avg and std
    model_results = {}
    for model in models:
        model_results[model] = {}
        for dimension in model_score_dict[model]:
            numbers = np.array(model_score_dict[model][dimension])
            if numbers.size > 0:
                avg = numbers.mean()
                std = numbers.std()
            else:
                avg = None
                std = None
            model_results[model][dimension] = {'avg': avg, 'std': std}

    print(model_results)
    
    for model in models:
        eval_type_dir = os.path.join(save_dir, model, "values", "culture_orientation")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        with open(json_filename, 'w') as jsonfile:
            json.dump(model_results[model], jsonfile, indent=4)

    return model_score_dict, model_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process models, dimensions, sub-tasks, and reliability from YAML.")
    parser.add_argument('config_file', nargs='?', default='', type=str, help='Path to the YAML configuration file')
    args = parser.parse_args()

    config = load_config(args.config_file) if args.config_file else {}

    models = validate_and_get_config(config, 'models', supported_models)
    dimensions = validate_and_get_config(config, 'dimensions', supported_dimensions)
    sub_tasks = validate_and_get_config(config, 'sub-tasks', supported_sub_tasks)
    reliability_measures = validate_and_get_config(config, 'reliability', supported_reliability)

    if "emotion" in dimensions:
        if "EA" in sub_tasks:
            model_score_dict, avg_dict = emotion_EA_eval(models, 'result')
        if "EU" in sub_tasks:
            model_score_dict, avg_dict = emotion_EU_eval(models, 'result')

def motivation_eff_eval(models: list, save_dir='result'):
    model_score_dict = {}
    for model in models:
        model_score_dict[model] = {}
        directory_path = os.path.join("answer", model, "motivation", "self-efficacy")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        model_score_dict = {model: {"type_1": [], "type_2": []} for model in models}
    avg_dict = {}

    for model in models:
        # Reading the first JSON file
        file_path = os.path.join(directory_path, json_files[0])
        with open(file_path, 'r') as f:
            data = json.load(f)
            for el in data:
                answer = parse_score(el['answer'])
                model_score_dict[model][el['type']].append(answer)

        # If there are two files, read the second and compute averages
        if len(json_files) > 1:
            file_path = os.path.join(directory_path, json_files[1])
            with open(file_path, 'r') as f:
                data = json.load(f)
                for i, el in enumerate(data):
                    answer = parse_score(el['answer'])
                    model_score_dict[model][el['type']].append(answer)
                    avg_score = sum(model_score_dict[model][el['type']][-2:]) / 2
                    avg_dict[el['type']] = avg_score

            # Calculate weighted Kappa coefficient
            kappa_coeff = cohen_kappa_score(
                model_score_dict[model]["type_1"][-2:],
                model_score_dict[model]["type_2"][-2:],
                weights='quadratic'
            )
        else:
            # If there's only one file, output the current dictionary
            avg_dict = {el['type']: scores[0] for el, scores in model_score_dict[model].items()}
            kappa_coeff = ""

        # Save results
        eval_type_dir = os.path.join(save_dir, model, "motivation", "self-efficacy")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        with open(json_filename, 'w') as jsonfile:
            json.dump({"average": avg_dict, "kappa_coeff": kappa_coeff}, jsonfile, indent=4)

    return avg_dict, kappa_coeff
    
    for model in models:
        eval_type_dir = os.path.join(save_dir, model, "motivation", "self-efficacy")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        with open(json_filename, 'w') as jsonfile:
            json.dump(avg_dict, jsonfile, indent=4)

    return model_score_dict, avg_dict


    if "motivation" in dimensions:
        if "self-efficacy" in sub_tasks:
            read json file with filename without "(" and ")"

            if "parallel_form" in reliability_measures:
                do parsing on 

        
    if "personality" in dimensions:
        if "big_five_inventory" in sub_tasks:
            model_score_dict, model_results = big_five_eval(models, 'result')
        if "dark_traits" in sub_tasks:
            model_score_dict, model_results = dark_traits_eval(models, 'result')
        if "vignette_test" in sub_tasks:
            break

    if "ToM" in dimensions:
        if "false_belief" in sub_tasks:
            break
        if "imposing_memory" in sub_tasks:
            break
        if "strange_stories" in sub_tasks:
            break
    if "values" in dimensions:
        if "culture_orientation" in sub_tasks:
            model_score_dict, model_results = culture_eval(models, 'result')
        if "human-centered_values" in sub_tasks:
            break
        if "moral_belief" in sub_tasks:
            break