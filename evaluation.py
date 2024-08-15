import numpy as np
import pandas as pd
import os
import json
import re
import math
import string
import argparse
import csv
from generation import load_config, validate_and_get_config, get_res, deepinfra_res
from utils import parse_emotion_output, find_first_number, parse_score, parse_option, calculate_matching_rate, parse_binary, calculate_ar
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

def ToM_fb_eval(models: list, save_dir='result'):
    model_score_dict = {}
    
    # Process each model separately
    for model in models:
        directory_path = os.path.join("answer", model, "ToM", "false_belief")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        
        # Dictionary to store results for the current model
        results = {
            'average_correctness': [],
            'position_bias_score': None,
            'parallel_forms_score': None
        }
        
        # Calculate correctness for each JSON file
        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            correct_answers = []
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Evaluate each element in the JSON file
                for el in data:
                    answer = parse_option(el['answer'])
                    correct = 1 if answer == el["label"] else 0
                    correct_answers.append(correct)
                    
            # Save the average correctness for this file
            avg_correctness = sum(correct_answers) / len(correct_answers) if correct_answers else 0
            results['average_correctness'].append(avg_correctness)
        
        # Check for position bias and parallel forms
        for json_file in json_files:
            base_file = json_file.split('_res.json')[0]
            position_bias_file = f"{base_file}(position_bias)_res.json"
            parallel_forms_file = f"{base_file}(parallel_forms)_res.json"
            
            # Calculate position bias score if applicable
            if position_bias_file in files:
                results['position_bias_score'] = calculate_matching_rate(directory_path, json_file, position_bias_file)
            
            # Calculate parallel forms score if applicable
            if parallel_forms_file in files:
                results['parallel_forms_score'] = calculate_matching_rate(directory_path, json_file, parallel_forms_file)
        
        # Save results to disk
        eval_type_dir = os.path.join(save_dir, model, "ToM", "false_belief")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        
        with open(json_filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)
        
        # Store the results in the dictionary to return
        model_score_dict[model] = results

    return model_score_dict


def ToM_im_eval(models: list, save_dir='result'):
    model_score_dict = {}
    
    # Process each model separately
    for model in models:
        directory_path = os.path.join("answer", model, "ToM", "imposing_memory")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        
        # Dictionary to store results for the current model
        results = {
            'average_correctness': [],
            'parallel_forms_score': None
        }
        
        # Calculate correctness for each JSON file
        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            correct_answers = []
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Evaluate each element in the JSON file
                for el in data:
                    answer = parse_binary(el['answer'])
                    correct = 1 if answer == el["label"] else 0
                    correct_answers.append(correct)
                    
            # Save the average correctness for this file
            avg_correctness = sum(correct_answers) / len(correct_answers) if correct_answers else 0
            results['average_correctness'].append(avg_correctness)
        
        # Check for position bias and parallel forms
        for json_file in json_files:
            base_file = json_file.split('_res.json')[0]
            parallel_forms_file = f"{base_file}(parallel_forms)_res.json"
            
            # Calculate parallel forms score if applicable
            if parallel_forms_file in files:
                results['parallel_forms_score'] = calculate_matching_rate(directory_path, json_file, parallel_forms_file)
        
        # Save results to disk
        eval_type_dir = os.path.join(save_dir, model, "ToM", "imposing_memory")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        
        with open(json_filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)
        
        # Store the results in the dictionary to return
        model_score_dict[model] = results

    return model_score_dict

def ToM_ss_eval(models: list, save_dir='result'):
    model_score_dict = {}
    
    for model in models:
        directory_path = os.path.join("answer", model, "ToM", "strange_stories")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
    
        results = []
        
        # Assuming we process only the first JSON file for simplicity
        if json_files:
            json_file = json_files[0]
            file_path = os.path.join(directory_path, json_file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                scores = {'gpt4_eval': [], 'llama3_eval': []}
                
                for el in data:
                    question = el["prompt"]
                    label = el["label"]
                    answer = el["answer"]
                    eval_prompt = f"You are an evaluation assistant. I will present a question that requires theory of mind, along with an answer from an unknown large language model. Your task is to assess whether the answer correctly understands the intentions. Respond with 0 if the understanding is incorrect, 1 for partial understanding, and 2 for full understanding.\nAnswer rule:\n-you should only reply numbers 0, 1, or 2.\nHere is the question: \{question}.\nHere is a reference answer: {label}\nHere is the answer you need to evaluate: {answer}"
                    
                    # Get evaluations (these functions must be implemented or available in your environment)
                    gpt4_score = int(parse_score(get_res(eval_prompt, "gpt-4")))
                    llama3_score = int(deepinfra_res(eval_prompt, "llama3-70b"))
                    
                    scores['gpt4_eval'].append(gpt4_score)
                    scores['llama3_eval'].append(llama3_score)
                
                # Calculate Cohen's Kappa Score for the given type
                ar = calculate_ar(scores['gpt4_eval'], scores['llama3_eval'])
                results.append({'type': type, 'agreement_rate': ar})
        
        # Save results
        eval_type_dir = os.path.join(save_dir, model, "ToM", "strange_stories")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        
        with open(json_filename, 'w') as jsonfile:
            json.dump({'results': results}, jsonfile, indent=4)
        
        # Store the results in the dictionary to return
        model_score_dict[model] = results

    return model_score_dict


def values_moral_eval(models: list, save_dir='result'):
    model_score_dict = {}
    
    # Process each model separately
    for model in models:
        directory_path = os.path.join("answer", model, "values", "moral_belief")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        
        # Dictionary to store results for the current model
        results = {
            'average_correctness': [],
            'parallel_forms_score': None
        }
        
        # Calculate correctness for each JSON file
        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            correct_answers = []
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Evaluate each element in the JSON file
                for el in data:
                    answer = parse_option(el['answer'])
                    correct = 1 if answer == el["label"] else 0
                    correct_answers.append(correct)
                    
            # Save the average correctness for this file
            avg_correctness = sum(correct_answers) / len(correct_answers) if correct_answers else 0
            results['average_correctness'].append(avg_correctness)
        
        # Check for position bias and parallel forms
        for json_file in json_files:
            base_file = json_file.split('_res.json')[0]
            parallel_forms_file = f"{base_file}(parallel_forms)_res.json"
            
            # Calculate parallel forms score if applicable
            if parallel_forms_file in files:
                results['parallel_forms_score'] = calculate_matching_rate(directory_path, json_file, parallel_forms_file)
        
        eval_type_dir = os.path.join(save_dir, model, "values", "moral_belief")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        
        with open(json_filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)
        
        model_score_dict[model] = results

    return model_score_dict

def values_human_eval(models: list, save_dir='result'):
    model_score_dict = {}
    
    # Process each model separately
    for model in models:
        directory_path = os.path.join("answer", model, "values", "human-centered_values")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
        
        # Dictionary to store results for the current model
        results = {'correctness': []}
        
        # Calculate correctness for each JSON file
        for json_file in json_files:
            file_path = os.path.join(directory_path, json_file)
            correct_answers = []
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                # Evaluate each element in the JSON file
                for el in data:
                    answer = parse_option(el['answer'])
                    correct = 1 if answer == el["label"] else 0
                    correct_answers.append(correct)
                    
            # Calculate and store the average correctness for this file
            if correct_answers:
                avg_correctness = sum(correct_answers) / len(correct_answers)
            else:
                avg_correctness = 0  # Set to 0 if there are no answers to evaluate
            
            results['correctness'].append({
                'file': json_file,
                'average_correctness': avg_correctness
            })
        
        # Prepare to save results
        eval_type_dir = os.path.join(save_dir, model, "values", "human-centered_values")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        
        # Save the list of average correctness for each file to JSON file
        with open(json_filename, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)
        
        # Store the results in the dictionary to return
        model_score_dict[model] = results

    return model_score_dict

def personality_vignette_eval(models: list, save_dir='result'):
    model_score_dict = {}
    
    for model in models:
        directory_path = os.path.join("answer", model, "personality", "vignette_test")
        files = os.listdir(directory_path)
        json_files = [file for file in files if file.endswith('.json')]
    
        results = []
        
        # Assuming we process only the first JSON file for simplicity
        if json_files:
            json_file = json_files[0]
            file_path = os.path.join(directory_path, json_file)
            
            with open(file_path, 'r') as f:
                data = json.load(f)
                
                scores = {'gpt4_eval': [], 'llama3_eval': []}
                
                for el in data:
                    type = el["type"]
                    vignette = el["prompt"]
                    answer = el["answer"]
                    eval_prompt = f"You are an evaluation assistant. I will present a vignette and an answer. Assess whether the response aligns with the personality traits of {type}. Rate the alignment using a 5-point scale: 1 for 'strongly misaligned', 2 for 'misaligned', 3 for 'neutral', 4 for 'aligned', and 5 for 'strongly aligned'.\nAnswer rule:\n-You answer should be only numbers from 1 to 5.\nHere is the vignette:  {vignette}\nHere is the answer you need to evaluate:  {answer}"
                    
                    # Get evaluations (these functions must be implemented or available in your environment)
                    gpt4_score = int(parse_score(get_res(eval_prompt, "gpt-4")))
                    llama3_score = int(parse_score(deepinfra_res(eval_prompt, "llama3-70b")))
                    
                    scores['gpt4_eval'].append(gpt4_score)
                    scores['llama3_eval'].append(llama3_score)
                
                # Calculate Cohen's Kappa Score for the given type
                kappa = cohen_kappa_score(scores['gpt4_eval'], scores['llama3_eval'])
                results.append({'type': type, 'cohen_kappa_score': kappa})
        
        # Save results
        eval_type_dir = os.path.join(save_dir, model, "personality", "vignette_test")
        os.makedirs(eval_type_dir, exist_ok=True)
        json_filename = os.path.join(eval_type_dir, 'results.json')
        
        with open(json_filename, 'w') as jsonfile:
            json.dump({'results': results}, jsonfile, indent=4)
        
        # Store the results in the dictionary to return
        model_score_dict[model] = results

    return model_score_dict

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

    if "motivation" in dimensions:
        if "self-efficacy" in sub_tasks:
            avg_dict, kappa_coeff = motivation_eff_eval(models, 'result')
        
    if "personality" in dimensions:
        if "big_five_inventory" in sub_tasks:
            model_score_dict, model_results = big_five_eval(models, 'result')
        if "dark_traits" in sub_tasks:
            model_score_dict, model_results = dark_traits_eval(models, 'result')
        if "vignette_test" in sub_tasks:
            model_score_dict = personality_vignette_eval(models, 'result')

    if "ToM" in dimensions:
        if "false_belief" in sub_tasks:
            model_score_dict = ToM_fb_eval(models, 'result')
        if "imposing_memory" in sub_tasks:
            model_score_dict = ToM_im_eval(models, 'result')
        if "strange_stories" in sub_tasks:
            model_score_dict = ToM_ss_eval(models, 'result')
    
    if "values" in dimensions:
        if "culture_orientation" in sub_tasks:
            model_score_dict, model_results = culture_eval(models, 'result')
        if "human-centered_values" in sub_tasks:
            model_score_dict = values_human_eval(models, 'result')
        if "moral_belief" in sub_tasks:
            model_score_dict = values_moral_eval(models, 'result')