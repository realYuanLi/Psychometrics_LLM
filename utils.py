import numpy as np
import pandas as pd
import os
import re
import json
import string

letters = string.ascii_lowercase

def find_first_number(text):
    # This regular expression pattern will match any sequence of digits
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())  # Returns the first occurrence of a digit sequence
    else:
        return "No numbers found"

def parse_score(text):
    # This regular expression matches numbers from 0 to 100
    match = re.search(r'\b(100|[1-9]?[0-9])\b', text)
    if match:
        return int(match.group())
    else:
        return "No numbers found"
  
def parse_option(text):
    # Regular expression to find (A) or (B)
    match = re.search(r'\(A\)|\(B\)', text)
    if match:
        return match.group()
    else:
        return None

def parse_binary(text):
    normalized_answer = text.strip().lower()

    if re.search(r'\byes\b', normalized_answer):
        return "Yes"
    elif re.search(r'\bno\b', normalized_answer):
        return "No"
    else:
        return "No binary result found"
    
def calculate_matching_rate(directory_path, file1, file2):
    path1 = os.path.join(directory_path, file1)
    path2 = os.path.join(directory_path, file2)
    
    with open(path1, 'r') as f1, open(path2, 'r') as f2:
        data1 = json.load(f1)
        data2 = json.load(f2)
        
        correctness1 = [1 if el['answer'] == el['label'] else 0 for el in data1]
        correctness2 = [1 if el['answer'] == el['label'] else 0 for el in data2]
        
        matches = sum(1 for i, correct in enumerate(correctness1) if correct == correctness2[i])
        total = len(correctness1)
        
        return matches / total if total > 0 else 0

def agreement_score(s1, s2):
    """
    Compute the agreement score between two scores assigned by two raters.

    Parameters:
    s1 (int): Score assigned by rater 1.
    s2 (int): Score assigned by rater 2.

    Returns:
    float: The agreement score as a percentage (100%, 50%, or 0%).
    """
    difference = abs(s1 - s2)
    if difference == 0:
        return 100.0  # 100% agreement when scores are identical
    elif difference == 1:
        return 50.0   # 50% agreement when scores differ by 1
    else:
        return 0.0    # 0% agreement otherwise

def agreement_rate(scores1, scores2):
    if len(scores1) != len(scores2):
        raise ValueError("Both raters must have scored the same number of items.")
    
    total_agreement = sum(agreement_score(s1, s2) for s1, s2 in zip(scores1, scores2))
    overall_ar = total_agreement / len(scores1)  # Average agreement rate
    return overall_ar
 
def parse_emotion_output(pred, choices, task):
    try:
        pred = pred.lower().replace("（", "(").replace("）", ")").replace(".", "")
        choices = [
            choice.replace(" & ", " and ")
            for choice in choices
        ]
        lines = pred.split("\n")
        for j in range(len(lines)):
            output = lines[len(lines) - 1 - j]
            if output:
                alphabets = {
                    "normal": [
                        f"({letters[i]})" for i in range(4 if task == "EA" else 6)
                    ],
                    "paranthese": [
                        f"[{letters[i]}]" for i in range(4 if task == "EA" else 6)
                    ],
                    "dot": [f": {letters[i]}" for i in range(4 if task == "EA" else 6)],
                    "option": [
                        f"option {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "option1": [
                        f"option ({letters[i]})"
                        for i in range(4 if task == "EA" else 6)
                    ],
                    "choice": [
                        f"choice {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "choice1": [
                        f"choice ({letters[i]})"
                        for i in range(4 if task == "EA" else 6)
                    ],
                    "选项": [
                        f"选项 {letters[i]}" for i in range(4 if task == "EA" else 6)
                    ],
                    "选项1": [
                        f"选项 ({letters[i]})" for i in range(4 if task == "EA" else 6)
                    ],
                }

                for v in alphabets.values():
                    for a in v:
                        if a in output:
                            return v.index(a)
                for c in choices:
                    if c.lower() in output:
                        return choices.index(c)
                if len(output) == 1 and output in letters[: 4 if task == "EA" else 6]:
                    return letters.index(output)
                if output[0] in letters[: 4 if task == "EA" else 6] and output[1] in [
                    "<",
                    "[",
                    "(",
                    ")",
                    ":",
                ]:
                    return letters.index(output[0])
    except Exception as e:
        print("Error in processing output", type(e).__name__, "–", e)

    return -1