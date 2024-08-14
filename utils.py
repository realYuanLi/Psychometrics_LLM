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