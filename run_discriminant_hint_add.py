import llm_client
from llm_client import single_turn
import csv
import os
import re

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2, prompt_3
from run_experiment_add import run_experiment_add
from run_discriminant_hint import discriminantHintFunct

successCount = 0
totalCount = 0
failedCount = 0




if __name__ == "__main__":
    
  DEBUG = False #True
    
  for mConf in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    modelName = mConf.litellm_model_id.replace("/", "_")

    fileTpl = "haiku_translation2l" #"test_haiku_translation"
    fileTpl2 = "haiku_translation2l_add" 

    prevExperimentTpl = "poison" #"ablation" #"poison"
    inputFileTpl = f"{fileTpl}_{prevExperimentTpl}_{modelName}"
    inputFileTpl2 = f"{fileTpl2}_{prevExperimentTpl}_{modelName}"

    experimentTpl = "discriminant_hint"
    newFields = ["response2Num","response2"]
    run_experiment_add(inputFileTpl, inputFileTpl2, experimentTpl, newFields, discriminantHintFunct)
    print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 