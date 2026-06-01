import llm_client
from llm_client import single_turn
import csv
import os

from config import ModelConfig, get_commercial_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2
from run_experiment_add import run_experiment_add
from run_poison import poisonFunct

successCount = 0
totalCount = 0
failedCount = 0



if __name__ == "__main__":
    
    fileTpl = "haiku_translation2l" #"test_haiku_translation"
    fileTpl2 = "haiku_translation2l_add" 
    experimentTpl = "poison" #"ablation"
    newFields = "response"
    run_experiment_add(fileTpl, fileTpl2, experimentTpl, newFields, poisonFunct)
    print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 