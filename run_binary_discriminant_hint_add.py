import llm_client
from llm_client import single_turn
import re

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2, prompt_b3
from run_binary_experiment_add import run_binary_experiment_add
from run_binary_discriminant_hint import binaryDiscriminantHintFunct

successCount = 0
totalCount = 0
failedCount = 0


if __name__ == "__main__":
    
  DEBUG = False #True
    
  for mConf in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    modelName = mConf.litellm_model_id.replace("/", "_")

    fileTpl = "haiku_translation" #"test_haiku_translation"
    fileTpl2 = "haiku_translation_add"

    prevExperimentTpl1 = "ablation"
    inputFileTpl1 = f"{fileTpl}_{prevExperimentTpl1}_{modelName}"
    inputFileTpl1a = f"{fileTpl2}_{prevExperimentTpl1}_{modelName}"

    prevExperimentTpl2 = "poison"
    inputFileTpl2 = f"{fileTpl}_{prevExperimentTpl2}_{modelName}"
    inputFileTpl2a = f"{fileTpl2}_{prevExperimentTpl2}_{modelName}"

    experimentTpl = "binary_discriminant_hint"
    newFields = ["response2","response3Num","response3","prevExp1","prevExp2"]
    run_binary_experiment_add(inputFileTpl1, inputFileTpl1a, inputFileTpl2, inputFileTpl2a, fileTpl, fileTpl2, modelName, prevExperimentTpl1, prevExperimentTpl2, experimentTpl, newFields, binaryDiscriminantHintFunct)
    print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 