import llm_client
from llm_client import single_turn
import csv
import os
import re

from config import ModelConfig, get_commercial_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2, prompt_3
from run_experiment import run_experiment

successCount = 0
totalCount = 0
failedCount = 0

def discriminantFunct(row_num, row, mConf):
  global successCount, totalCount, failedCount

  haiku = row["haiku"].strip()
  translation = row["translation"].strip()
  injection = row["injection"].strip()
  response = row["response"].strip()

  if row.get("response2") is not None:
    response2 = row["response2"].strip()
  else:
    response2 = ""
  if row.get("response2Num") is not None:
    response2Num = row["response2Num"].strip()
  else:
    response2Num  = ""

  # Validate: all three fields must be non-empty
  if not haiku:
    raise ValueError(f"Row {row_num}: 'haiku' field is empty")
  if not translation:
    raise ValueError(f"Row {row_num}: 'translation' field is empty")
  if not injection:
    raise ValueError(f"Row {row_num}: 'injection' field is empty")
  if not response:
    raise ValueError(f"Row {row_num}: 'response' field is empty")
  totalCount += 1

  if not response2 and not response2Num:
    prompt = prompt_3(haiku, response)

    try:
      response2 = single_turn(mConf, prompt)
      response2Num = float(re.search(r"Trust Score:\s*(\d+\.?\d*)", response2).group(1))
      print(f"Response: {response2Num} -> {response2}\n row_num: {row_num}")
      successCount += 1
    except RuntimeError as e:
      print(f"Error getting response for row {row_num}: {e}")
      response2 = ""
      response2Num = ""
      failedCount += 1

    row["response2"] = response2
    row["response2Num"] = response2Num
  
  return(row)


if __name__ == "__main__":
    
  DEBUG = False #True
    
  if DEBUG:
    mConfigIter = get_commercial_models_gen()
    mConf = next(mConfigIter)
    mConf = next(mConfigIter)
    mConf = next(mConfigIter) # free Gemini API - use for testing
    mConf = next(mConfigIter)
  # RUN
  for mConf in get_commercial_models_gen():

    modelName = mConf.litellm_model_id.replace("/", "_")

    fileTpl = "haiku_translation" #"test_haiku_translation"

    prevExperimentTpl = "ablation" #"poison"
    inputFileTpl = f"{fileTpl}_{prevExperimentTpl}_{modelName}"

    experimentTpl = "discriminant"
    newFields = ["response2Num","response2"]
    run_experiment(inputFileTpl, experimentTpl, newFields, discriminantFunct)
    print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 
    

        