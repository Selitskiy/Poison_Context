import llm_client
from llm_client import single_turn
import csv
import os

from config import ModelConfig, get_commercial_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2
from run_2lsubstitution import run_2lsubstitution

successCount = 0
totalCount = 0
failedCount = 0

def ablationFunct2l(row_num, row, row2, mConf):
  global successCount, totalCount, failedCount

  haiku = row["haiku"].strip()
  translation = row["translation"].strip()
  translation2l = row2["translation"].strip()
  injection = row["injection"].strip()
  if row.get("response") is not None:
    response = row["response"].strip()
  else:
    response = ""

  # Validate: all three fields must be non-empty
  if not haiku:
    raise ValueError(f"Row {row_num}: 'haiku' field is empty")
  if not translation:
    raise ValueError(f"Row {row_num}: 'translation' field is empty")
  if not translation2l:
    raise ValueError(f"Row {row_num}: 'translation2l' field is empty")
  if not injection:
    raise ValueError(f"Row {row_num}: 'injection' field is empty")
  totalCount += 1

  row["translation"] = translation2l

  if not response:
    prompt = prompt_1(haiku)
    #prompt = prompt_2(haiku, injection)

    try:
      response = single_turn(mConf, prompt)
      print(f"Response: {response}\n row_num: {row_num}")
      successCount += 1
    except RuntimeError as e:
      print(f"Error getting response for row {row_num}: {e}")
      response = ""
      failedCount += 1

    row["response"] = response
  
  return(row)


if __name__ == "__main__":
    
    fileTpl = "haiku_translation_add" #"test_haiku_translation"
    fileTpl2 = "haiku_translation2l_add"
    experimentTpl = "ablation"
    newFields = "response"
    run_2lsubstitution(fileTpl, fileTpl2, experimentTpl, newFields, ablationFunct2l)
    print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 
    

        