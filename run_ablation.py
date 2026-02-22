import llm_client
from llm_client import single_turn
import csv
import os

from config import ModelConfig, get_commercial_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2
from run_experiment import run_experiment


def ablationFunct(row_num, row, mConf):

  haiku = row["haiku"].strip()
  translation = row["translation"].strip()
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
  if not injection:
    raise ValueError(f"Row {row_num}: 'injection' field is empty")

  if not response:
    prompt = prompt_1(haiku)
    #prompt = prompt_2(haiku, injection)

    try:
      response = single_turn(mConf, prompt)
      print(f"Response: {response}")
    except RuntimeError as e:
      print(f"Error getting response for row {row_num}: {e}")
      response = ""

    row["response"] = response
  
  return(row)


if __name__ == "__main__":
    
    fileTpl = "test_haiku_translation"
    experimentTpl = "ablation"
    newFields = "response"
    run_experiment(fileTpl, experimentTpl, newFields, ablationFunct)
    

        