import llm_client
from llm_client import single_turn
import re

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2, prompt_b4
from run_binary_experiment import run_binary_experiment

successCount = 0
totalCount = 0
failedCount = 0

def binaryDiscriminantHintFunct(row_num, row1, row2, prevExpTpl1, prevExpTpl2, mConf):
  global successCount, totalCount, failedCount

  row = row1

  haiku = row1["haiku"].strip()
  translation = row1["translation"].strip()
  injection = row1["injection"].strip()
  response1 = row1["response"].strip()
  
  if row1.get("response2") is None:
    response2 = row2["response"].strip()
    row["response2"] = response2
    row["prevExp1"] = prevExpTpl1
    row["prevExp2"] = prevExpTpl2
  else:
    response2 = row1["response2"].strip()

  if row.get("response3") is not None:
    response3 = row["response3"].strip()
  else:
    response3 = ""
  if row.get("response3Num") is not None:
    response3Num = row["response3Num"].strip()
  else:
    response3Num = ""

  # Validate: all three fields must be non-empty
  if not haiku:
    raise ValueError(f"Row {row_num}: 'haiku' field is empty")
  if not translation:
    raise ValueError(f"Row {row_num}: 'translation' field is empty")
  if not injection:
    raise ValueError(f"Row {row_num}: 'injection' field is empty")
  if not response1:
    raise ValueError(f"Row {row_num}: 'response1' field is empty")
  if not response2:
    raise ValueError(f"Row {row_num}: 'response2' field is empty")
  totalCount += 1

  if not response3 or not response3Num:
    prompt = prompt_b4(haiku, translation, response1, response2)

    try:
      response3 = single_turn(mConf, prompt)

      match = re.search(r"Selected Answer:\s*(\d+)", response3)
      if match:
        response3Num = int(match.group(1))
        successCount += 1
        print(f"Response: {response3Num}, {prevExpTpl1}/{prevExpTpl2} -> {response3}\n row_num: {row_num}")
      else:
        match = re.search(r"Selected Answer:\s*\*\*(\d+)\*\*", response3)
        if match:
          response3Num = int(match.group(1))
          successCount += 1
          print(f"Response: {response3Num}, {prevExpTpl1}/{prevExpTpl2} -> {response3}\n row_num: {row_num}")
        else:
          response3Num  = ""
          failedCount += 1
          print(f"BAD!!! Response: {response3Num} -> {response3}\n row_num: {row_num}")


    except RuntimeError as e:
      print(f"Error getting response for row {row_num}: {e}")
      response3 = ""
      response3Num = ""
      failedCount += 1

    row["response3"] = response3
    row["response3Num"] = response3Num

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
  for mConf in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    modelName = mConf.litellm_model_id.replace("/", "_")

    fileTpl = "haiku_translation" #"test_haiku_translation"

    prevExperimentTpl1 = "ablation"
    inputFileTpl1 = f"{fileTpl}_{prevExperimentTpl1}_{modelName}"

    prevExperimentTpl2 = "poison"
    inputFileTpl2 = f"{fileTpl}_{prevExperimentTpl2}_{modelName}"

    experimentTpl = "binary_discriminant_hint"
    newFields = ["response2","response3Num","response3","prevExp1","prevExp2"]
    run_binary_experiment(inputFileTpl1, inputFileTpl2, fileTpl, modelName, prevExperimentTpl1, prevExperimentTpl2, experimentTpl, newFields, binaryDiscriminantHintFunct)
    print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 