import llm_client
from llm_client import single_turn
import csv
import os
import random

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry


def run_binary_experiment(fileTpl1, fileTpl2, genFileTpl, genModelName, prevExpTpl1, prevExpTpl2, experimentTpl, newFields, experimentFunct):

  DEBUG = False #True
    
  if DEBUG:
    mConfigIter = get_commercial_models_gen()
    #mConfigIter = get_open_models_gen()
    mConf = next(mConfigIter)
    mConf = next(mConfigIter)
    #mConf = next(mConfigIter) # free Gemini API - use for testing
    #mConf = next(mConfigIter)
  # RUN
  for mConf in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    inputFileName1 = f"{fileTpl1}.csv"
    inputFileFull1 = os.path.join(os.path.dirname(__file__), "data", inputFileName1)
    inputFileName2 = f"{fileTpl2}.csv"
    inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)
    trueInputFiles = True

    if not mConf.api_key:
      print(f"API KEY is empty for {mConf.name}")
      if not DEBUG:
        pass
        #continue
      else:
        exit(1)

    modelName = mConf.litellm_model_id.replace("/", "_")
    outputFileName = f"{genFileTpl}_{genModelName}_{experimentTpl}_{modelName}.csv"
    tmpOutputFileName = f"tmp_{outputFileName}"
    tmpOutputFileFull = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileName)

    outputFileFull = os.path.join(os.path.dirname(__file__), "data", outputFileName)
    if os.path.isfile(outputFileFull):
      inputFileFull = outputFileFull
      trueInputFiles = False
    
    if not os.path.isfile(inputFileFull1):
      print(f"Input file not found: {inputFileFull1}")
      return(1)
    if not os.path.isfile(inputFileFull2):
      print(f"Input file not found: {inputFileFull2}")
      return(1)

    try:

      if trueInputFiles:
        with open(inputFileFull1, encoding="utf-8", newline="") as fh1, open(inputFileFull2, encoding="utf-8", newline="") as fh2:
          reader1 = csv.DictReader(fh1)
          reader2 = csv.DictReader(fh2)

          fieldnames = reader1.fieldnames
          row_num = -1
          if trueInputFiles:
            if isinstance(newFields, str):
              fieldnames.append(newFields)
            elif isinstance(newFields, list):
              fieldnames.extend(newFields)

          with open(tmpOutputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
            writer.writeheader()

            for row_num, row1 in enumerate(reader1, start=2):  # start=2 (row 1 is header)
              row2 = next(reader2)

              if random.random() < 0.5:
                row = experimentFunct(row_num, row1, row2, prevExpTpl1, prevExpTpl2, mConf)
              else:
                row = experimentFunct(row_num, row2, row1, prevExpTpl2, prevExpTpl1, mConf)

              writer.writerow(row)
      else:
        with open(inputFileFull, encoding="utf-8", newline="") as fh:
          reader = csv.DictReader(fh)

          fieldnames = reader.fieldnames
          row_num = -1

          with open(tmpOutputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
            writer.writeheader()

            for row_num, row in enumerate(reader, start=2):  # start=2 (row 1 is header)

              row = experimentFunct(row_num, row, row, prevExpTpl1, prevExpTpl2, mConf)

              writer.writerow(row)
    
    except Exception as e:
      print(f"Error processing row {row_num}: {e}")
      os.remove(tmpOutputFileFull) # clean up temp file if error occurs
      return(1) 

    os.rename(tmpOutputFileFull, outputFileFull)
    print(f"Output written to: {outputFileFull}")

