import llm_client
from llm_client import single_turn
import csv
import os

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry


def run_2lsubstitution(fileTpl, fileTpl2, experimentTpl, newFields, experimentFunct):

  DEBUG = False

  for mConf in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    inputFileName = f"{fileTpl}.csv"
    inputFileFull = os.path.join(os.path.dirname(__file__), "data", inputFileName)
    trueInputFile = True

    inputFileName2 = f"{fileTpl2}.csv"
    inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)

    if not mConf.api_key:
      print(f"API KEY is empty for {mConf.name}")
      if not DEBUG:
        pass
        #continue
      else:
        exit(1)

    modelName = mConf.litellm_model_id.replace("/", "_")
    outputFileName = f"{fileTpl}_{experimentTpl}_{modelName}.csv"
    #tmpOutputFileName = f"tmp_{outputFileName}"
    #tmpOutputFileFull = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileName)

    outputFileFull = os.path.join(os.path.dirname(__file__), "data", outputFileName)
    if os.path.isfile(outputFileFull):
      inputFileFull = outputFileFull
      trueInputFile = False

    outputFileName2 = f"{fileTpl2}_{experimentTpl}_{modelName}.csv"
    tmpOutputFileName2 = f"tmp_{outputFileName2}"
    tmpOutputFileFull2 = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileName2)
    outputFileFull2 = os.path.join(os.path.dirname(__file__), "data", outputFileName2)
    
    if not os.path.isfile(inputFileFull):
      print(f"Input file not found: {inputFileFull}")
      return(1)

    try:
      with open(inputFileFull, encoding="utf-8", newline="") as fh, open(inputFileFull2, encoding="utf-8", newline="") as fh2:
        reader = csv.DictReader(fh)
        reader2 = csv.DictReader(fh2)

        fieldnames = reader.fieldnames
        row_num = -1
        if trueInputFile:
          if isinstance(newFields, str):
            fieldnames.append(newFields)
          elif isinstance(newFields, list):
            fieldnames.extend(newFields)

        with open(tmpOutputFileFull2, mode='w', encoding="utf-8", newline='') as out_fh:
          writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
          writer.writeheader()

          for row_num, (row, row2) in enumerate(zip(reader, reader2), start=2):  # start=2 (row 1 is header)

            row = experimentFunct(row_num, row, row2, mConf)

            writer.writerow(row)

    except Exception as e:
      print(f"Error processing row {row_num}: {e}")
      os.remove(tmpOutputFileFull2) # clean up temp file if error occurs
      return(1) 

    os.rename(tmpOutputFileFull2, outputFileFull2)
    print(f"Output written to: {outputFileFull2}")

