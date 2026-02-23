import llm_client
from llm_client import single_turn
import csv
import os

from config import ModelConfig, get_commercial_models_gen
from data_loader import HaikuEntry


def run_experiment(fileTpl, experimentTpl, newFields, experimentFunct):

  DEBUG = True

  inputFileName = f"{fileTpl}.csv"
  inputFileFull = os.path.join(os.path.dirname(__file__), "data", inputFileName)
  trueInputFile = True
    
  if DEBUG:
    mConfigIter = get_commercial_models_gen()
    mConf = next(mConfigIter)
    #mConf = next(mConfigIter)
    #mConf = next(mConfigIter) # free Gemini API - use for testing
    #mConf = next(mConfigIter)
  # RUN
  #for mConf in get_commercial_models_gen():

    if not mConf.api_key:
      print(f"API KEY is empty for {mConf.name}")
      if not DEBUG:
        pass
        #continue
      else:
        exit(1)

    modelName = mConf.litellm_model_id.replace("/", "_")
    outputFileName = f"{fileTpl}_{experimentTpl}_{modelName}.csv"
    tmpOutputFileName = f"tmp_{outputFileName}"
    tmpOutputFileFull = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileName)

    outputFileFull = os.path.join(os.path.dirname(__file__), "data", outputFileName)
    if os.path.isfile(outputFileFull):
      inputFileFull = outputFileFull
      trueInputFile = False
    
    if not os.path.isfile(inputFileFull):
      print(f"Input file not found: {inputFileFull}")
      return(1)

    try:
      with open(inputFileFull, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)

        fieldnames = reader.fieldnames
        row_num = -1
        if trueInputFile:
          fieldnames.append(newFields)

        with open(tmpOutputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
          writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
          writer.writeheader()

          for row_num, row in enumerate(reader, start=2):  # start=2 (row 1 is header)
              
              row = experimentFunct(row_num, row, mConf)

              writer.writerow(row)

    except Exception as e:
      print(f"Error processing row {row_num}: {e}")
      os.remove(tmpOutputFileFull) # clean up temp file if error occurs
      return(1) 

    os.rename(tmpOutputFileFull, outputFileFull)
    print(f"Output written to: {outputFileFull}")

