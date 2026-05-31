import llm_client
from llm_client import single_turn
import csv
import os

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry


def run_experiment_add(fileTpl, fileTpl2, experimentTpl, newFields, experimentFunct):

  DEBUG = False #True

  # Filter doubles and append output files  
  for mConf in get_all_models_gen(): 
    inputFileName = f"{fileTpl}.csv"
    inputFileFull = os.path.join(os.path.dirname(__file__), "data", inputFileName)
    trueInputFile = True

    inputFileName2 = f"{fileTpl2}.csv"
    inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)
    trueInputFile2 = True

    finNum = 0

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

    outputFileName2 = f"{fileTpl2}_{experimentTpl}_{modelName}.csv"
    tmpOutputFileName2 = f"tmp_{outputFileName2}"
    tmpOutputFileFull2 = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileName2)

    outputFileFull2 = os.path.join(os.path.dirname(__file__), "data", outputFileName2)
    if os.path.isfile(outputFileFull2):
      inputFileFull2 = outputFileFull2
      trueInputFile2 = False

    if not os.path.isfile(inputFileFull):
      print(f"Input file not found: {inputFileFull}")
      return(1)
    
    if not os.path.isfile(inputFileFull2):
      print(f"Input file not found: {inputFileFull2}")
      return(1)

    try:
      with open(inputFileFull, encoding="utf-8", newline="") as fh, open(inputFileFull2, encoding="utf-8", newline="") as fh2:
        reader = csv.DictReader(fh)
        reader2 = csv.DictReader(fh2)

        doubleMap = {"瘤あるが故の親しさ一冬木": 0, "除夜の湯に肌触れあへり生くるべし": 0, "向ふから俳句が来るよ冬日和": 0, "裸木の側にしばらく居てやりぬ":0, "生きねばや鳥とて雪を払ひ立つ":0}

        fieldnames = reader.fieldnames
        row_num = -1

        if trueInputFile:
          if isinstance(newFields, str):
            fieldnames.append(newFields)
          elif isinstance(newFields, list):
            fieldnames.extend(newFields)

        with open(tmpOutputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
          writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
          writer.writeheader()

          # Filter out knwn double entries (based on "haiku" field) and write to output
          for row_num, row in enumerate(reader, start=2):  # start=2 (row 1 is header)
              
              row = experimentFunct(row_num, row, mConf)
              haiku = row["haiku"].strip()
              haiku = "".join(haiku.split())

              if doubleMap.get(haiku) is not None:
                doubleMap[haiku] += 1
                if doubleMap[haiku] < 2:
                  writer.writerow(row)
                  finNum += 1
                else:
                  print(f"Skipping duplicate entry for haiku: {row['haiku']} (row {row_num})")
              else:
                writer.writerow(row)
                finNum += 1

          # Append new entries from second file (fileTpl2) to output
          for row_num, row in enumerate(reader2, start=2):  # start=2 (row 1 is header)
              
              row = experimentFunct(row_num, row, mConf)

              writer.writerow(row)
              finNum += 1

    except Exception as e:
      print(f"Error processing row {row_num}: {e}")
      os.remove(tmpOutputFileFull) # clean up temp file if error occurs
      return(1) 

    print(f"Experiment complete for {mConf.name}. Total rows added: {finNum}")
    os.rename(tmpOutputFileFull, outputFileFull)
    print(f"Output written to: {outputFileFull}")



# Filter doubles and append input file  
  if newFields == "response" and experimentTpl == "ablation": # only do this for ablation experiment, not for other experiments that may use different input files
    inputFileName = f"{fileTpl}.csv"
    inputFileFull = os.path.join(os.path.dirname(__file__), "data", inputFileName)

    inputFileName2 = f"{fileTpl2}.csv"
    inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)

    finNum = 0

    tmpInputFileName = f"tmp_{inputFileName}"
    tmpInputFileFull = os.path.join(os.path.dirname(__file__), "data", tmpInputFileName)
      
    if not os.path.isfile(inputFileFull):
      print(f"Input file not found: {inputFileFull}")
      return(1)

    try:
      with open(inputFileFull, encoding="utf-8", newline="") as fh, open(inputFileFull2, encoding="utf-8", newline="") as fh2:
        reader = csv.DictReader(fh)
        reader2 = csv.DictReader(fh2)

        doubleMap = {"瘤あるが故の親しさ一冬木": 0, "除夜の湯に肌触れあへり生くるべし": 0, "向ふから俳句が来るよ冬日和": 0, "裸木の側にしばらく居てやりぬ":0, "生きねばや鳥とて雪を払ひ立つ":0}

        fieldnames = reader.fieldnames
        row_num = -1

        with open(tmpInputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
          writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
          writer.writeheader()

          # Filter out knwn double entries (based on "haiku" field) and write to output
          for row_num, row in enumerate(reader, start=2):  # start=2 (row 1 is header)

                haiku = row["haiku"].strip()
                haiku = "".join(haiku.split())

                if doubleMap.get(haiku) is not None:
                  doubleMap[haiku] += 1
                  if doubleMap[haiku] < 2:
                    writer.writerow(row)
                    finNum += 1
                  else:
                    print(f"Skipping duplicate entry for haiku: {row['haiku']} (row {row_num})")
                else:
                  writer.writerow(row)
                  finNum += 1

          # Append new entries from second file (fileTpl2) to output
          for row_num, row in enumerate(reader2, start=2):  # start=2 (row 1 is header)

              writer.writerow(row)
              finNum += 1

    except Exception as e:
      print(f"Error processing row {row_num}: {e}")
      os.remove(tmpInputFileFull) # clean up temp file if error occurs
      return(1) 

    print(f"Input file processing complete. Total rows after filtering and appending: {finNum}")
    os.rename(tmpInputFileFull, inputFileFull)
    print(f"Input written to: {inputFileFull}")