import llm_client
from llm_client import single_turn
import csv
import os
import random

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry


def run_binary_experiment_add(fileTpl1, fileTpl1a, fileTpl2, fileTpl2a, genFileTpl, genFileTpl2, genModelName, prevExpTpl1, prevExpTpl2, experimentTpl, newFields, experimentFunct):

  DEBUG = False #True
    
  for mConf in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    inputFileName1 = f"{fileTpl1}.csv"
    inputFileFull1 = os.path.join(os.path.dirname(__file__), "data", inputFileName1)
    inputFileName2 = f"{fileTpl2}.csv"
    inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)
    trueInputFiles = True

    inputFileName1a = f"{fileTpl1a}.csv"
    inputFileFull1a = os.path.join(os.path.dirname(__file__), "data", inputFileName1a)
    inputFileName2a = f"{fileTpl2a}.csv"
    inputFileFull2a = os.path.join(os.path.dirname(__file__), "data", inputFileName2a)
    trueInputFilesA = True

    finNum = 0

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

    outputFileNameA = f"{genFileTpl2}_{genModelName}_{experimentTpl}_{modelName}.csv"
    tmpOutputFileNameA = f"tmp_{outputFileNameA}"
    tmpOutputFileFullA = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileNameA)

    outputFileFullA = os.path.join(os.path.dirname(__file__), "data", outputFileNameA)
    if os.path.isfile(outputFileFullA):
      inputFileFullA = outputFileFullA
      trueInputFilesA = False
    
    if not os.path.isfile(inputFileFull1):
      print(f"Input file not found: {inputFileFull1}")
      return(1)
    if not os.path.isfile(inputFileFull2):
      print(f"Input file not found: {inputFileFull2}")
      return(1)
    
    if not os.path.isfile(inputFileFull1a):
      print(f"Input file not found: {inputFileFull1a}")
      return(1)
    if not os.path.isfile(inputFileFull2a):
      print(f"Input file not found: {inputFileFull2a}")
      return(1)


    try:

      #if trueInputFiles and trueInputFilesA:
      #  with open(inputFileFull1, encoding="utf-8", newline="") as fh1, open(inputFileFull2, encoding="utf-8", newline="") as fh2, open(inputFileFull1a, encoding="utf-8", newline="") as fh1a, open(inputFileFull2a, encoding="utf-8", newline="") as fh2a:
      #    reader1 = csv.DictReader(fh1)
      #    reader2 = csv.DictReader(fh2)
      #    reader1a = csv.DictReader(fh1a)
      #    reader2a = csv.DictReader(fh2a)

      #    fieldnames = reader1.fieldnames
      #    row_num = -1
      #    if trueInputFiles:
      #      if isinstance(newFields, str):
      #        fieldnames.append(newFields)
      #      elif isinstance(newFields, list):
      #        fieldnames.extend(newFields)

      #    with open(tmpOutputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
      #      writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
      #      writer.writeheader()

      #      for row_num, row1 in enumerate(reader1, start=2):  # start=2 (row 1 is header)
      #        row2 = next(reader2)

      #        if random.random() < 0.5:
      #          row = experimentFunct(row_num, row1, row2, prevExpTpl1, prevExpTpl2, mConf)
      #        else:
      #          row = experimentFunct(row_num, row2, row1, prevExpTpl2, prevExpTpl1, mConf)

      #        writer.writerow(row)
      #el
      if (not trueInputFiles) and (not trueInputFilesA):
        with open(inputFileFull, encoding="utf-8", newline="") as fh, open(inputFileFullA, encoding="utf-8", newline="") as fhA:
          reader = csv.DictReader(fh)
          readerA = csv.DictReader(fhA)

          doubleMap = {"瘤あるが故の親しさ一冬木": 0, "除夜の湯に肌触れあへり生くるべし": 0, "向ふから俳句が来るよ冬日和": 0, "裸木の側にしばらく居てやりぬ":0, "生きねばや鳥とて雪を払ひ立つ":0, "五月来ぬ心ひらけし五月来ぬ":0, "五十聟天窓をかくす扇かな":0, "梅遠近南すべく北すべく":0, "梅遠近そぞろあるきす昨日今日":0, "雨の中に立春大吉の光あり":0}

          fieldnames = reader.fieldnames
          row_num = -1

          with open(tmpOutputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
            writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
            writer.writeheader()

            for row_num, row in enumerate(reader, start=2):  # start=2 (row 1 is header)

              row = experimentFunct(row_num, row, row, prevExpTpl1, prevExpTpl2, mConf)
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


            for row_num, row in enumerate(readerA, start=2):  # start=2 (row 1 is header)

              row = experimentFunct(row_num, row, row, prevExpTpl1, prevExpTpl2, mConf)
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


      else:
        print(f"Mixed trueInputFiles and trueInputFilesA: {trueInputFiles}, {trueInputFilesA}")
        return(1)
    
    except Exception as e:
      print(f"Error processing row {row_num}: {e}")
      os.remove(tmpOutputFileFull) # clean up temp file if error occurs
      return(1) 

    os.rename(tmpOutputFileFull, outputFileFull)
    print(f"Output written to: {outputFileFull}")

