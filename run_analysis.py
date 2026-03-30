import llm_client
from llm_client import single_turn
import csv
import os

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry
import pandas as pd


def run_analysis(fileTpl, expTpl1, expTpl2, experimentTpl, experimentFunct):

  #outputFileName = f"{fileTpl}_{experimentTpl}_{modelName}.csv"
  #tmpOutputFileName = f"tmp_{outputFileName}"
  #tmpOutputFileFull = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileName)

  #outputFileFull = os.path.join(os.path.dirname(__file__), "data", outputFileName)
  #if os.path.isfile(outputFileFull):
  #  inputFileFull = outputFileFull
  #  trueInputFile = False

  df_accuracy = pd.DataFrame()
  df_precision = pd.DataFrame()
  df_recall = pd.DataFrame()
  df_f1 = pd.DataFrame()

  # generator models
  for mConf1 in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    modelName1 = mConf1.litellm_model_id.replace("/", "_")
    modelAlias1 = mConf1.name

    acc_cr = {"generator": modelAlias1}
    precision_cr = {"generator": modelAlias1}
    recall_cr = {"generator": modelAlias1}
    f1_cr = {"generator": modelAlias1}

    # discriminator models
    for mConf2 in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

      modelName2 = mConf2.litellm_model_id.replace("/", "_")
      modelAlias2 = mConf2.name

      inputFileName1 = f"{fileTpl}_{expTpl1}_{modelName1}_{experimentTpl}_{modelName2}.csv"
      inputFileFull1 = os.path.join(os.path.dirname(__file__), "data", inputFileName1)

      inputFileName2 = f"{fileTpl}_{expTpl2}_{modelName1}_{experimentTpl}_{modelName2}.csv"
      inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)

    
      if not os.path.isfile(inputFileFull1):
        print(f"Input file 1 not found: {inputFileFull1}")
        return(1)
      if not os.path.isfile(inputFileFull2):
        print(f"Input file 2 not found: {inputFileFull2}")
        return(1)

      try:

        df1 = pd.read_csv(inputFileFull1)
        df1 = df1.rename(columns={"response2Num": expTpl1}) # rename to avoid conflict when merging
        df1 = df1[expTpl1] # get just the column we need 

        df2 = pd.read_csv(inputFileFull2)
        df2 = df2.rename(columns={"response2Num": expTpl2}) # rename to avoid conflict when merging
        df2 = df2[expTpl2] # get just the column we need

        dfi = pd.concat([df1, df2], axis=1)

        tp = dfi[expTpl1].mean()
        fn = 1 - tp
        fp = dfi[expTpl2].mean()
        tn = 1 - fp
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp/(tp+fp) 
        recall = tp/(tp+fn) 
        f1 = 2*precision * recall/(precision + recall)

        acc_cr[modelAlias2] = accuracy
        precision_cr[modelAlias2] = precision
        recall_cr[modelAlias2] = recall
        f1_cr[modelAlias2] = f1

        #with open(tmpOutputFileFull, mode='w', encoding="utf-8", newline='') as out_fh:
          #writer = csv.DictWriter(out_fh, fieldnames=fieldnames)
          #writer.writeheader()

          #for row_num, row in enumerate(reader, start=2):  # start=2 (row 1 is header)
              
          #row = experimentFunct(row_num, row, mConf)

          #writer.writerow(row)

      except Exception as e:
        print(f"Error processing: {e}")
        #os.remove(tmpOutputFileFull) # clean up temp file if error occurs
        return(1) 
    
    df_accuracy = pd.concat([df_accuracy, pd.DataFrame([acc_cr])], ignore_index=True)
    df_precision = pd.concat([df_precision, pd.DataFrame([precision_cr])], ignore_index=True)
    df_recall = pd.concat([df_recall, pd.DataFrame([recall_cr])], ignore_index=True)
    df_f1 = pd.concat([df_f1, pd.DataFrame([f1_cr])], ignore_index=True)

  print(f"Accuracy metrics for: {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl}")
  print("Accuracy:")
  print(df_accuracy)
  print("Precision:")
  print(df_precision)
  print("Recall:")
  print(df_recall)
  print("F1:")
  print(df_f1)

  #os.rename(tmpOutputFileFull, outputFileFull)
  #print(f"Output written to: {outputFileFull}")

