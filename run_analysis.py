import llm_client
from llm_client import single_turn
import csv
import os
import pandas as pd

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry
#from run_accuracy import modelPairAccuracyFunct


def run_analysis(fileTpl, expTpl1, expTpl2, experimentTpl, accuracyFunct):

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

  dfr_accuracy = pd.DataFrame()
  dfr_precision = pd.DataFrame()
  dfr_recall = pd.DataFrame()
  dfr_f1 = pd.DataFrame()

  dfc_accuracy = pd.DataFrame()
  dfc_precision = pd.DataFrame()
  dfc_recall = pd.DataFrame()
  dfc_f1 = pd.DataFrame()

  acc_exp1_list = []
  acc_exp2_list = []

  acc_cl = {}
  precision_cl = {}
  recall_cl = {}
  f1_cl = {}

  # generator models
  for mConf1 in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    modelName1 = mConf1.litellm_model_id.replace("/", "_")
    modelAlias1 = mConf1.name

    acc_cr = {"generator": modelAlias1}
    precision_cr = {"generator": modelAlias1}
    recall_cr = {"generator": modelAlias1}
    f1_cr = {"generator": modelAlias1}

    exp1_list = []
    exp2_list = []

    # discriminator models
    for i, mConf2 in enumerate(get_all_models_gen()): #get_open_models_gen(): #get_commercial_models_gen():

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

        metrics = accuracyFunct(dfi, expTpl1, expTpl2)
        accuracy = metrics['accuracy']
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']

        acc_cr[modelAlias2] = accuracy
        precision_cr[modelAlias2] = precision
        recall_cr[modelAlias2] = recall
        f1_cr[modelAlias2] = f1


        # by row
        exp1_list.append(df1)
        exp2_list.append(df2)

        # by column
        if len(acc_exp1_list) <= i:
          acc_exp1_list.append(exp1_list[i])
        else:
          acc_exp1_list[i] = pd.concat([acc_exp1_list[i], exp1_list[i]])

        if len(acc_exp2_list) <= i:
          acc_exp2_list.append(exp2_list[i])
        else:
          acc_exp2_list[i] = pd.concat([acc_exp2_list[i], exp2_list[i]])

        #tp = dfi[expTpl1].mean()
        #fn = 1 - tp
        #fp = dfi[expTpl2].mean()
        #tn = 1 - fp
        #accuracy = (tp + tn) / (tp + tn + fp + fn)
        #precision = tp/(tp+fp) 
        #recall = tp/(tp+fn) 
        #f1 = 2*precision * recall/(precision + recall)



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

    if exp1_list and exp2_list:
      #by row
      exp1_row = pd.concat(exp1_list)
      exp2_row = pd.concat(exp2_list)
      dfi_r = pd.concat([exp1_row, exp2_row], axis=1)
      metrics_r = accuracyFunct(dfi_r, expTpl1, expTpl2)
      accuracy_r = metrics_r['accuracy']
      precision_r = metrics_r['precision']
      recall_r = metrics_r['recall']
      f1_r = metrics_r['f1']

      dfr_accuracy = pd.concat([dfr_accuracy, pd.DataFrame([{"generator": modelAlias1, "accuracy": accuracy_r}])], ignore_index=True)
      dfr_precision = pd.concat([dfr_precision, pd.DataFrame([{"generator": modelAlias1, "precision": precision_r}])], ignore_index=True)
      dfr_recall = pd.concat([dfr_recall, pd.DataFrame([{"generator": modelAlias1, "recall": recall_r}])], ignore_index=True)
      dfr_f1 = pd.concat([dfr_f1, pd.DataFrame([{"generator": modelAlias1, "f1": f1_r}])], ignore_index=True)

  #by column
  for i, mConf2 in enumerate(get_all_models_gen()): #get_open_models_gen(): #get_commercial_models_gen():

    #modelName2 = mConf2.litellm_model_id.replace("/", "_")
    modelAlias2 = mConf2.name
    dfi_c = pd.concat([acc_exp1_list[i], acc_exp2_list[i]], axis=1)
    metrics_c = accuracyFunct(dfi_c, expTpl1, expTpl2)
    accuracy_c = metrics_c['accuracy']
    precision_c = metrics_c['precision']
    recall_c = metrics_c['recall']
    f1_c = metrics_c['f1']

    acc_cl[modelAlias2] = accuracy_c
    precision_cl[modelAlias2] = precision_c
    recall_cl[modelAlias2] = recall_c
    f1_cl[modelAlias2] = f1_c

  dfc_accuracy = pd.DataFrame([acc_cl])
  dfc_precision = pd.DataFrame([precision_cl])
  dfc_recall = pd.DataFrame([recall_cl])
  dfc_f1 = pd.DataFrame([f1_cl])

  print(f"Accuracy metrics for: {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl}")
  print("Accuracy:")
  print(df_accuracy)
  print("Precision:")
  print(df_precision)
  print("Recall:")
  print(df_recall)
  print("F1:")
  print(df_f1)

  print(f"Generator accuracy metrics for: {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl}")
  print("Accuracy:")
  print(dfr_accuracy)
  print("Precision:")
  print(dfr_precision)
  print("Recall:")
  print(dfr_recall)
  print("F1:")
  print(dfr_f1)

  print(f"Discriminator accuracy metrics for: {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl}")
  print("Accuracy:")
  print(dfc_accuracy)
  print("Precision:")
  print(dfc_precision)
  print("Recall:")
  print(dfc_recall)
  print("F1:")
  print(dfc_f1)

  #os.rename(tmpOutputFileFull, outputFileFull)
  #print(f"Output written to: {outputFileFull}")

