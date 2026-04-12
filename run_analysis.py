import llm_client
from llm_client import single_turn
import csv
import os
import pandas as pd

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry


def run_analysis(fileTpl, expTpl1, expTpl2, experimentTpl, accuracyFunct):

  # Get metric names from a sample call to accuracyFunct
  # Create dummy data for this purpose
  dummy_df = pd.DataFrame({expTpl1: [1, 0], expTpl2: [0, 1]})
  sample_metrics = accuracyFunct(dummy_df, expTpl1, expTpl2)
  metric_names = list(sample_metrics.keys())

  #outputFileName = f"{fileTpl}_{experimentTpl}_{modelName}.csv"
  #tmpOutputFileName = f"tmp_{outputFileName}"
  #tmpOutputFileFull = os.path.join(os.path.dirname(__file__), "data", tmpOutputFileName)

  #outputFileFull = os.path.join(os.path.dirname(__file__), "data", outputFileName)
  #if os.path.isfile(outputFileFull):
  #  inputFileFull = outputFileFull
  #  trueInputFile = False

  df_metrics = {name: pd.DataFrame() for name in metric_names}

  dfr_metrics = {name: pd.DataFrame() for name in metric_names}

  dfc_metrics = {name: pd.DataFrame() for name in metric_names}

  acc_exp1_list = []
  acc_exp2_list = []

  acc_cl = {name: {} for name in metric_names}

  # generator models
  for mConf1 in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

    modelName1 = mConf1.litellm_model_id.replace("/", "_")
    modelAlias1 = mConf1.name

    metrics_cr = {name: {"generator": modelAlias1} for name in metric_names}

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

        for name in metric_names:
          metrics_cr[name][modelAlias2] = metrics[name]


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
    
    for name in metric_names:
      df_metrics[name] = pd.concat([df_metrics[name], pd.DataFrame([metrics_cr[name]])], ignore_index=True)

    if exp1_list and exp2_list:
      #by row
      exp1_row = pd.concat(exp1_list)
      exp2_row = pd.concat(exp2_list)
      dfi_r = pd.concat([exp1_row, exp2_row], axis=1)
      metrics_r = accuracyFunct(dfi_r, expTpl1, expTpl2)

      for name in metric_names:
        dfr_metrics[name] = pd.concat([dfr_metrics[name], pd.DataFrame([{"generator": modelAlias1, name: metrics_r[name]}])], ignore_index=True)

  #by column
  for i, mConf2 in enumerate(get_all_models_gen()): #get_open_models_gen(): #get_commercial_models_gen():

    #modelName2 = mConf2.litellm_model_id.replace("/", "_")
    modelAlias2 = mConf2.name
    dfi_c = pd.concat([acc_exp1_list[i], acc_exp2_list[i]], axis=1)
    metrics_c = accuracyFunct(dfi_c, expTpl1, expTpl2)

    for name in metric_names:
      acc_cl[name][modelAlias2] = metrics_c[name]

  for name in metric_names:
    dfc_metrics[name] = pd.DataFrame([acc_cl[name]])

  print(f"Accuracy metrics for: {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl}")
  for name in metric_names:
    print(f"{name.capitalize()}:")
    print(df_metrics[name])

  print(f"Generator accuracy metrics for: {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl}")
  for name in metric_names:
    print(f"{name.capitalize()}:")
    print(dfr_metrics[name])

  print(f"Discriminator accuracy metrics for: {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl}")
  for name in metric_names:
    print(f"{name.capitalize()}:")
    print(dfc_metrics[name])

  #os.rename(tmpOutputFileFull, outputFileFull)
  #print(f"Output written to: {outputFileFull}")

