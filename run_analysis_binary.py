#import llm_client
#from llm_client import single_turn
import csv
import os
import pandas as pd

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry
from latex_output import print_latex_table

def run_analysis_binary(fileTpl, experimentTpl, accuracyFunct):

  # Get metric names from a sample call to accuracyFunct
  # Create dummy data for this purpose
  dummy_df = pd.DataFrame({"response3Num":[1, 2], "prevExp1": ["ablation", "poison"], "prevExp2": ["poison", "ablation"]})
  sample_metrics = accuracyFunct(dummy_df)
  metric_names = list(sample_metrics.keys())


  df_metrics = {name: pd.DataFrame() for name in metric_names}

  dfr_metrics = {name: pd.DataFrame() for name in metric_names}

  dfc_metrics = {name: pd.DataFrame() for name in metric_names}

  acc_exp1_list = []
  #acc_exp2_list = []

  acc_cl = {name: {} for name in metric_names}

  # generator models
  for mConf1 in get_all_models_gen(keysFl=False): #get_open_models_gen(): #get_commercial_models_gen():

    modelName1 = mConf1.litellm_model_id.replace("/", "_")
    modelAlias1 = mConf1.name

    metrics_cr = {name: {"generator": modelAlias1} for name in metric_names}

    exp1_list = []
    #exp2_list = []

    # discriminator models
    for i, mConf2 in enumerate(get_all_models_gen(keysFl=False)): #get_open_models_gen(): #get_commercial_models_gen():

      modelName2 = mConf2.litellm_model_id.replace("/", "_")
      modelAlias2 = mConf2.name

      inputFileName1 = f"{fileTpl}_{modelName1}_{experimentTpl}_{modelName2}.csv"
      inputFileFull1 = os.path.join(os.path.dirname(__file__), "data", inputFileName1)

    
      if not os.path.isfile(inputFileFull1):
        print(f"Input file 1 not found: {inputFileFull1}")
        return(1)


      try:

        df1 = pd.read_csv(inputFileFull1)
        dfi = df1[["response3Num", "prevExp1", "prevExp2"]] # get just the column we need 

        metrics = accuracyFunct(dfi)

        for name in metric_names:
          metrics_cr[name][modelAlias2] = metrics[name]


        # by row
        exp1_list.append(dfi)

        # by column
        if len(acc_exp1_list) <= i:
          acc_exp1_list.append(exp1_list[i])
        else:
          acc_exp1_list[i] = pd.concat([acc_exp1_list[i], exp1_list[i]])



      except Exception as e:
        print(f"Error processing: {e}")
        return(1) 
    
    for name in metric_names:
      df_metrics[name] = pd.concat([df_metrics[name], pd.DataFrame([metrics_cr[name]])], ignore_index=True)

    if exp1_list:
      #by row
      exp1_row = pd.concat(exp1_list)
      #exp2_row = pd.concat(exp2_list)
      #dfi_r = pd.concat([exp1_row, exp2_row], axis=1)
      metrics_r = accuracyFunct(exp1_row)

      for name in metric_names:
        dfr_metrics[name] = pd.concat([dfr_metrics[name], pd.DataFrame([{"generator": modelAlias1, name: metrics_r[name]}])], ignore_index=True)

  #by column
  for i, mConf2 in enumerate(get_all_models_gen(keysFl=False)): #get_open_models_gen(): #get_commercial_models_gen():

    modelAlias2 = mConf2.name
    dfi_c = acc_exp1_list[i]
    metrics_c = accuracyFunct(dfi_c)

    for name in metric_names:
      acc_cl[name][modelAlias2] = metrics_c[name]

  for name in metric_names:
    dfc_metrics[name] = pd.DataFrame([acc_cl[name]])


  for name in metric_names:
    print_latex_table(
        df_metrics[name],
        f"Accuracy metrics for: {fileTpl}, {experimentTpl} ({name})",
        f"tab:bin_gd_{fileTpl}_{experimentTpl}_{name}",
    )
    #print(df_metrics[name])

  for name in metric_names:
    print_latex_table(
        dfr_metrics[name],
        f"Generator accuracy metrics for: {fileTpl}, {experimentTpl} ({name})",
        f"tab:bin_g_{fileTpl}_{experimentTpl}_{name}",
    )
    #print(dfr_metrics[name])

  for name in metric_names:
    print_latex_table(
        dfc_metrics[name],
        f"Discriminator accuracy metrics for: {fileTpl}, {experimentTpl} ({name})",
        f"tab:g_{fileTpl}_{experimentTpl}_{name}",
    )
    #print(dfc_metrics[name])

