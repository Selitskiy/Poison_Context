#import llm_client
#from llm_client import single_turn
import csv
import os
import pandas as pd

from config import ModelConfig, get_commercial_models_gen, get_open_models_gen, get_all_models_gen
from data_loader import HaikuEntry
from latex_output import print_latex_table


def run_analysis_hypot_self(fileTpl, expTpl1, expTpl2, experimentTpl, accuracyFunct):

  # Get metric names from a sample call to accuracyFunct
  # Create dummy data for this purpose
  dummy_df = pd.DataFrame({expTpl1: [1, 0], expTpl2: [0, 1]})
  sample_metrics = accuracyFunct(dummy_df, dummy_df, expTpl1, expTpl2)
  metric_names = list(sample_metrics.keys())


  df_metrics = {name: pd.DataFrame() for name in metric_names}

  dfr_metrics = {name: pd.DataFrame() for name in metric_names}

  dfc_metrics = {name: pd.DataFrame() for name in metric_names}

  acc_exp1_list = []
  acc_exp2_list = []
  acc_exp3_list = []
  acc_exp4_list = []

  acc_cl = {name: {} for name in metric_names}


  # Horizontal pairwise comparison

  # generator models
  for mConf1 in get_all_models_gen(keysFl=False): #get_open_models_gen(): #get_commercial_models_gen():

    modelName1 = mConf1.litellm_model_id.replace("/", "_")
    modelAlias1 = mConf1.name

    metrics_cr = {name: {"generator": modelAlias1} for name in metric_names}

    # discriminator models
    for i, mConf2 in enumerate(get_all_models_gen(keysFl=False)): #get_open_models_gen(): #get_commercial_models_gen():

      modelName2 = mConf2.litellm_model_id.replace("/", "_")
      modelAlias2 = mConf2.name

      inputFileName1 = f"{fileTpl}_{expTpl1}_{modelName1}_{experimentTpl}_{modelName2}.csv"
      inputFileFull1 = os.path.join(os.path.dirname(__file__), "data", inputFileName1)

      inputFileName2 = f"{fileTpl}_{expTpl2}_{modelName1}_{experimentTpl}_{modelName2}.csv"
      inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)

      # For inputFileFull3/4, use the next discriminator model in the loop;
      # wrap around to the first (i=0) model for the last element
      discriminator_models = list(get_all_models_gen(keysFl=False))
      mConf3 = discriminator_models[(i + 1) % len(discriminator_models)]
      modelName3 = mConf3.litellm_model_id.replace("/", "_")

      inputFileName3 = f"{fileTpl}_{expTpl1}_{modelName1}_{experimentTpl}_{modelName3}.csv"
      inputFileFull3 = os.path.join(os.path.dirname(__file__), "data", inputFileName3)

      inputFileName4 = f"{fileTpl}_{expTpl2}_{modelName1}_{experimentTpl}_{modelName3}.csv"
      inputFileFull4 = os.path.join(os.path.dirname(__file__), "data", inputFileName4)

    
      if not os.path.isfile(inputFileFull1):
        print(f"Input file 1 not found: {inputFileFull1}")
        return(1)
      if not os.path.isfile(inputFileFull2):
        print(f"Input file 2 not found: {inputFileFull2}")
        return(1)
      if not os.path.isfile(inputFileFull3):
        print(f"Input file 3 not found: {inputFileFull3}")
        return(1)
      if not os.path.isfile(inputFileFull4):
        print(f"Input file 4 not found: {inputFileFull4}")
        return(1)

      try:

        df1 = pd.read_csv(inputFileFull1)
        df1 = df1.rename(columns={"response2Num": expTpl1}) # rename to avoid conflict when merging
        df1 = df1[expTpl1] # get just the column we need 

        df2 = pd.read_csv(inputFileFull2)
        df2 = df2.rename(columns={"response2Num": expTpl2}) # rename to avoid conflict when merging
        df2 = df2[expTpl2] # get just the column we need

        df3 = pd.read_csv(inputFileFull3)
        df3 = df3.rename(columns={"response2Num": expTpl1}) # rename to avoid conflict when merging
        df3 = df3[expTpl1] # get just the column we need

        df4 = pd.read_csv(inputFileFull4)
        df4 = df4.rename(columns={"response2Num": expTpl2}) # rename to avoid conflict when merging
        df4 = df4[expTpl2] # get just the column we need

        # Concat rows (drop rows without a matching pair to avoid NaN propagating into the test)
        dfi1 = pd.concat([df1, df2], axis=1).dropna()
        dfi2 = pd.concat([df3, df4], axis=1).dropna()

        metrics = accuracyFunct(dfi1, dfi2, expTpl1, expTpl2)

        for name in metric_names:
          metrics_cr[name][modelAlias2] = metrics[name]

        # by column
        if len(acc_exp1_list) <= i:
          acc_exp1_list.append(df1)
        else:
          acc_exp1_list[i] = pd.concat([acc_exp1_list[i], df1], ignore_index=True)

        if len(acc_exp2_list) <= i:
          acc_exp2_list.append(df2)
        else:
          acc_exp2_list[i] = pd.concat([acc_exp2_list[i], df2], ignore_index=True)

        if len(acc_exp3_list) <= i:
          acc_exp3_list.append(df3)
        else:
          acc_exp3_list[i] = pd.concat([acc_exp3_list[i], df3], ignore_index=True)

        if len(acc_exp4_list) <= i:
          acc_exp4_list.append(df4)
        else:
          acc_exp4_list[i] = pd.concat([acc_exp4_list[i], df4], ignore_index=True)


      except Exception as e:
        print(f"Error processing: {e}")
        #os.remove(tmpOutputFileFull) # clean up temp file if error occurs
        return(1) 
    
    for name in metric_names:
      df_metrics[name] = pd.concat([df_metrics[name], pd.DataFrame([metrics_cr[name]])], ignore_index=True)

  #by column
  for i, mConf2 in enumerate(get_all_models_gen(keysFl=False)): #get_open_models_gen(): #get_commercial_models_gen():

    #modelName2 = mConf2.litellm_model_id.replace("/", "_")
    modelAlias2 = mConf2.name
    dfi_c1 = pd.concat([acc_exp1_list[i], acc_exp2_list[i]], axis=1).dropna()
    dfi_c2 = pd.concat([acc_exp3_list[i], acc_exp4_list[i]], axis=1).dropna()

    metrics_c = accuracyFunct(dfi_c1, dfi_c2, expTpl1, expTpl2)

    for name in metric_names:
      acc_cl[name][modelAlias2] = metrics_c[name]

  for name in metric_names:
    dfc_metrics[name] = pd.DataFrame([acc_cl[name]])


  for name in metric_names:
    print_latex_table(
        df_metrics[name],
        f"P-valuers for Kolmogorov-Smirnov left-right model comparison {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl} ({name})",
        f"tab:m_ud_{fileTpl}_{expTpl1}_{expTpl2}_{experimentTpl}_{name}", 5
    )
    #print(df_metrics[name])

  for name in metric_names:
    print_latex_table(
        dfc_metrics[name],
        f"P-valuers for Kolmogorov-Smirnov left-right discriminator comparison {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl} ({name})",
        f"tab:d_ud_{fileTpl}_{expTpl1}_{expTpl2}_{experimentTpl}_{name}", 5
    )
    #print(dfc_metrics[name])




  df_metrics = {name: pd.DataFrame() for name in metric_names}

  dfr_metrics = {name: pd.DataFrame() for name in metric_names}

  dfc_metrics = {name: pd.DataFrame() for name in metric_names}

  acc_exp1_list = []
  acc_exp2_list = []
  acc_exp3_list = []
  acc_exp4_list = []

  acc_cl = {name: {} for name in metric_names}

  # Vertical pairwise comparison

  # generator models
  for j, mConf1 in enumerate(get_all_models_gen(keysFl=False)): #get_open_models_gen(): #get_commercial_models_gen():

    modelName1 = mConf1.litellm_model_id.replace("/", "_")
    modelAlias1 = mConf1.name

    metrics_cr = {name: {"generator": modelAlias1} for name in metric_names}

    exp1_list = []
    exp2_list = []
    exp3_list = []
    exp4_list = []

    # discriminator models
    for i, mConf2 in enumerate(get_all_models_gen(keysFl=False)): #get_open_models_gen(): #get_commercial_models_gen():

      modelName2 = mConf2.litellm_model_id.replace("/", "_")
      modelAlias2 = mConf2.name

      inputFileName1 = f"{fileTpl}_{expTpl1}_{modelName1}_{experimentTpl}_{modelName2}.csv"
      inputFileFull1 = os.path.join(os.path.dirname(__file__), "data", inputFileName1)

      inputFileName2 = f"{fileTpl}_{expTpl2}_{modelName1}_{experimentTpl}_{modelName2}.csv"
      inputFileFull2 = os.path.join(os.path.dirname(__file__), "data", inputFileName2)

      # For inputFileFull3/4, use the next generator model in the loop;
      # wrap around to the first (i=0) model for the last element
      generator_models = list(get_all_models_gen(keysFl=False))
      mConf3 = generator_models[(j + 1) % len(generator_models)]
      modelName3 = mConf3.litellm_model_id.replace("/", "_")

      inputFileName3 = f"{fileTpl}_{expTpl1}_{modelName3}_{experimentTpl}_{modelName2}.csv"
      inputFileFull3 = os.path.join(os.path.dirname(__file__), "data", inputFileName3)

      inputFileName4 = f"{fileTpl}_{expTpl2}_{modelName3}_{experimentTpl}_{modelName2}.csv"
      inputFileFull4 = os.path.join(os.path.dirname(__file__), "data", inputFileName4)

    
      if not os.path.isfile(inputFileFull1):
        print(f"Input file 1 not found: {inputFileFull1}")
        return(1)
      if not os.path.isfile(inputFileFull2):
        print(f"Input file 2 not found: {inputFileFull2}")
        return(1)
      if not os.path.isfile(inputFileFull3):
        print(f"Input file 3 not found: {inputFileFull3}")
        return(1)
      if not os.path.isfile(inputFileFull4):
        print(f"Input file 4 not found: {inputFileFull4}")
        return(1)

      try:

        df1 = pd.read_csv(inputFileFull1)
        df1 = df1.rename(columns={"response2Num": expTpl1}) # rename to avoid conflict when merging
        df1 = df1[expTpl1] # get just the column we need 

        df2 = pd.read_csv(inputFileFull2)
        df2 = df2.rename(columns={"response2Num": expTpl2}) # rename to avoid conflict when merging
        df2 = df2[expTpl2] # get just the column we need

        df3 = pd.read_csv(inputFileFull3)
        df3 = df3.rename(columns={"response2Num": expTpl1}) # rename to avoid conflict when merging
        df3 = df3[expTpl1] # get just the column we need

        df4 = pd.read_csv(inputFileFull4)
        df4 = df4.rename(columns={"response2Num": expTpl2}) # rename to avoid conflict when merging
        df4 = df4[expTpl2] # get just the column we need

        # Concat rows (drop rows without a matching pair to avoid NaN propagating into the test)
        dfi1 = pd.concat([df1, df2], axis=1).dropna()
        dfi2 = pd.concat([df3, df4], axis=1).dropna()

        metrics = accuracyFunct(dfi1, dfi2, expTpl1, expTpl2)

        for name in metric_names:
          metrics_cr[name][modelAlias2] = metrics[name]


        # by row
        exp1_list.append(df1)
        exp2_list.append(df2)
        exp3_list.append(df3)
        exp4_list.append(df4)


      except Exception as e:
        print(f"Error processing: {e}")
        #os.remove(tmpOutputFileFull) # clean up temp file if error occurs
        return(1) 
    
    for name in metric_names:
      df_metrics[name] = pd.concat([df_metrics[name], pd.DataFrame([metrics_cr[name]])], ignore_index=True)

    if exp1_list and exp2_list:
      #by row
      exp1_row = pd.concat(exp1_list, ignore_index=True)
      exp2_row = pd.concat(exp2_list, ignore_index=True)
      dfi_r1 = pd.concat([exp1_row, exp2_row], axis=1).dropna()

      exp3_row = pd.concat(exp3_list, ignore_index=True)
      exp4_row = pd.concat(exp4_list, ignore_index=True)
      dfi_r2 = pd.concat([exp3_row, exp4_row], axis=1).dropna()

      metrics_r = accuracyFunct(dfi_r1, dfi_r2, expTpl1, expTpl2)

      for name in metric_names:
        dfr_metrics[name] = pd.concat([dfr_metrics[name], pd.DataFrame([{"generator": modelAlias1, name: metrics_r[name]}])], ignore_index=True)


  for name in metric_names:
    print_latex_table(
        df_metrics[name],
        f"P-valuers for Kolmogorov-Smirnov up-down model comparison {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl} ({name})",
        f"tab:m_ud_{fileTpl}_{expTpl1}_{expTpl2}_{experimentTpl}_{name}", 5
    )
    #print(df_metrics[name])

  for name in metric_names:
    print_latex_table(
        dfr_metrics[name],
        f"P-valuers for Kolmogorov-Smirnov up-down generator comparison {fileTpl}, {expTpl1}, {expTpl2}, {experimentTpl} ({name})",
        f"tab:g_ud_{fileTpl}_{expTpl1}_{expTpl2}_{experimentTpl}_{name}", 5
    )
    #print(dfr_metrics[name])


  #os.rename(tmpOutputFileFull, outputFileFull)
  #print(f"Output written to: {outputFileFull}")

