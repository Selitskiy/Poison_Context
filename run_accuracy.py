import llm_client
from llm_client import single_turn
import csv
import os
import re


from run_analysis import run_analysis


def modelPairAccuracyFunct(dfi, expTpl1, expTpl2):
        
  tp = dfi[expTpl1].mean()
  fn = 1 - tp
  fp = dfi[expTpl2].mean()
  tn = 1 - fp
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  precision = tp/(tp+fp) 
  recall = tp/(tp+fn) 
  f1 = 2*precision * recall/(precision + recall)

  return(accuracy, precision, recall, f1)


if __name__ == "__main__":
    
  #for mConf in get_all_models_gen(): #get_open_models_gen(): #get_commercial_models_gen():

  #modelName = mConf.litellm_model_id.replace("/", "_")

  fileTpl = "haiku_translation" #"test_haiku_translation"

  prevExperimentTpl1 = "ablation"
  prevExperimentTpl2 = "poison"
  experimentTpl = "discriminant"

  run_analysis(fileTpl, prevExperimentTpl1, prevExperimentTpl2, experimentTpl, modelPairAccuracyFunct)
  #print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 