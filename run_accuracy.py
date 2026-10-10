#import llm_client
#from llm_client import single_turn
import csv
import os
import re

from run_analysis import run_analysis


def modelPairAccuracyFunct(dfi, expTpl1, expTpl2):
        
  tp = dfi[expTpl1]#.mean()
  fn = 1 - tp
  fp = dfi[expTpl2]#.mean()
  tn = 1 - fp
  accuracy = (tp + tn) / (tp + tn + fp + fn)
  accuracy_m = accuracy.mean()
  accuracy_s = accuracy.std()

  precision = tp/(tp+fp) 
  precision_m = precision.mean()
  precision_s = precision.std()

  recall = tp/(tp+fn)
  recall_m = recall.mean()
  recall_s = recall.std()

  f1 = 2*precision * recall/(precision + recall)
  f1_m = f1.mean()
  f1_s = f1.std()

  return {
    'accuracy': tuple([accuracy_m, accuracy_s]),
    'precision': tuple([precision_m, precision_s]),
    'recall': tuple([recall_m, recall_s]),
    'f1': tuple([f1_m, f1_s])
  }


if __name__ == "__main__":


  fileTpl = "haiku_translation" #"test_haiku_translation"

  prevExperimentTpl1 = "ablation"
  prevExperimentTpl2 = "poison"
  experimentTpl = "discriminant"

  run_analysis(fileTpl, prevExperimentTpl1, prevExperimentTpl2, experimentTpl, modelPairAccuracyFunct)
