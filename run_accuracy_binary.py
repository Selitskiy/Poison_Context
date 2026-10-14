#import llm_client
#from llm_client import single_turn
import csv
import os
import re

from run_analysis_binary import run_analysis_binary


def modelAccuracyBinaryFunct(dfi):

  nAll = len(dfi)
  df1 = dfi[dfi["response3Num"] == 1]
  df2 = dfi[dfi["response3Num"] == 2]
  nTP = len(df1[df1["prevExp1"] == "ablation"]) + len(df2[df2["prevExp2"] == "ablation"])
  nFP = nAll - nTP

  accuracy = nTP / nAll

  return {
    'accuracy': accuracy,
  }


if __name__ == "__main__":


  fileTpl = "haiku_translation" #"test_haiku_translation"

  #prevExperimentTpl1 = "ablation"
  #prevExperimentTpl2 = "poison"

  experimentTpl = "binary_discriminant"

  run_analysis_binary(fileTpl, experimentTpl, modelAccuracyBinaryFunct)
