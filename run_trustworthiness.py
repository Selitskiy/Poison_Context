#import llm_client
#from llm_client import single_turn
import csv
import os
import re

from run_analysis_single import run_analysis_single



def modelTrustworthinessFunct(dfi):
        
  tr = dfi
  tr_m = tr.mean()
  tr_s = tr.std()

  return {
    'trustworthiness': tuple([tr_m, tr_s])
  }


if __name__ == "__main__":

  fileTpl = "haiku_translation" #"test_haiku_translation"

  prevExperimentTpl = "ablation" #"poison"

  experimentTpl = "discriminant"

  run_analysis_single(fileTpl, prevExperimentTpl,  experimentTpl, modelTrustworthinessFunct)