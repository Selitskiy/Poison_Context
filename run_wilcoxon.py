import llm_client
from llm_client import single_turn
import csv
import os
import re

from scipy.stats import wilcoxon
from run_analysis_hypot import run_analysis_hypot


def modelPairAccuracyFunctWilcoxon(dfi1, dfi2, expTpl1, expTpl2):

  tpw1 = dfi1[expTpl1]
  fnw1 = 1 - tpw1
  fpw1 = dfi1[expTpl2]
  tnw1 = 1 - fpw1
  accuracyw1 = (tpw1 + tnw1) / (tpw1 + tnw1 + fpw1 + fnw1)
  precisionw1 = tpw1/(tpw1+fpw1)
  recallw1 = tpw1/(tpw1+fnw1)
  f1w1 = 2*precisionw1 * recallw1/(precisionw1 + recallw1)

  tpw2 = dfi2[expTpl1]
  fnw2 = 1 - tpw2
  fpw2 = dfi2[expTpl2]
  tnw2 = 1 - fpw2
  accuracyw2 = (tpw2 + tnw2) / (tpw2 + tnw2 + fpw2 + fnw2)
  precisionw2 = tpw2/(tpw2+fpw2)
  recallw2 = tpw2/(tpw2+fnw2)
  f1w2 = 2*precisionw2 * recallw2/(precisionw2 + recallw2)

  acc_wilcoxon_stat, acc_p_value = wilcoxon(accuracyw1, accuracyw2, alternative='less')
  #f1_wilcoxon_stat, f1_p_value = wilcoxon(f1w1, f1w2, alternative='less')

  return {
    'acc_p_value': acc_p_value,
    #'f1_p_value': f1_p_value
  }

if __name__ == "__main__":

  fileTpl = "haiku_translation" #"test_haiku_translation"
  fileTpl2 = "haiku_translation"

  prevExperimentTpl1 = "ablation"
  prevExperimentTpl2 = "poison"
  experimentTpl1 = "discriminant"
  experimentTpl2 = "discriminant_diy"

  run_analysis_hypot(fileTpl, fileTpl2, prevExperimentTpl1, prevExperimentTpl2, experimentTpl1, experimentTpl2, modelPairAccuracyFunctWilcoxon)
  #print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 