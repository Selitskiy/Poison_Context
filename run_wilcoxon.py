import llm_client
from llm_client import single_turn
import csv
import os
import re

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from run_analysis_hypot import run_analysis_hypot


def wilcoxon_p_value(left, right, alternative='less'):
  paired = pd.DataFrame({"left": left, "right": right}).dropna()
  if paired.empty:
    return 1.0

  diffs = paired["left"] - paired["right"]
  non_zero_diffs = diffs[~np.isclose(diffs, 0.0)]
  if non_zero_diffs.empty:
    return 1.0

  try:
    _, p_value = wilcoxon(
        paired["left"],
        paired["right"],
        alternative=alternative,
    )
  except ValueError:
    return 1.0

  if np.isnan(p_value):
    return 1.0

  return float(p_value)


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

  acc_p_value = wilcoxon_p_value(accuracyw1, accuracyw2, alternative='less')

  return {
    'acc_p_value': acc_p_value,
  }

if __name__ == "__main__":

  # May be (likely) the same
  fileTpl = "haiku_translation" 
  fileTpl2 = "haiku_translation" #"haiku_translation2l"

  # Don't change
  prevExperimentTpl1 = "ablation"
  prevExperimentTpl2 = "poison"

  # Compares if mean of the second distribution is 'better' to the righ tof the first
  experimentTpl1 = "discriminant_self"
  experimentTpl2 = "discriminant_self_orig"

  # Compares distribution in fileTpl with both prevEperimetTpl* columns and experiemntTpl1 experiemmt, with distribution in fileTpl2 with prevEperimetTpl* columns and experiemntTpl2 experiment

  run_analysis_hypot(fileTpl, fileTpl2, prevExperimentTpl1, prevExperimentTpl2, experimentTpl1, experimentTpl2, modelPairAccuracyFunctWilcoxon)
  #print(f"Experiment complete. Success count: {successCount}, Total count: {totalCount}, Failed count: {failedCount}") 
