#import llm_client
#from llm_client import single_turn
import csv
import os
import re

from scipy.stats import ks_2samp, spearmanr
from run_analysis_hypot_self import run_analysis_hypot_self


def modelPairAccuracyFunctKS(dfi1, dfi2, expTpl1, expTpl2):

  tpw1 = dfi1[expTpl1]
  fnw1 = 1 - tpw1
  fpw1 = dfi1[expTpl2]
  tnw1 = 1 - fpw1
  accuracyw1 = (tpw1 + tnw1) / (tpw1 + tnw1 + fpw1 + fnw1)

  tpw2 = dfi2[expTpl1]
  fnw2 = 1 - tpw2
  fpw2 = dfi2[expTpl2]
  tnw2 = 1 - fpw2
  accuracyw2 = (tpw2 + tnw2) / (tpw2 + tnw2 + fpw2 + fnw2)


  acc_ks_stat, acc_p_value = ks_2samp(accuracyw1, accuracyw2)

  # spearmanr is a paired correlation test and requires both samples to have the
  # same length; dfi1/dfi2 can come from files with different row counts, so
  # truncate to the common length
  n = min(len(accuracyw1), len(accuracyw2))
  acc_spearman_corr, acc_spearman_p_value = spearmanr(accuracyw1[:n], accuracyw2[:n])

  return {
    'acc_ks_stat': acc_ks_stat,
    'acc_p_value': acc_p_value,
    'acc_spearman_corr': acc_spearman_corr,
    'acc_spearman_p_value': acc_spearman_p_value,
  }

if __name__ == "__main__":

  # May be (likely) the same
  fileTpl = "haiku_translation" 

  # Don't change
  prevExperimentTpl1 = "ablation"
  prevExperimentTpl2 = "poison"

  # Compares if mean of the second distribution is 'better' to the righ tof the first
  experimentTpl = "discriminant_self_orig"

  # Compares distribution in fileTpl with both prevEperimetTpl* columns and experiemntTpl1 experiemmt, with distribution in fileTpl2 with prevEperimetTpl* columns and experiemntTpl2 experiment

  run_analysis_hypot_self(fileTpl, prevExperimentTpl1, prevExperimentTpl2, experimentTpl, modelPairAccuracyFunctKS)