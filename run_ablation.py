import llm_client
from llm_client import single_turn

from config import ModelConfig, get_commercial_models_gen
from data_loader import HaikuEntry, load_haiku
from prompts import prompt_1, prompt_2


if __name__ == "__main__":
    
    mConfigIter = get_commercial_models_gen()
    mConf = next(mConfigIter)
    #mConf = next(mConfigIter)
    #mConf = next(mConfigIter) # free Gemini API - use for testing
    #mConf = next(mConfigIter)
    if mConf.api_key: 
      haikuList = load_haiku()
      haiku = haikuList[0]

      #prompt = prompt_1(haiku.haiku)
      prompt = prompt_2(haiku.haiku, haiku.injection)
      answer = single_turn(mConf, prompt)
      print(f"{answer}")
    else:
      print(f"API KEY is empty for {mConf.name}")