# MyOwnAI
Building a GPT-style language model completely from scratch using PyTorch.

i am currently building aiml language model in my local system

we are using hugging face tokenizer later we need to make our own tokenizer

and custom layer norm also we should build it later
and rotary embeddings also gelu

-----

m in C:\Users\adity\.cache\huggingface\hub\models--gpt2. Caching files will still work but in a degraded version that might require more space on your disk. This warning can be disabled by setting the `HF_HUB_DISABLE_SYMLINKS_WARNING` environment variable. For more details, see https://huggingface.co/docs/huggingface_hub/how-to-cache#limitations.
To support symlinks on Windows, you either need to activate Developer Mode or to run Python as an administrator. In order to activate developer mode, see this article: https://docs.microsoft.com/en-us/windows/apps/get-started/enable-your-device-for-development
  warnings.warn(message)


---------


  Update train.py

Instead of

config = ModelConfig()

use

from utils.profiles import development_config

config = development_config()

Later, when you're ready for full training:

from utils.profiles import production_config

config = production_config()