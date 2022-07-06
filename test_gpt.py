import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

def test_gpt(text="Today is a nice day"):
    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    input_ids = tokenizer(text, return_tensors="pt").input_ids

    generated_outputs = gpt2.generate(input_ids, do_sample=True, num_return_sequences=1, output_scores=True)

    # only use id's that were generated
    # gen_sequences has shape [3, 15]
    gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]

    print("Generated completion: \n")
    print(text + str(tokenizer.decode(gen_sequences[0])))

test_gpt()