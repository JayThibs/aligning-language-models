import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from time import time
import argparse
import numpy as np
import pandas as pd


def test_gpt():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--text", type=str, default="Today is a nice day")
    argparser.add_argument("--txt_path", type=str)
    argparser.add_argument("--num_return_sequences", type=int, default=1)
    argparser.add_argument("--gpu", type=bool, default=False)
    argparser.add_argument("--with_log_probs", type=bool, default=False)
    argparser.add_argument("--max_length", type=int, default=50)
    argparser.add_argument("--no_outputs", type=bool, default=False)

    args = argparser.parse_args()
    if args.gpu:
        device_str = "GPU"
        device = torch.device("cuda")
    else:
        device_str = "CPU"
        device = torch.device("cpu")

    print(f"Using device: {device}.")

    if args.txt_path:
        with open(args.txt_path, "r") as f:
            args.text = f.read()


    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
    gpt2.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer(args.text, return_tensors="pt").input_ids.to(device)
    length = args.max_length + len(input_ids[0])

    start = time()
    generated_outputs = gpt2.generate(input_ids, do_sample=True, max_length=length, num_return_sequences=args.num_return_sequences, output_scores=True, device=device)
    end = time()
    print("-----------------------------------------------------")
    print(f"Generated {args.num_return_sequences} sequences in {end-start:.2f} seconds with a {device_str}.")
    print("-----------------------------------------------------")

    if not args.no_outputs:
        print("~~~ Generated completion(s): ~~~ \n")
        for i, sequence in enumerate(generated_outputs.sequences):
            if args.with_log_probs:
                token_list = []
                for token in sequence:
                    token_list.append(tokenizer.decode(token))
            generated_text = tokenizer.decode(sequence)
            print(f"Generation {i+1}. {generated_text}")
            # print(".".join(generated_text.split(".")[0:-2]) + ".")

            if args.with_log_probs:
                gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]
                # print(gen_sequences)
                # print(gen_sequences[i])
                print("----------------------------------------------------")
                print("Here are the log probabilities of the generated tokens:")
                all_log_probs = torch.stack(generated_outputs.scores, dim=1)
                log_probs = torch.gather(all_log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)[i]
                token_with_log_probs = [token_list[len(input_ids[0]):], log_probs.numpy()]
                df = pd.DataFrame(token_with_log_probs).T
                print(df)
                print("----------------------------------------------------")

test_gpt()