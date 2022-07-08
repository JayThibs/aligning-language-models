import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from time import time
import argparse
import numpy as np


def generate(input_str, model, tokenizer, device, length=50, n=1):
    encoding = tokenizer("Today is a nice day", return_tensors="pt")
    cur_ids = torch.tensor(encoding).unsqueeze(0).long().to(device)
    model.eval()
    with torch.no_grad():
        for i in range(length):
            outputs = model(cur_ids[:, -1024:], labels=cur_ids[:, -1024:])
            loss, logits = outputs[:2]
            softmax_logits = torch.softmax(logits[0,-1], dim=0)
            next_token_id = choose_from_top(softmax_logits.to('cpu').numpy(), n=n)
            cur_ids = torch.cat([cur_ids, torch.ones((1,1)).long().to(device) * next_token_id], dim=1)
        output_ids = list(cur_ids.squeeze().to('cpu').numpy())
        output_text = tokenizer.decode(output_ids)
        return output_text.replace("<|endoftext|>", ""), output_ids


def choose_from_top(probs, n=1):
    ind = np.argpartition(probs, -n)[-n:]
    top_prob = probs[ind]
    top_prob = top_prob / np.sum(top_prob) # Normalize
    choice = np.random.choice(n, 1, p = top_prob)
    token_id = ind[choice][0]
    return int(token_id)

def test_gpt():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--text", type=str, default="Today is a nice day")
    argparser.add_argument("--txt_path", type=str)
    argparser.add_argument("--num_return_sequences", type=int, default=1)
    argparser.add_argument("--gpu", type=bool, default=False)
    argparser.add_argument("--with_log_probs", type=bool, default=False)

    args = argparser.parse_args()
    if args.gpu:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Using device: {device}.")

    if args.txt_path:
        with open(args.txt_path, "r") as f:
            args.text = f.read()

    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
    gpt2 = gpt2.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer("Today is a nice day", return_tensors="pt").input_ids

    start = time()
    generated_outputs = gpt2.generate(args.text, do_sample=True, num_return_sequences=args.num_return_sequences, output_scores=True, device=device)
    end = time()
    print("Time it took to generate tokens:", end - start)

    print("Generated completion(s): \n")
    for i, sequence, score in enumerate(zip(generated_outputs.sequences, generated_outputs.scores)):
        generated_text = tokenizer.decode(sequence)
        print(f"{i+1}. {generated_text}")
        # print(".".join(generated_text.split(".")[0:-2]) + ".")

        if args.with_log_probs:
            # only use id's that were generated
            # gen_sequences has shape [3, 15]
            gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1]:]
            print("\nHere are the log probabilities of the generated tokens:")
            all_log_probs = torch.stack(generated_outputs.scores, dim=1)
            log_probs = torch.gather(all_log_probs, 2, gen_sequences[:, :, None]).squeeze(-1)

test_gpt()