import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from time import time
import pandas as pd


def gpt_generate(
    text="Hello, world!",
    txt_path=None,
    num_return_sequences=1,
    gpu=False,
    with_log_probs=False,
    max_length=50,
    no_outputs=False,
    time_test=False,
):

    if gpu:
        device_str = "GPU"
        device = torch.device("cuda")
    else:
        device_str = "CPU"
        device = torch.device("cpu")

    if not time_test:
        print(f"Using device: {device}.")

    if txt_path:
        with open(txt_path, "r") as f:
            text = f.read()

    gpt2 = AutoModelForCausalLM.from_pretrained("gpt2", return_dict_in_generate=True)
    gpt2.to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)
    length = max_length + len(input_ids[0])

    start = time()
    generated_outputs = gpt2.generate(
        input_ids,
        do_sample=True,
        max_length=length,
        num_return_sequences=num_return_sequences,
        output_scores=True,
        device=device,
        pad_token_id=tokenizer.eos_token_id,
    )
    end = time()

    if time_test:
        return end - start

    print("-----------------------------------------------------")
    print(
        f"Generated {num_return_sequences} sequences in {end-start:.2f} seconds with a {device_str}."
    )
    print("-----------------------------------------------------")

    if not no_outputs:
        print("~~~ Generated completion(s): ~~~ \n")
        for i, sequence in enumerate(generated_outputs.sequences):
            if with_log_probs:
                token_list = []
                for token in sequence:
                    token_list.append(tokenizer.decode(token))
            generated_text = tokenizer.decode(sequence)
            print(f"Generation {i+1}. {generated_text}")
            # print(".".join(generated_text.split(".")[0:-2]) + ".")

            if with_log_probs:
                gen_sequences = generated_outputs.sequences[:, input_ids.shape[-1] :]
                # print(gen_sequences)
                # print(gen_sequences[i])
                print("----------------------------------------------------")
                print("Here are the log probabilities of the generated tokens:")
                all_log_probs = torch.stack(generated_outputs.scores, dim=1)
                log_probs = torch.gather(
                    all_log_probs, 2, gen_sequences[:, :, None]
                ).squeeze(-1)[i]
                token_with_log_probs = [
                    token_list[len(input_ids[0]) :],
                    log_probs.cpu().numpy(),
                ]
                df = pd.DataFrame(token_with_log_probs).T
                print(df)
                print("----------------------------------------------------")