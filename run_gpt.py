from test_gpt import test_gpt
import argparse

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--text", type=str, default="Today is a nice day")
    argparser.add_argument("--txt_path", type=str)
    argparser.add_argument("--num_return_sequences", type=int, default=1)
    argparser.add_argument("--gpu", type=bool, default=False)
    argparser.add_argument("--with_log_probs", type=bool, default=False)
    argparser.add_argument("--max_length", type=int, default=50)
    argparser.add_argument("--no_outputs", type=bool, default=False)
    argparser.add_argument("--time_test", type=bool, default=False)

    args = argparser.parse_args()

    test_gpt(
        text=args.text,
        txt_path=args.txt_path,
        num_return_sequences=args.num_return_sequences,
        gpu=args.gpu,
        with_log_probs=args.with_log_probs,
        max_length=args.max_length,
        no_outputs=args.no_outputs,
        time_test=args.time_test,
    )
