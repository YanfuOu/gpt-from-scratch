import argparse

import torch

import bigram


def main():
    parser = argparse.ArgumentParser(description='Generate text from a saved model checkpoint.')
    parser.add_argument('--checkpoint', default='model.pt', help='Path to the saved checkpoint')
    parser.add_argument('--prompt', default='', help='Optional text prompt to start generation')
    parser.add_argument('--max-new-tokens', type=int, default=500, help='Number of tokens to generate')
    args = parser.parse_args()

    model = bigram.load_checkpoint(args.checkpoint)

    if args.prompt:
        context = torch.tensor([bigram.encode(args.prompt)], dtype=torch.long, device=bigram.device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=bigram.device)

    with torch.no_grad():
        output = model.generate(context, max_new_tokens=args.max_new_tokens)

    print(bigram.decode(output[0].tolist()))


if __name__ == '__main__':
    main()
