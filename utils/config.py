import argparse

def unet_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--iterations', type=int, default=100000)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--n_classes', type=int, default=4)
    parser.add_argument('--n_filters', type=int, default=32)
    parser.add_argument('--n_eval', type=int, default=500)
    parser.add_argument('--print_it', type=int, default=10)
    parser.add_argument('--model_checkpoint', type=str, default=None)
    parser.add_argument('--model_name', type=str, default="model")
    parser.add_argument('--model_type', type=str, default="efficientnet-b5")
    parser.add_argument('--validate', default=1, type=int)
    args = parser.parse_args()
    print(args)
    with open("config.txt", 'w') as file:
        file.write(str(args))
    return args