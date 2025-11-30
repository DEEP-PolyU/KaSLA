from finetune.train import run_exp
import torch

def main():
    run_exp()


if __name__ == "__main__":

    print("torch.__version__ {}".format(torch.__version__))
    print("torch.version.cuda {}".format(torch.version.cuda))
    print("torch.cuda.is_available() {}".format(torch.cuda.is_available()))
    print("torch.cuda.device_count() {}".format(torch.cuda.device_count()))

    main()
