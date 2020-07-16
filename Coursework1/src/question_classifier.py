import argparse
from train import Train
from test import Test

parser = argparse.ArgumentParser(description="Text Mining Course Work 1")
parser.add_argument('phase', type=str, help="Phase of question classification.(train/test)")
parser.add_argument('-config',type=str,help="configuration_file_path")
args = parser.parse_args()

with open(args.config,'r') as f:
    config_dict = dict()
    for line in f.readlines():
        line.replace(" ","")
        if line.count(":") > 0:
            head,_,tail = line.partition(":")
            config_dict[head] = tail.strip("\n")

if args.phase == 'train':
    Train(config_dict).train()
if args.phase == 'test':
    Test(config_dict).test()