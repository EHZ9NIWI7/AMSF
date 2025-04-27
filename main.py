import argparse
import os
from re import S

import yaml

from gen_split import Split_Generator
from gen_text_feat import Text_Generator
from GZSSAR.processor import Processor


def init_parser():
    parser = argparse.ArgumentParser(description='')
    
    parser.add_argument('--config', '-c', type=str, default='')
    
    parser.add_argument('--data_dir', '-dd', type=str, default='./data')
    parser.add_argument('--log_dir', '-ld', type=str, default='./log')
    parser.add_argument('--result_dir', '-rd', type=str, default='./result')
    parser.add_argument('--gpu', '-g', type=str, default='0')
    
    parser.add_argument('--skeleton_model', '-sm', type=str, default='shift')
    parser.add_argument('--language_model', '-lm', type=str, default='longclip-L')
    parser.add_argument('--latent_size', '-ls', type=int, default=0, help="dimension of the latent feature")
    parser.add_argument('--fusion_strategy', default=dict(), help='args for the fusion strategy')
    
    parser.add_argument('--num_cycles', '-nc', type=int, default=0, help="training cycles of the VAE")
    parser.add_argument('--num_epoch_per_cycle', '-nepc', type=int, default=0, help="training epochs of the VAE")
    parser.add_argument('--thresh', '-th', type=int, default=50, help='threshold of the classification gate')
    parser.add_argument('--temp', '-t', type=int, default=2, help='temperature of the classification gate')
    parser.add_argument('--acc_type', '-at', type=str, default='avg', help='calculation method')
    
    parser.add_argument('--text_type', '-tt', type=str, default='lb_MAad_MAmd_LLMad_LLMmd', help='text type')
    parser.add_argument('--split_type', '-st', type=str, default='r', help="split type")
    
    parser.add_argument('--gen_text_feat', '-gt', action='store_true', help="generate text feature")
    parser.add_argument('--gen_split', '-gs', action='store_true', help="generate splits")
    parser.add_argument('--load_weight', '-lw', action='store_true', help="load weight")
    parser.add_argument('--mode', '-m', type=str, default='', help='zsl, gzsl, gate')
    
    
    return parser

def update_parameters(parser, args):
    if os.path.exists('./config/{}.yaml'.format(args.config)):
        with open('./config/{}.yaml'.format(args.config), 'r') as f:
            try:
                yaml_arg = yaml.load(f, Loader=yaml.FullLoader)
            except:
                yaml_arg = yaml.load(f)
            default_arg = vars(args)
            for k in yaml_arg.keys():
                if k not in default_arg.keys():
                    raise ValueError('Do NOT exist this parameter {}'.format(k))
            parser.set_defaults(**yaml_arg)
    else:
        raise ValueError('Do NOT exist this file in \'config\' folder: {}.yaml!'.format(args.config))
    return parser.parse_args()

def main():
    parser = init_parser()
    args = parser.parse_args()
    if args.config:
        args = update_parameters(parser, args) # cmd > yaml > default

    if args.gen_text_feat:
        tg = Text_Generator(args)
        tg.start()
    elif args.gen_split:
        sg = Split_Generator(args)
        sg.start()
    else:
        p = Processor(args)
        p.start()


if __name__ == '__main__':
    main()
