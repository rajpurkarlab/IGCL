"""
Run sampling to build contrastive examples offline.
"""

import logging
import argparse

from utils import logging_config
from model.sampler import RandomSampler

logger = logging.getLogger('transfer')

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("indexed_report_file", type=str)
    parser.add_argument("output_file", type=str)
    parser.add_argument("--method", type=str, choices=['random'], default='random')
    parser.add_argument("--sections", nargs='+', type=str, default=['impression', 'findings'], help="Sections to use in the report.")
    parser.add_argument("--n_study", type=int, default=-1, help="Number of total studies to sample; by default build samples for all studies")
    parser.add_argument("--n_negative", type=int, default=10, help="Number of negative examples for each positive sentence")

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    sampler = None
    if args.method == 'random':
        sampler = RandomSampler(args.indexed_report_file, n_study=args.n_study, k=args.n_negative, sections=args.sections)
    else:
        raise Exception("Unsupported sampling method: " + args.method)
    
    logger.info(f"Running sampling method: {sampler.__class__.__name__}, with indexed report file: {args.indexed_report_file}")
    sampler.run(args.output_file)

if __name__ == "__main__":
    main()
