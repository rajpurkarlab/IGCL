"""
Sampler modules that take in a list of sentences and generate contrastive pairs.
"""

import logging
import json
import random
from tqdm import tqdm
from collections import defaultdict

logger = logging.getLogger('transfer')
random.seed(1234)

def load_json(filename):
    with open(filename) as infile:
        data = json.load(infile)
    return data

class Sampler():
    """
    A sampler base class.
    """
    def __init__(self):
        pass

    def run(self):
        raise NotImplementedError("Function not implemented.")

    def get_candidiates(self, sentence, all, N):
        """
        Get a list of N candidate contrastive sentences for a sentence. Usually
        this is done via fast search method, such as uniformly random sampling, 
        or tf-idf based match.
        """
        raise NotImplementedError("Function not implemented.")

    def get_contrastives(self, sentence, candidates, k):
        """
        Find k most contrastive sentences from the list of candidates. This is done
        via exact contrastive search method, such as NLI or BERTScore based.
        """
        raise NotImplementedError("Function not implemented.")


class RandomSampler(Sampler):
    """
    A sampler class that builds contrastive pairs based on random sampling.
    """
    def __init__(self, indexed_report_file, n_study=-1, k=10, sections=['impression', 'findings']):
        self.report_file = indexed_report_file
        self.n_study = n_study
        self.k = k
        self.sections = sections
    
    def run(self, output_file=None):
        """
        Run random sampling by loading reports from file and generating random contrastive examples.
        """
        # first read from report file
        uniq_sentences, sid2sections = self.load_indexed_reports(self.report_file)
        logger.info(f"{len(uniq_sentences)} unique sentences and {len(sid2sections)} reports loaded from {self.report_file}")
        # keep and merge relevant sections
        sid2sentences = dict()
        total_sentences = 0
        for sid, sections in sid2sections.items():
            sents = []
            for sec_name in self.sections:
                if sec_name in sections:
                    sents += sections[sec_name] # merge sections
            total_sentences += len(sents)
            sid2sentences[sid] = sents
        logger.info(f"A total of {total_sentences} sentences are found in all reports.")

        # build contrastive samples for a subset of all studies
        if self.n_study > 0:
            sampled_items = random.sample(sid2sentences.items(), self.n_study)
            sid2sentences = dict(sampled_items)
            logger.info(f"Sampled only {len(sid2sentences)} individual studies to build contrastive samples with.")
        
        # for every sentence in a report, sample k contrastive examples
        all_samples = self.generate_all_contrastives(uniq_sentences, sid2sentences)

        self.sentences = uniq_sentences
        self.examples = all_samples
        
        # write to file
        if output_file is not None:
            output_data = {
                'sentences': self.sentences,
                'examples': self.examples
            }
            with open(output_file, 'w') as outfile:
                json.dump(output_data, outfile)
            logger.info(f"Results written to file: {output_file}")
    
    def load_indexed_reports(self, indexed_report_file):
        data = load_json(indexed_report_file)
        sentences = data['sentences']
        sid2sections = data['indexed_reports']
        return sentences, sid2sections

    def generate_all_contrastives(self, uniq_sentences, sid2sentences):
        """
        Return all list of contrastive examples in the following format:
            - sid -> study id
            - ctvs -> list of list of (k+1) sentence ids
                - [pos, neg, neg, ...]
                - [pos, neg, neg, ...]
                - ...
        """
        logger.info("Start generating contrastive examples with random sampling...")
        all_samples = []
        total_ctvs = 0 # total number of contrastive lists
        for sid, sent_idxs in tqdm(sid2sentences.items()):
            # build candidate pool by removing indices of existing sentences in report
            # candidates = build_candidates(len(uniq_sentences), sent_idxs)
            # we use all sentences as candidates since low chance to sample sentences from the same report
            candidates = list(range(len(uniq_sentences)))
            entry = dict()
            entry['sid'] = sid
            entry['ctvs'] = []
            for sidx in sent_idxs:
                ctv = [sidx] # first item is positive example
                negs = random.sample(candidates, self.k) # k negative sentences
                ctv += negs
                entry['ctvs'].append(ctv) # [pos, neg, neg, ...]
                total_ctvs += 1
            all_samples.append(entry)
        logger.info(f"{total_ctvs} total contrastive lists created for {len(all_samples)} individual studies.")
        return all_samples