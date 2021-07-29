#!/use/bin/env python3

# Copyright (c)  2021  Xiaomi Corporation (authors: Mingshuang Luo)

import k2
import os
import csv
import glob
import sys
import logging
import os
import torch
import argparse
import torchaudio
from tqdm import tqdm
from k2 import Fsa, SymbolTable
from pathlib import Path
from typing import List, Union

from snowfall.common import write_error_stats
from snowfall.common import get_texts
from snowfall.common import str2bool
from snowfall.common import find_first_disambig_symbol
from snowfall.common import get_phone_symbols
from snowfall.common import get_texts
from snowfall.decoding.graph import compile_HLG
from snowfall.training.ctc_graph import build_ctc_topo, build_ctc_topo2

from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats

from lm_rescore import rescore_with_n_best_list
from lm_rescore import rescore_with_whole_lattice

logger = logging.getLogger(__name__)
SAMPLERATE = 16000

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def create_csv(save_folder, info_lst, split, select_n_sentences=None):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    info_lst : list
        The list of info (including id, wav, words and spk_id) from wav files of a given data split.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.
    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, split + ".csv")

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "wav", "wrd", "spk_id", "duration"]]

    snt_cnt = 0
    # Processing all the wav files in wav_lst
    for info in info_lst:

        snt_id = info[0]
        spk_id = info[3]
        
        signal, fs = torchaudio.load(info[1])
        signal = signal.squeeze(0)
        duration = signal.shape[0] / SAMPLERATE

        csv_line = [
            snt_id,
            info[1],
            str(info[2]),
            spk_id,
            str(duration)
        ]

        #  Appending current file to the csv_lines list
        csv_lines.append(csv_line)
        snt_cnt = snt_cnt + 1

        if select_n_sentences is not None and snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )
        ##up
        #print(csv_lines[4])
        #csv_lines[1:].sort(key=float(csv_lines[1:][4]))
        sorted(csv_lines[1:], key=lambda x:x[4])

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)

def load_model():
    model = EncoderDecoderASR.from_hparams(
            source = 'speechbrain/asr-transformer-transformerlm-librispeech',
            savedir = 'pretrained_models/asr-transformer-transformerlm-librispeech',)

    model.modules.to(device)
    model.hparams.ctc_lin.to(device)

    return model

def nbest_decoding(lats: k2.Fsa, num_paths: int):
    '''
    (Ideas of this function are from Dan)

    It implements something like CTC prefix beam search using n-best lists

    The basic idea is to first extra n-best paths from the given lattice,
    build a word seqs from these paths, and compute the total scores
    of these sequences in the log-semiring. The one with the max score
    is used as the decoding output.
    '''

    # First, extract `num_paths` paths for each sequence.
    # paths is a k2.RaggedInt with axes [seq][path][arc_pos]
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)

    # word_seqs is a k2.RaggedInt sharing the same shape as `paths`
    # but it contains word IDs. Note that it also contains 0s and -1s.
    # The last entry in each sublist is -1.

    word_seqs = k2.index(lats.aux_labels, paths)
    # Note: the above operation supports also the case when
    # lats.aux_labels is a ragged tensor. In that case,
    # `remove_axis=True` is used inside the pybind11 binding code,
    # so the resulting `word_seqs` still has 3 axes, like `paths`.
    # The 3 axes are [seq][path][word]

    # Remove epsilons and -1 from word_seqs
    word_seqs = k2.ragged.remove_values_leq(word_seqs, 0)

    # Remove repeated sequences to avoid redundant computation later.
    #
    # Since k2.ragged.unique_sequences will reorder paths within a seq,
    # `new2old` is a 1-D torch.Tensor mapping from the output path index
    # to the input path index.
    # new2old.numel() == unique_word_seqs.num_elements()
    unique_word_seqs, _, new2old = k2.ragged.unique_sequences(
        word_seqs, need_num_repeats=False, need_new2old_indexes=True)
    # Note: unique_word_seqs still has the same axes as word_seqs

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seqs.shape(), 0)

    # path_to_seq_map is a 1-D torch.Tensor.
    # path_to_seq_map[i] is the seq to which the i-th path
    # belongs.
    path_to_seq_map = seq_to_path_shape.row_ids(1)

    # Remove the seq axis.
    # Now unique_word_seqs has only two axes [path][word]
    unique_word_seqs = k2.ragged.remove_axis(unique_word_seqs, 0)

    # word_fsas is an FsaVec with axes [path][state][arc]
    word_fsas = k2.linear_fsa(unique_word_seqs)

    word_fsas_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsas)

    # lats has phone IDs as labels and word IDs as aux_labels.
    # inv_lats has word IDs as labels and phone IDs as aux_labels
    inv_lats = k2.invert(lats)
    inv_lats = k2.arc_sort(inv_lats) # no-op if inv_lats is already arc-sorted

    path_lats = k2.intersect_device(inv_lats,
                                    word_fsas_with_epsilon_loops,
                                    b_to_a_map=path_to_seq_map,
                                    sorted_match_a=True)
    # path_lats has word IDs as labels and phone IDs as aux_labels

    path_lats = k2.top_sort(k2.connect(path_lats.to('cpu')).to(lats.device))

    tot_scores = path_lats.get_tot_scores(True, True)
    # RaggedFloat currently supports float32 only.
    # We may bind Ragged<double> as RaggedDouble if needed.
    ragged_tot_scores = k2.RaggedFloat(seq_to_path_shape,
                                       tot_scores.to(torch.float32))

    argmax_indexes = k2.ragged.argmax_per_sublist(ragged_tot_scores)

    # Since we invoked `k2.ragged.unique_sequences`, which reorders
    # the index from `paths`, we use `new2old`
    # here to convert argmax_indexes to the indexes into `paths`.
    #
    # Use k2.index here since argmax_indexes' dtype is torch.int32
    best_path_indexes = k2.index(new2old, argmax_indexes)

    paths_2axes = k2.ragged.remove_axis(paths, 0)

    # best_paths is a k2.RaggedInt with 2 axes [path][arc_pos]
    best_paths = k2.index(paths_2axes, best_path_indexes)

    # labels is a k2.RaggedInt with 2 axes [path][phone_id]
    # Note that it contains -1s.
    labels = k2.index(lats.labels.contiguous(), best_paths)

    labels = k2.ragged.remove_values_eq(labels, -1)

    # lats.aux_labels is a k2.RaggedInt tensor with 2 axes, so
    # aux_labels is also a k2.RaggedInt with 2 axes
    aux_labels = k2.index(lats.aux_labels, best_paths.values())

    best_path_fsas = k2.linear_fsa(labels)
    best_path_fsas.aux_labels = aux_labels

    return best_path_fsas

def decode(sample: List, 
           model,
           output_beam_size: int,
           num_paths: int,
           HLG: k2.Fsa,
           G: None,
           lm_scale_list: List,
           symbols: SymbolTable,
           use_whole_lattice: bool
           ):
    # model = load_model()
    #lm_scale_list = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    #lm_scale_list = [lm_scale]
    with torch.no_grad():
        wav = sample[1]
        txt = sample[2]
        wav_lens = torch.tensor([1.0]).to(device)
        sig, sr = torchaudio.load(wav, channels_first=False)
        sig = model.audio_normalizer(sig, sr)
        sig = sig.unsqueeze(0).float().to(device)

        encoder_output = model.modules.encoder(sig, wav_lens)
        ctc_logits = model.hparams.ctc_lin(encoder_output)
        ctc_log_probs = model.hparams.log_softmax(ctc_logits)

        supervision_segments = torch.tensor([[0, 0, ctc_log_probs.size(1)]],
                                                    dtype=torch.int32)

        indices = torch.argsort(supervision_segments[:, 2], descending=True)

        dense_fsa_vec = k2.DenseFsaVec(ctc_log_probs, supervision_segments)

        lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, output_beam_size, 30, 10000)

        if G is None:
            if num_paths > 1:
                best_paths = nbest_decoding(lattices, num_paths)
                key = f'no_resocre-{num_paths}'
            else:
                key = 'no_rescore'
                best_paths = k2.shortest_path(lattices, use_double_scores=True)

            hyps = get_texts(best_paths, indices)
            hyps = ' '.join([symbols.get(x) for x in hyps[0]])
    
            return hyps, txt
        
        logging.debug('use_whole_lattice: ', use_whole_lattice)

        if use_whole_lattice:
            logging.debug(f'Using whole lattice for decoding:')
            best_paths_dict = rescore_with_whole_lattice(lattices, G, lm_scale_list)

        else:
            logging.debug(f'Using nbest paths for decoding:')
            best_paths_dict = rescore_with_n_best_list(lattices, G, num_paths, lm_scale_list)
        
        for lm_scale_str, best_paths in best_paths_dict.items():
            hyps = get_texts(best_paths, indices) 
            hyps = ' '.join([symbols.get(x) for x in hyps[0]])
            
        return hyps, txt

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--use-lm-rescoring',
        type=str2bool,
        default=True,
        help='When enabled, it uses LM for rescoring')
    
    parser.add_argument(
        '--use-whole-lattice',
        type=str2bool,
        default=True,
        help='When enabled, it uses the whole lattice for decoding.')

    parser.add_argument(
        '--num-paths',
        type=int,
        default=-1,
        help='Number of paths for rescoring using n-best list.' \
             'If it is negative, then rescore with the whole lattice.'\
             'CAUTION: You have to reduce max_duration in case of CUDA OOM'
             )

    parser.add_argument(
        '--output-beam-size',
        type=float,
        default=8,
        help='Output beam size. Used in k2.intersect_dense_pruned.'\
             'Choose a large value (e.g., 20), for 1-best decoding '\
             'and n-best rescoring. Choose a small value (e.g., 8) for ' \
             'rescoring with the whole lattice')
    
    return parser

def locate_corpus(*corpus_dirs):
    for d in corpus_dirs:
        if os.path.exists(d):
            return d
    logging.debug(f"Please create a place on your system to put the downloaded Librispeech data "
          "and add it to `corpus_dirs`")
    sys.exit(1)

def main():
    parser = get_parser()
    args = parser.parse_args()

    num_paths = args.num_paths
    use_lm_rescoring = args.use_lm_rescoring
    use_whole_lattice = args.use_whole_lattice

    lang_dir = Path('data/lang_nosp')

    symbol_table = k2.SymbolTable.from_file(lang_dir / 'words.txt')
    phone_symbol_table = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    phone_ids = get_phone_symbols(phone_symbol_table)
    phone_ids_with_blank = [0] + phone_ids

    ctc_topo = k2.arc_sort(build_ctc_topo(phone_ids_with_blank))
    #ctc_topo = k2.arc_sort(build_ctc_topo2(list(range(5000))))

    if not os.path.exists(lang_dir / 'HLG.pt'):
        logging.debug("Loading L_disambig.fst.txt")
        with open(lang_dir / 'L_disambig.fst.txt') as f:
            L = k2.Fsa.from_openfst(f.read(), acceptor=False)
        logging.debug("Loading G.fst.txt")
        with open(lang_dir / 'G.fst.txt') as f:
            G = k2.Fsa.from_openfst(f.read(), acceptor=False)
        first_phone_disambig_id = find_first_disambig_symbol(phone_symbol_table)
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        HLG = compile_HLG(L=L,
                         G=G,
                         H=ctc_topo,
                         labels_disambig_id_start=first_phone_disambig_id,
                         aux_labels_disambig_id_start=first_word_disambig_id)
        torch.save(HLG.as_dict(), lang_dir / 'HLG.pt')
    else:
        logging.debug("Loading pre-compiled HLG")
        d = torch.load(lang_dir / 'HLG.pt')
        HLG = k2.Fsa.from_dict(d)

    if use_lm_rescoring:
        if use_whole_lattice:
            logging.info('Rescoring with the whole lattice')
        else:
            logging.info(f'Rescoring with n-best list, n is {num_paths}')
        first_word_disambig_id = find_first_disambig_symbol(symbol_table)
        if not os.path.exists(lang_dir / 'G_4_gram.pt'):
            logging.debug('Loading G_4_gram.fst.txt')
            with open(lang_dir / 'G_4_gram.fst.txt') as f:
                G = k2.Fsa.from_openfst(f.read(), acceptor=False)
                # G.aux_labels is not needed in later computations, so
                # remove it here.
                del G.aux_labels
                # CAUTION(fangjun): The following line is crucial.
                # Arcs entering the back-off state have label equal to #0.
                # We have to change it to 0 here.
                G.labels[G.labels >= first_word_disambig_id] = 0
                G = k2.create_fsa_vec([G]).to(device)
                G = k2.arc_sort(G)
                torch.save(G.as_dict(), lang_dir / 'G_4_gram.pt')
        else:
            logging.debug('Loading pre-compiled G_4_gram.pt')
            d = torch.load(lang_dir / 'G_4_gram.pt')
            G = k2.Fsa.from_dict(d).to(device)

        if use_whole_lattice:
            # Add epsilon self-loops to G as we will compose
            # it with the whole lattice later
            G = k2.add_epsilon_self_loops(G)
            G = k2.arc_sort(G)
            G = G.to(device)
        # G.lm_scores is used to replace HLG.lm_scores during
        # LM rescoring.
        G.lm_scores = G.scores.clone()
    else:
        logging.debug('Decoding without LM rescoring')
        G = None
        if num_paths > 1:
            logging.debug(f'Use n-best list decoding, n is {num_paths}')
        else:
            logging.debug('Use 1-best decoding')

    logging.debug("convert HLG to device")
    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)

    if not hasattr(HLG, 'lm_scores'):
        HLG.lm_scores = HLG.scores.clone()

    model = load_model()

    data_dir = locate_corpus(
        '/export/corpora5/LibriSpeech',
        #'/home/storage04/zhuangweiji/data/open-source-data/librispeech/LibriSpeech',
        '/root/fangjun/data/librispeech/LibriSpeech',
        '/kome/luomingshuang/audio-data/LibriSpeech'
    )

    test_dirs = ['test-clean', 'test-other']

    #lm_scale_list = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]
    lm_scale_list = [0.2] 

    wer_metric = ErrorRateStats()

    for lm_scale in lm_scale_list:
        lm_scale_list = [lm_scale]
        logging.debug(f'Using the lm_scale {lm_scale} for decoding...')

        for test_dir in test_dirs:
            samples = []

            csv_file = os.path.join(data_dir, str(test_dir)+'.csv')

            if not os.path.exists(csv_file):
                info_lists = []
                txt_files = glob.glob(os.path.join(data_dir, test_dir, '*', '*', '*.txt'))
                for txt_file in txt_files:
                    with open(txt_file, 'r') as f:
                        lines = f.readlines()
                        for line in lines:
                            items = line.strip().split(' ')
                            flac = os.path.join(data_dir, test_dir, '/'.join(items[0].split('-')[:2])+'/'+items[0]+'.flac')
                            text = ' '.join(items[1:])
                            spk_id = '-'.join(items[0].split('-')[0:2])
                            id = items[0]

                            samples.append((id, flac, text, spk_id))

                create_csv(data_dir, samples, test_dir)
            
            else:
                with open(csv_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[1:]:
                        items = line.split(',')
                        id = items[0]
                        flac = items[1]
                        text = items[2]
                        spk_id = items[3]

                        samples.append((id, flac, text, spk_id))

            for sample in tqdm(samples):
                idx = sample[0]
                hyps, ref = decode(sample=sample, model=model, 
                       output_beam_size=args.output_beam_size, 
                       num_paths=args.num_paths, HLG=HLG, G=G, 
                       lm_scale_list=lm_scale_list, symbols=symbol_table,
                       use_whole_lattice=use_whole_lattice)

                pred = [str(hyps).split(' ')]
                grth = [str(ref).split(' ')]
                
                wer_metric.append(idx, pred, grth)

            if use_lm_rescoring:
                if test_dir == 'test-clean':
                    with open(f'test-clean-lm-scale-{lm_scale}.txt', 'w') as f:
                        wer_metric.write_stats(f)

                if test_dir == 'test-other':
                    with open(f'test-other-lm-scale-{lm_scale}.txt', 'w') as f:
                        wer_metric.write_stats(f)
            else:
                if test_dir == 'test-clean':
                    with open(f'test-clean-no-lm-rescoring.txt', 'w') as f:
                        wer_metric.write_stats(f)

                if test_dir == 'test-other':
                    with open(f'test-other-no-lm-rescoring.txt', 'w') as f:
                        wer_metric.write_stats(f)

if __name__  == '__main__':
    main()
