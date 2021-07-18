#!/usr/bin/env python3
import speechbrain as sb

def dataio_prepare(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions."""
    data_folder = hparams["data_folder"]

    train_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["train_csv"], replacements={"data_root": data_folder},
    )

    valid_data = sb.dataio.dataset.DynamicItemDataset.from_csv(
        csv_path=hparams["valid_csv"], replacements={"data_root": data_folder},
    )
    valid_data = valid_data.filtered_sorted(sort_key="duration")

    # test is separate
    test_datasets = {}
    for csv_file in hparams["test_csv"]:
        name = Path(csv_file).stem
        test_datasets[name] = sb.dataio.dataset.DynamicItemDataset.from_csv(
            csv_path=csv_file, replacements={"data_root": data_folder}
        )
        test_datasets[name] = test_datasets[name].filtered_sorted(
            sort_key="duration"
        )

    datasets = [train_data, valid_data] + [i for k, i in test_datasets.items()]

    # We get the tokenizer as we need it to encode the labels when creating
    # mini-batches.
    tokenizer = hparams["tokenizer"]

    # 2. Define audio pipeline:
    @sb.utils.data_pipeline.takes("wav")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav):
        sig = sb.dataio.dataio.read_audio(wav)
        return sig

    sb.dataio.dataset.add_dynamic_item(datasets, audio_pipeline)

    # 3. Define text pipeline:
    @sb.utils.data_pipeline.takes("wrd")
    @sb.utils.data_pipeline.provides(
        "wrd", "tokens_list", "tokens_bos", "tokens_eos", "tokens"
    )
    def text_pipeline(wrd):
        yield wrd
        tokens_list = tokenizer.encode_as_ids(wrd)
        yield tokens_list
        tokens_bos = torch.LongTensor([hparams["bos_index"]] + (tokens_list))
        yield tokens_bos
        tokens_eos = torch.LongTensor(tokens_list + [hparams["eos_index"]])
        yield tokens_eos
        tokens = torch.LongTensor(tokens_list)
        yield tokens

    sb.dataio.dataset.add_dynamic_item(datasets, text_pipeline)

    # 4. Set output:
    sb.dataio.dataset.set_output_keys(
        datasets, ["id", "wav", "wrd"],
    )
    return train_data, valid_data, test_datasets, tokenizer

import k2
import torch
import argparse
import torchaudio
from tqdm import tqdm
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from snowfall.common import write_error_stats
from snowfall.common import get_texts
from snowfall.common import str2bool
from snowfall.decoding.lm_rescore import rescore_with_n_best_list
from snowfall.decoding.lm_rescore import rescore_with_whole_lattice
from snowfall.training.ctc_graph import build_ctc_topo2
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats

def load_model():
    model = EncoderDecoderASR.from_hparams(
            source = 'speechbrain/asr-transformer-transformerlm-librispeech',
            savedir = 'pretrained_models/asr-transformer-transformerlm-librispeech',)

    return model

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--beam-size',
        type=int,
        default=8,
        help='Beam size for decoding.')
    parser.add_argument(
        '--use-lm-score',
        type=str2bool,
        default=True,
        help='Use lm score for decoding when True.')
    
    return parser

if __name__ == "__main__":

    parser = get_parser()
    args = parser.parse_args()

    hparams_file = 'transformer.yaml'
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    # here we create the datasets objects as well as tokenization and encoding
    
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)
    
    # Trainer initialization
    model = load_model()

    device = model.device 

    lang_dir = Path('lang_nosp')
   
    print("Loading G_4_gram.pt")
    d = torch.load(lang_dir / 'G_4_gram.pt')
    G = k2.Fsa.from_dict(d).to(device)

    use_whole_lattice = args.use_lm_score

    if use_whole_lattice:
        G = k2.add_epsilon_self_loops(G)
        G = k2.arc_sort(G)
        G = G.to(device)
    
    G.lm_scores = G.scores.clone()
    
    lm_scale_list = [1.2]

    print("Loading pre-compiled HLG")
    d = torch.load(lang_dir / 'HLG.pt')
    HLG = k2.Fsa.from_dict(d)
    HLG = HLG.to(device)
    HLG.aux_labels = k2.ragged.remove_values_eq(HLG.aux_labels, 0)
    HLG.requires_grad_(False)
    
    symbols = k2.SymbolTable.from_file(lang_dir / 'words.txt')

    for k in test_datasets.keys():

        dataloader = torch.utils.data.DataLoader(test_datasets[k], 
                                                 batch_size=1,
                                                 shuffle=False,
                                                 sampler=None,
                                                 pin_memory=False)
        i = 0
         
        wer_metric = ErrorRateStats()
         
        with torch.no_grad():
            for batch in tqdm(dataloader):
                i += 1
                idx = batch['id']
                wav= batch['wav']
                
                wav_lens = torch.tensor([1.0]).to(device)
                wav, sr = torchaudio.load(wav[0], channels_first=False)
                wav = model.audio_normalizer(wav, sr)
                
                wavs = wav.unsqueeze(0).float().to(device)

                encoder_out = model.modules.encoder(wavs, wav_lens)
                ctc_logits = model.hparams.ctc_lin(encoder_out)
                ctc_log_probs = model.hparams.log_softmax(ctc_logits)
                            
                supervision_segments = torch.tensor([[0, 0, ctc_log_probs.size(1)]],
                                                  dtype=torch.int32)
                
                indices = torch.argsort(supervision_segments[:, 2], descending=True)

                dense_fsa_vec = k2.DenseFsaVec(ctc_log_probs, supervision_segments)
                
                bs = args.beam_size

                lattices = k2.intersect_dense_pruned(HLG, dense_fsa_vec, 20.0, bs, 30, 10000)
                
                best_path_dict = ''
                best_paths = ''
                hyps = ''
                
                if use_whole_lattice:
                    best_path_dict = rescore_with_whole_lattice(lattices, G, lm_scale_list)
                    for lm_scale_str, best_paths in best_path_dict.items():
                        hyps = get_texts(best_paths, indices)
                        hyps = ' '.join([symbols.get(x) for x in hyps[0]])
                
                else:
                    best_paths = k2.shortest_path(lattices, use_double_scores=True)
                    hyps = get_texts(best_paths, indices)
                    hyps = ' '.join([symbols.get(x) for x in hyps[0]])

                ref = batch['wrd']

                predicted_words = [str(hyps).split(' ')]
                target_words = [str(ref[0]).split(' ')]
                
                wer_metric.append(idx, predicted_words, target_words)
        
        if k == 'test-clean':
            with open('k2-encode_out_test_clean_bs_{}_HLG_use_whole_lattices_lm_scale_1.2.txt'.format(bs), 'w') as w:
                wer_metric.write_stats(w)
        
        if k == 'test-other':
            with open('k2-encode_out_test_other_bs_{}_HLG.txt_use_whole_lattices_lm_scale_1.2'.format(bs), 'w') as w:
                wer_metric.write_stats(w)
