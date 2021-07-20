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
        datasets, ["id", "wav", "sig", "wrd", "tokens_bos", "tokens_eos", "tokens"],
    )
    return train_data, valid_data, test_datasets, tokenizer

import k2
import torch
import torchaudio
from tqdm import tqdm
from pathlib import Path
from hyperpyyaml import load_hyperpyyaml
from snowfall.common import write_error_stats
from snowfall.common import get_texts, get_phone_symbols
from snowfall.training.ctc_graph import build_ctc_topo, build_ctc_topo2
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.utils.metric_stats import ErrorRateStats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model():
    model = EncoderDecoderASR.from_hparams(
            source = 'speechbrain/asr-transformer-transformerlm-librispeech',
            savedir = 'pretrained_models/asr-transformer-transformerlm-librispeech',)
    
    model.modules.to(device)
    model.hparams.ctc_lin.to(device)

    return model

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # CLI:
    hparams_file = 'transformer.yaml'
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    # here we create the datasets objects as well as tokenization and encoding
    
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)
    
    # Trainer initialization
    model = load_model()
    
    model.device = device

    lang_dir = Path('data/lang_nosp')
       
    symbols = k2.SymbolTable.from_file(lang_dir / 'phones.txt')
    phone_ids = get_phone_symbols(symbols)

    phone_ids_with_blank = [0] + phone_ids

    ctc_topo = build_ctc_topo(phone_ids_with_blank)
    ctc_topo = k2.create_fsa_vec([ctc_topo]).to(device)

    for k in test_datasets.keys():

        dataloader = torch.utils.data.DataLoader(test_datasets[k], 
                                                 batch_size=1,
                                                 shuffle=False,
                                                 sampler=None,
                                                 pin_memory=False)

        results0 = []
        results1 = []
         
        wer_metric = ErrorRateStats()
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                i += 1
                #print(batch.keys())
                #batch = batch.to(device)
                idx = batch['id']
                wav = batch['wav']
                tokens_bos = batch['tokens_bos'].to(device)
                wav_lens = torch.tensor([1.0]).to(device)
                wav_lens = wav_lens.to(device)
                #print(wav)
                #wav = model.load_audio(wav[0])
                wav, sr = torchaudio.load(wav[0], channels_first=False)
                wav = model.audio_normalizer(wav, sr)

                wavs = wav.unsqueeze(0).float().to(device)
                encoder_out = model.modules.encoder(wavs, wav_lens)
                ctc_logits = model.hparams.ctc_lin(encoder_out)
                ctc_log_probs = model.hparams.log_softmax(ctc_logits)
                            
                vocab_size = model.tokenizer.get_piece_size()
                 
                ####we can have a option.####
                log_probs = ctc_log_probs

                supervision_segments = torch.tensor([[0, 0, ctc_log_probs.size(1)]],
                                                  dtype=torch.int32)
                
                indices = torch.argsort(supervision_segments[:, 2], descending=True)

                dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

                lattices = k2.intersect_dense_pruned(ctc_topo, dense_fsa_vec, 20.0, 8, 30, 10000)
                #lattices = k2.intersect_dense(HLG, dense_fsa_vec, 10.0)
                
                best_paths = k2.shortest_path(lattices, use_double_scores=True)
                aux_labels = best_paths[0].aux_labels

                aux_labels = aux_labels[aux_labels.nonzero().squeeze()]

                aux_labels = aux_labels[:-1]

                hyps = model.tokenizer.decode_ids(aux_labels.tolist())
            
                ref = batch['wrd']
                
                predicted_words = [str(hyps).split(' ')]
                target_words = [str(ref[0]).split(' ')]
                
                wer_metric.append(idx, predicted_words, target_words)
        
        if k == 'test-clean':
            with open('k2-encode_out_test-clean-ctc_topo.txt', 'w') as w:
                wer_metric.write_stats(w)
        
        if k == 'test-other':
            with open('k2-encode_out_test-other-ctc_topo.txt', 'w') as w:
                wer_metric.write_stats(w)


