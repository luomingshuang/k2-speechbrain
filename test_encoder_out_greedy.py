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
from snowfall.common import get_texts
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


if __name__ == "__main__":
    # CLI:
    hparams_file = 'hparams/transformer.yaml'
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin)
    
    # here we create the datasets objects as well as tokenization and encoding
    
    train_data, valid_data, test_datasets, tokenizer = dataio_prepare(hparams)
    
    # Trainer initialization
    model = load_model()

    model.device = device

    for k in test_datasets.keys():

        dataloader = torch.utils.data.DataLoader(test_datasets[k], 
                                                 batch_size=1,
                                                 shuffle=False,
                                                 sampler=None,
                                                 pin_memory=False)
           
        wer_metric = ErrorRateStats()
        
        with torch.no_grad():
            for batch in tqdm(dataloader):
                i += 1
                idx = batch['id']
                wav = batch['wav']
                tokens_bos = batch['tokens_bos'].to(device)
                wav_lens = torch.tensor([1.0]).to(device)
                #print(wav)
                #wav = model.load_audio(wav[0])
                wav, sr = torchaudio.load(wav[0], channels_first=False)
                wav = model.audio_normalizer(wav, sr)

                wavs = wav.unsqueeze(0).float().to(device)
                encoder_out = model.modules.encoder(wavs, wav_lens)
                
                ctc_logits = model.hparams.ctc_lin(encoder_out)
                ctc_log_probs = model.hparams.log_softmax(ctc_logits)
                
                _, indices_ctc = ctc_log_probs[0].topk(1,1,True,True)
                ctc_ids = torch.squeeze(indices_ctc, -1).tolist()

                ctc_ids = list(filter(lambda x:x != 0, ctc_ids))
                ctc_hyp = model.tokenizer.decode_ids(ctc_ids)
                
                ref = batch['wrd']
        
                predicted_words = [str(ctc_hyp).split(' ')]
                target_words = [str(ref[0]).split(' ')]

                wer_metric.append(idx, predicted_words, target_words)
         
        if k == 'test-clean':
            with open('speechbrain-test-clean-encoder_out-max-top1.txt', 'w') as w:
                wer_metric.write_stats(w)
        
        if k == 'test-other':
            with open('speechbrain-test-other-encoder_out-max-top1.txt', 'w') as w:
                wer_metric.write_stats(w)

        
