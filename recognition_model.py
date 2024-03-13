import os
import sys
import numpy as np
import logging
import subprocess
#from ctcdecode import CTCBeamDecoder
import jiwer
import tqdm

from datasets import load_dataset
import torch
from torch import nn
import torch.nn.functional as F

from read_emg import EMGDataset, SizeAwareSampler
from align import align_from_distances
from architecture import Model, KoLeoLoss, CrossConLoss, WeightedSupTConLoss
from data_utils import combine_fixed_length, decollate_tensor

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('debug', False, 'debug')
flags.DEFINE_string('output_directory', 'output', 'where to save models and outputs')
flags.DEFINE_integer('batch_size', 32, 'training batch size')
flags.DEFINE_float('learning_rate', 3e-4, 'learning rate')
flags.DEFINE_integer('learning_rate_warmup', 1000, 'steps of linear warmup')
flags.DEFINE_integer('learning_rate_patience', 5, 'learning rate decay patience')
flags.DEFINE_string('start_training_from', None, 'start training from this model')
flags.DEFINE_float('l2', 0, 'weight decay')
flags.DEFINE_string('evaluate_saved', None, 'run evaluation on given model file')
flags.DEFINE_float('emg_ctc_loss_weight', 1.0, 'emg loss weighting')
flags.DEFINE_float('audio_ctc_loss_weight', 1.0, 'audio loss weighting')
flags.DEFINE_float('crosscon_loss_weight', 1.0, 'crosscon loss weighting')
flags.DEFINE_float('suptcon_loss_weight', 1.0, 'suptcon loss weighting')
flags.DEFINE_float('koleo_loss_weight', 0.1, 'koleo loss weighting')

PHONEME_WEIGHTS = {
    # Lower weights for challenging distinctions
    'b': 0.5, 'p': 0.5,  # Voiced-voiceless pairs
    'd': 0.5, 't': 0.5,
    'g': 0.5, 'k': 0.5,
    'v': 0.5, 'f': 0.5,
    'z': 0.5, 's': 0.5,
    'm': 0.7, 'n': 0.7, 'ng': 0.7,  # Nasals, given their difficulty due to velum positioning
}

"""
def test(model, testset, device):
    model.eval()

    blank_id = len(testset.text_transform.chars)
    decoder = CTCBeamDecoder(testset.text_transform.chars+'_', blank_id=blank_id, log_probs_input=True,
            model_path='lm.binary', alpha=1.5, beta=1.85)

    dataloader = torch.utils.data.DataLoader(testset, batch_size=1)
    references = []
    predictions = []
    with torch.no_grad():
        for example in tqdm.tqdm(dataloader, 'Evaluate', disable=None):
            X = example['emg'].to(device)
            X_raw = example['raw_emg'].to(device)
            sess = example['session_ids'].to(device)

            pred  = F.log_softmax(model(X, X_raw, sess), -1)

            beam_results, beam_scores, timesteps, out_lens = decoder.decode(pred)
            pred_int = beam_results[0,0,:out_lens[0,0]].tolist()

            pred_text = testset.text_transform.int_to_text(pred_int)
            target_text = testset.text_transform.clean_text(example['text'][0])

            references.append(target_text)
            predictions.append(pred_text)

    model.train()
    return jiwer.wer(references, predictions)
"""

def train_model(trainset, devset, device, n_epochs=200):
    crosscon_loss_function = CrossConLoss()
    weighted_suptcon_loss_function = WeightedSupTConLoss(PHONEME_WEIGHTS)
    koleo_loss_function = KoLeoLoss()

    dataloader = torch.utils.data.DataLoader(trainset, pin_memory=(device=='cuda'), num_workers=0, collate_fn=EMGDataset.collate_raw, batch_sampler=SizeAwareSampler(trainset, 128000))
                                 
    n_chars = len(devset.text_transform.chars)
    model = Model(devset.num_features, n_chars+1).to(device)

    if FLAGS.start_training_from is not None:
        state_dict = torch.load(FLAGS.start_training_from, map_location=torch.device(device))
        model.load_state_dict(state_dict, strict=False)

    optim = torch.optim.AdamW(model.parameters(), lr=FLAGS.learning_rate, weight_decay=FLAGS.l2)
    lr_sched = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=[125,150,175], gamma=.5)

    def set_lr(new_lr):
        for param_group in optim.param_groups:
            param_group['lr'] = new_lr

    target_lr = FLAGS.learning_rate
    def schedule_lr(iteration):
        iteration = iteration + 1
        if iteration <= FLAGS.learning_rate_warmup:
            set_lr(iteration*target_lr/FLAGS.learning_rate_warmup)

    batch_idx = 0
    optim.zero_grad()
    for epoch_idx in range(n_epochs):
        losses = []
        for example in tqdm.tqdm(dataloader, 'Train step', disable=None):
            schedule_lr(batch_idx)

            emg = combine_fixed_length(example['raw_emg'], 200*8).to(device)
            emg_voiced_parallel = combine_fixed_length(example['parallel_voiced_emg'], 200).to(device)
            audio = combine_fixed_length(example['audio_features'], 200).to(device)
            sess = combine_fixed_length(example['session_ids'], 200).to(device)

            emg_length = example['lengths']
            audio_length = example['audio_feature_lengths']
            parallel_emg_audio_label = example['text_int']
            parallel_emg_audio_label_length = example['text_int_lengths']
       
            emg_pred, emg_latent, emg_parallel_latent, audio_pred, audio_latent = model(
                emg, 
                emg_voiced_parallel, 
                audio,
                sess
            )
            emg_pred = nn.utils.rnn.pad_sequence(decollate_tensor(emg_pred, emg_length)).to(device)
            emg_latent = nn.utils.rnn.pad_sequence(decollate_tensor(emg_latent, emg_length)).to(device)
            emg_parallel_latent = nn.utils.rnn.pad_sequence(decollate_tensor(emg_parallel_latent, emg_length)).to(device)
            #audio_pred = nn.utils.rnn.pad_sequence(decollate_tensor(audio_pred, audio_length))
            emg_audio_label = nn.utils.rnn.pad_sequence(parallel_emg_audio_label, batch_first=True).to(device)
            phonemes = nn.utils.rnn.pad_sequence(example['phonemes'], batch_first=True).to(device)

            # Indv modalities ctc losses
            emg_ctc_loss = F.ctc_loss(
                emg_pred, 
                torch.cat(parallel_emg_audio_label),
                torch.tensor(emg_length, dtype=torch.int), 
                torch.tensor(parallel_emg_audio_label_length, dtype=torch.int), 
            )
            """
            audio_ctc_loss = F.ctc_loss(
                audio_pred, 
                torch.cat(parallel_emg_audio_label),
                torch.tensor(audio_length, dtype=torch.int), 
                torch.tensor(parallel_emg_audio_label_length, dtype=torch.int), 
            )
            """
            # TODO: Figure out exactly what should we align, from the paper it sounds like the latents.
            # Align using aligner from transducer code (since it's silent speech)        
            costs = torch.cdist(
                emg_latent, 
                emg_parallel_latent,
            ).squeeze(0)
            
            alignment = align_from_distances(costs.permute(0, 2, 1).detach().cpu().numpy())
            _, aligned_parallel_indices = zip(*alignment)
            aligned_indices_tensor = torch.tensor(aligned_parallel_indices, dtype=torch.long)
            emg_parallel_aligned = emg_parallel_latent[aligned_indices_tensor]
            # Phoneme parallels to voiced emg, so it'll be dtw-adjusted the same way
            # TODO: May need to subsample to phoneme's temporal resolution as per chatgpt suggestion
            phoneme_parallel_aligned = phonemes[aligned_indices_tensor]

            # CrossCon (between silent latent and aligned emg latent)
            crosscon_loss =  crosscon_loss_function(emg_latent, emg_parallel_aligned)

            # supTcon (between predicted phonemes and emg_latent)
            wsuptcon_loss = weighted_suptcon_loss_function(emg_latent, phoneme_parallel_aligned)

            # KoLeo Regularizer Loss
            # SInce we're cross-conning with emg_parallel, let's try it with emg_parallel first.
            koleo_loss = koleo_loss_function(emg_latent, emg_parallel_aligned) 

            loss = (
                FLAGS.emg_ctc_loss_weight * emg_ctc_loss
                #+ FLAGS.audio_ctc_loss_weight * audio_ctc_loss
                + FLAGS.crosscon_loss_weight * crosscon_loss
                + FLAGS.suptcon_loss_weight *  wsuptcon_loss
                + FLAGS.koleo_loss_weight *  koleo_loss
            )

            losses.append(loss.item())

            loss.backward()
            if (batch_idx+1) % 2 == 0:
                optim.step()
                optim.zero_grad()

            batch_idx += 1
        train_loss = np.mean(losses)
        val = test(model, devset, device)
        lr_sched.step()
        logging.info(f'finished epoch {epoch_idx+1} - training loss: {train_loss:.4f} validation WER: {val*100:.2f}')
        torch.save(model.state_dict(), os.path.join(FLAGS.output_directory,'model.pt'))

    model.load_state_dict(torch.load(os.path.join(FLAGS.output_directory,'model.pt'), map_location=torch.device(device))) # re-load best parameters
    return model

def evaluate_saved():
    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'
    testset = EMGDataset(test=True)
    n_chars = len(testset.text_transform.chars)
    model = Model(testset.num_features, n_chars+1).to(device)
    model.load_state_dict(torch.load(FLAGS.evaluate_saved, map_location=torch.device(device)))
    print('WER:', test(model, testset, device))

def main():
    os.makedirs(FLAGS.output_directory, exist_ok=True)
    logging.basicConfig(handlers=[
            logging.FileHandler(os.path.join(FLAGS.output_directory, 'log.txt'), 'w'),
            logging.StreamHandler()
            ], level=logging.INFO, format="%(message)s")

    logging.info(subprocess.run(['git','rev-parse','HEAD'], stdout=subprocess.PIPE, universal_newlines=True).stdout)
    logging.info(subprocess.run(['git','diff'], stdout=subprocess.PIPE, universal_newlines=True).stdout)

    logging.info(sys.argv)

    # EMG Data
    trainset = EMGDataset(dev=False,test=False)
    devset = EMGDataset(dev=True)
    logging.info('output example: %s', devset.example_indices[0])
    logging.info('train / dev split: %d %d',len(trainset),len(devset))

    device = 'cuda' if torch.cuda.is_available() and not FLAGS.debug else 'cpu'

    model = train_model(trainset, devset, device)

if __name__ == '__main__':
    FLAGS(sys.argv)
    if FLAGS.evaluate_saved is not None:
        evaluate_saved()
    else:
        main()
