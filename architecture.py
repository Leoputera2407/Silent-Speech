import random

from torch import nn
import torch.nn.functional as F

from transformer import TransformerEncoderLayer

from absl import flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('model_size', 768, 'number of hidden dimensions')
flags.DEFINE_integer('num_layers', 6, 'number of layers')
flags.DEFINE_float('dropout', .2, 'dropout')


class WeightedSupTConLoss(nn.Module):
    def __init__(self, weights):
        """
        Initialize the weighted supervised contrastive loss.

        Args:
            weights (dict): A dictionary mapping phoneme classes to their weights.
        """
        super(WeightedSupTConLoss, self).__init__()
        self.weights = weights

    def forward(self, embeddings, phonemes):
        """
        Calculate the Weighted Supervised Contrastive Loss.

        Args:
            embeddings (torch.Tensor): The embeddings for each phoneme, shape (batch_size, embedding_dim).
            phonemes (torch.Tensor): The phoneme classes for each embedding, shape (batch_size,).

        Returns:
            torch.Tensor: The calculated loss.
        """
        normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
        cos_sim = torch.einsum('ik,jk->ij', normalized_embeddings, normalized_embeddings)
        exp_sim = torch.exp(cos_sim)

        phoneme_classes = torch.unique(phonemes)
        final_loss = 0.0

        for phoneme_class in phoneme_classes:
            class_mask = phonemes == phoneme_class
            pos_mask = torch.einsum('i,j->ij', class_mask, class_mask) & ~torch.eye(phonemes.size(0), dtype=bool, device=embeddings.device)
            den_mask = ~torch.eye(phonemes.size(0), dtype=bool, device=embeddings.device)

            numerator = (exp_sim * pos_mask).sum(dim=1)[class_mask]
            denominator = (exp_sim * den_mask).sum(dim=1)[class_mask]

            class_loss = -torch.log(numerator / denominator).mean()
            
            # Apply weights to the phoneme class, if not present in dict, defaults to 1.0, meaning no extra punishment or reward.
            weight = self.weights.get(phoneme_class.item(), 1.0)
            weighted_class_loss = class_loss * weight

            final_loss += weighted_class_loss

        return final_loss / len(phoneme_classes)


class CrossConLoss(nn.Module):
    def __init__(self):
        super(CrossConLoss, self).__init__()

    def forward(self, x, y):
        """
        Compute the Cross-Domain Contrastive Loss between EMG (x) and Audio (y) latent representations.

        Args:
            x (torch.Tensor): EMG latent representations of shape (B, T, C).
            y (torch.Tensor): Audio latent representations of shape (B, T, C).

        Returns:
            torch.Tensor: The computed cross-domain contrastive loss.
        """
        T = x.size(1)  # Time steps
        B, _, C = x.shape  # Batch size, Time steps, Channels

        combined = torch.concat([x, y], dim=1)  # shape (B, 2T, C)

        norm_combined = combined / combined.norm(dim=2, keepdim=True)
        sim_matrix = torch.einsum('bik,bjk->bij', norm_combined, norm_combined)

        # Positive mask
        pos_mask = torch.zeros(2*T, 2*T, dtype=bool)
        rows = torch.arange(T)
        pos_mask[rows, rows + T] = True  # (tri_u)
        pos_mask[rows + T, rows] = True  # (tri_d)

        # Negative mask
        neg_mask = (~pos_mask)
        neg_mask[torch.eye(2*T, dtype=bool)] = False  # Remove self-similarity

        numerator = torch.exp(sim_matrix.masked_select(pos_mask)).view(B, -1)
        denominator = torch.sum(torch.exp(sim_matrix) * neg_mask.unsqueeze(0), dim=2)

        contrastive_loss = torch.mean(-torch.log(numerator / denominator.view(B, -1)))

        return contrastive_loss


class KoLeoLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[::(n + 1)].fill_(-1)  # Fill diagonal with -1 to ignore self-comparison
        _, I = torch.max(dots, dim=1)  # Find index of max dot product (nearest neighbor)
        return I

    def forward(self, emg_latent, audio_latent, eps=1e-8):
        joint_embedding = torch.cat([emg_latent, audio_latent], dim=1)
        joint_embedding = F.normalize(joint_embedding, p=2, dim=1, eps=eps)
        
        I = self.pairwise_NNs_inner(joint_embedding)
        distances = self.pdist(joint_embedding, joint_embedding[I])
        
        loss = -torch.log(distances + eps).mean()
        return loss


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape: int, eps: float = 0.00001):
        super().__init__()
        self.eps = eps
        # For 1D convolutions, `normalized_shape` should be the number of channels
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        var = x.var(dim=(1, 2), keepdim=True, unbiased=False)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x

class ResBlock(nn.Module):
    def __init__(self, num_ins, num_outs, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(num_ins, num_outs, 3, padding=1, stride=stride)
        self.ln1 = LayerNorm(num_outs)
        self.conv2 = nn.Conv1d(num_outs, num_outs, 3, padding=1)
        self.ln2 = LayerNorm(num_outs)

        if stride != 1 or num_ins != num_outs:
            self.residual_path = nn.Conv1d(num_ins, num_outs, 1, stride=stride)
            self.res_norm = LayerNorm(num_outs)
        else:
            self.residual_path = None

    def forward(self, x):
        input_value = x

        # Use SiLU (Swish) instead of ReLU
        x = F.silu(self.ln1(self.conv1(x)))
        x = self.ln2(self.conv2(x))

        if self.residual_path is not None:
            res = self.res_norm(self.residual_path(input_value))
        else:
            res = input_value

        # Use SiLU (Swish) in the final activation, as per conformer's recommendation.
        return F.silu(x + res)

class Model(nn.Module):
    def __init__(self, num_features, num_outs, num_aux_outs=None):
        super().__init__()

        self.emg_conv_blocks = nn.Sequential(
            ResBlock(8, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
            ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )

        self.emg_linear = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        self.audio_conv_blocks = nn.Sequential(
                ResBlock(
                    80, FLAGS.model_size, 2
                ),
                ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
                ResBlock(FLAGS.model_size, FLAGS.model_size, 2),
        )
        
        self.audio_linear = nn.Linear(FLAGS.model_size, FLAGS.model_size)

        encoder_layer = TransformerEncoderLayer(d_model=FLAGS.model_size, nhead=8, relative_positional=True, relative_positional_distance=100, dim_feedforward=3072, dropout=FLAGS.dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, FLAGS.num_layers)
        self.w_out = nn.Linear(FLAGS.model_size, num_outs)

        self.has_aux_out = num_aux_outs is not None
        if self.has_aux_out:
            self.w_aux = nn.Linear(FLAGS.model_size, num_aux_outs)

    def _decoder(self, x):
        x = x.transpose(0, 1)  
        x = self.transformer(x)
        x = x.transpose(0, 1)
        x = self.w_out(x)

        return F.log_softmax(x, 2)
   
   def _conv(x_raw, conv_block, linear_layer):
        x_raw = x_raw.transpose(1,2) # put channel before time for conv
        x_raw = conv_block(x_raw)
        x_raw = x_raw.transpose(1,2)
        x_raw = linear_layer(x_raw)

        return x_raw

   def forward(self, emg_x, emg_voiced_parallel_x, audio_x, session_ids):
        # emg_x shape is (batch, time, electrode)
        # audio_x is (B, T, C) Mel-specto
        # What's session for?


        # EMG
        if self.training:
            r = random.randrange(8)
            if r > 0:
                emg_x[:,:-r,:] = emg_x[:,r:,:] # shift left r
                emg_x[:,-r:,:] = 0

                emg_voiced_parallel_x[:,:-r,:] = emg_voiced_parallel_x[:,r:,:] # shift left r
                emg_voiced_parallel_x[:,-r:,:] = 0

        emg_latent = self._conv(emg_x,  self.emg_conv_blocks, self.emg_linear)
        emg_pred = self._decoder(emg_latent)

        emg_parallel_latent= self._conv(emg_voiced_parallel_x, self.emg_conv_blocks, self.emg_linear)

        # Audio
        audio_latent = self._conv(audio_x,  self.audio_conv_blocks, self.audio_linear)
        audio_pred = self._decoder(audio_latent)

        return emg_pred, emg_latent, emg_parallel_latent, audio_pred, audio_latent
