# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

from dataset.consep.dataset import MORPHOLOGICAL_FEATURE_SIZE

ENABLE_DOUBLE_KEY_STORE = False


def projection_head_generator(in_features, layers_size, normalization):
    modules = []
    last_dim = in_features
    for size in layers_size[:-1]:
        modules.extend([
            normalization(nn.Linear(last_dim, size)),
            nn.BatchNorm1d(size),
            nn.ReLU(),
        ])
        last_dim = size
    modules.append(normalization(nn.Linear(layers_size[-2], layers_size[-1])))
    return nn.Sequential(*modules)


class MaskedEnvMoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """

    def __init__(self, base_encoder, env_encoder, dim=128, m=0.999, mlp=None, prediction_head=None, mlp_embedding=False,
                 spectral_normalization=False, queue_size=0, shared_encoder=False, teacher=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        mlp: tuple of hidden sizes (default: None)
        """
        super(MaskedEnvMoCo, self).__init__()

        self.m = m
        self.queue_size = queue_size
        self.mlp_embedding = mlp_embedding
        self.shared_encoder = shared_encoder
        self.teacher = teacher
        normalization = (lambda x: x) if not spectral_normalization else nn.utils.spectral_norm

        # create the encoders
        # num_classes is the output fc dimension
        self.encoder_q = base_encoder(num_classes=dim)
        self.encoder_k = base_encoder(num_classes=dim)

        self.q_projection_head, self.k_projection_head = None, None
        if mlp:  # hack: brute-force replacement #
            # replace the fully connected layer with identity
            dim_mlp = self.encoder_q.fc.in_features
            self.encoder_q.fc = nn.Identity()
            self.encoder_k.fc = nn.Identity()
            self.q_projection_head = projection_head_generator(dim_mlp, mlp + [dim], normalization)
            self.k_projection_head = projection_head_generator(dim_mlp, mlp + [dim], normalization)

        self.q_prediction_head = None
        if prediction_head:
            self.q_prediction_head = nn.Sequential(
                normalization(nn.Linear(dim, prediction_head)),
                nn.BatchNorm1d(prediction_head),
                nn.ReLU(),
                normalization(nn.Linear(prediction_head, dim)),
            )

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        if self.q_projection_head is not None and self.k_projection_head is not None:
            for param_q, param_k in zip(self.q_projection_head.parameters(), self.k_projection_head.parameters()):
                param_k.data.copy_(param_q.data)  # initialize
                param_k.requires_grad = False  # not update by gradient

        if not self.shared_encoder:
            self.env_encoder = env_encoder(num_classes=dim)
            env_dim = self.env_encoder.fc.in_features
            self.env_encoder.fc = nn.Identity()
        else:
            self.env_encoder = self.encoder_q
            env_dim = dim_mlp

        self.env_projection_head = projection_head_generator(env_dim, mlp + [dim], normalization)
        self.q_env_projection_head = projection_head_generator(dim_mlp, mlp + [dim], normalization)
        self.q_env_prediction_head = nn.Sequential(
            normalization(nn.Linear(dim, prediction_head)),
            nn.BatchNorm1d(prediction_head),
            nn.ReLU(),
            normalization(nn.Linear(prediction_head, dim)),
        )

        if self.queue_size != 0:
            self.register_buffer("queue", torch.randn(dim, self.queue_size))
            self.queue = nn.functional.normalize(self.queue, dim=0)
            self.register_buffer("extra_feat_queue", torch.randn(MORPHOLOGICAL_FEATURE_SIZE, self.queue_size))
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

        for param_q, param_k in zip(self.q_projection_head.parameters(), self.k_projection_head.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    def embedding(self, x1):
        # set values
        encoder = self.encoder_q
        projection_head = self.q_projection_head
        if self.teacher:
            encoder = self.encoder_k
            projection_head = self.k_projection_head
        # run embedding
        embedding = encoder(x1)
        if self.mlp_embedding and projection_head is not None:
            embedding = projection_head(embedding)
        return embedding

    def forward(self, x1, x2=None, patch1=None, patch2=None, extra_feat1=None, extra_feat2=None, patch_meta_data=None,
                return_patch=False, normalize_embedding=False):
        """
        Input:
            x1: a batch of query images
            x2: a batch of key images (default: None)
        Output:
            q1,q2,k1,k2 if x2 is not None, embedding of x1 otherwise
        """
        # return embedding of x1 if x2 is not given
        if x2 is None:
            normalizer = lambda e: nn.functional.normalize(e, dim=1) if normalize_embedding else e
            embedding = self.embedding(x1)
            if return_patch:
                assert patch1 is not None
                patch_embedding = self.env_encoder(patch1)
                return normalizer(embedding), normalizer(patch_embedding)
            return normalizer(embedding)

        # compute query features
        q1, q2 = self.encoder_q(x1), self.encoder_q(x2)
        q_env_1, q_env_2 = self.q_env_prediction_head(self.q_env_projection_head(q1)), self.q_env_prediction_head(self.q_env_projection_head(q2))
        q_env_1, q_env_2 = nn.functional.normalize(q_env_1, dim=1), nn.functional.normalize(q_env_2, dim=1)
        if self.q_projection_head is not None:
            q1, q2 = self.q_projection_head(q1), self.q_projection_head(q2)
        if self.q_prediction_head is not None:
            q1, q2 = self.q_prediction_head(q1), self.q_prediction_head(q2)

        q1 = nn.functional.normalize(q1, dim=1)
        q2 = nn.functional.normalize(q2, dim=1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            k1, k2 = self.encoder_k(x1), self.encoder_k(x2)
            if self.k_projection_head is not None:
                k1, k2 = self.k_projection_head(k1), self.k_projection_head(k2)
            k1 = nn.functional.normalize(k1, dim=1)
            k2 = nn.functional.normalize(k2, dim=1)

            queue = None
            if self.queue_size != 0:
                queue = self.queue.clone()
                extra_feat_queue = self.extra_feat_queue.clone()
                self._dequeue_and_enqueue(k1, k2, extra_feat1, extra_feat2)

            # add queue to negatives
            if queue is not None:
                k1 = torch.cat([k1, queue.t()], dim=0)
                k2 = torch.cat([k2, queue.t()], dim=0)
                extra_feat1 = torch.cat([extra_feat1, extra_feat_queue.t()], dim=0)
                extra_feat2 = torch.cat([extra_feat2, extra_feat_queue.t()], dim=0)

        assert patch1 is not None
        env = self.env_encoder(patch1)
        env = self.env_projection_head(env)
        env = nn.functional.normalize(env, dim=1)

        return (q1, q2), (k1, k2), (q_env_1, q_env_2, env), (None, None), (extra_feat1, extra_feat2), None, None, None  # quantization loss is none

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2, extra_feat1, extra_feat2):
        if torch.distributed.is_initialized():
            # gather keys before updating queue
            keys1 = concat_all_gather(keys1)
            keys2 = concat_all_gather(keys2)
            extra_feat1 = concat_all_gather(extra_feat1)
            extra_feat2 = concat_all_gather(extra_feat2)

        if ENABLE_DOUBLE_KEY_STORE:
            keys1 = torch.cat([keys1, keys2], dim=0)
            extra_feat1 = torch.cat([extra_feat1, extra_feat2], dim=0)

        batch_size = keys1.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys1.T
        self.extra_feat_queue[:, ptr:ptr + batch_size] = extra_feat1.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
