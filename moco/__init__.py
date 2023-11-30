import backbones as models
import moco.maskedenvmoco.builder
import moco.mocov3.builder


def build(config):
    moco_type = config['moco_type'].lower()
    if moco_type == 'v3':
        model = moco.mocov3.builder.MoCoV3(
            models.__dict__[config['arch']],
            config['moco_dim'], config['moco_m'], config['mlp'], config['prediction_head'], config['mlp_embedding'],
            config['spectral_norm'], config["queue_size"], config['teacher'])
    elif moco_type == 'vq':
        assert config['embedding_size'] != 0
        model = moco.vqmoco.builder.VQMoCo(
            models.__dict__[config['arch']],
            config['moco_dim'], config['moco_m'], config['mlp'], config['prediction_head'], config['mlp_embedding'],
            config['spectral_norm'], config["queue_size"], config['embedding_size'], config['commitment_cost'], config['teacher'])
    elif moco_type == 'env':
        model = moco.envmoco.builder.EnvMoCo(
            models.__dict__[config['arch']],
            models.__dict__[config['env_arch']],
            config['moco_dim'], config['moco_m'], config['mlp'], config['prediction_head'], config['mlp_embedding'],
            config['spectral_norm'], config["queue_size"], config['shared_encoder'], config['teacher'])
    elif moco_type == 'maskedenv':
        model = moco.maskedenvmoco.builder.MaskedEnvMoCo(
            models.__dict__[config['arch']],
            models.__dict__[config['env_arch']],
            config['moco_dim'], config['moco_m'], config['mlp'], config['prediction_head'], config['mlp_embedding'],
            config['spectral_norm'], config["queue_size"], config['shared_encoder'], config['teacher'])
    elif moco_type == 'maskedenvmorpho':
        model = moco.maskedenvmorphomoco.builder.MaskedEnvMorphoMoCo(
            models.__dict__[config['arch']],
            models.__dict__[config['env_arch']],
            config['moco_dim'], config['moco_m'], config['mlp'], config['prediction_head'], config['mlp_embedding'],
            config['spectral_norm'], config["queue_size"], config['shared_encoder'], config['morphological_layers'], config['teacher'])
    elif moco_type == 'simclr':
        model = moco.simclr.builder.SimCLR(models.__dict__[config['arch']],
                                           config['moco_dim'], config['moco_m'], config['mlp'],
                                           config['prediction_head'], config['mlp_embedding'],
                                           config['spectral_norm'], config["queue_size"])
    else:
        raise ValueError('invalid MoCo type %s' % moco_type)
    return model
