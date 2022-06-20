from text.symbols import symbols
import pprint
import ast


class HParams(object):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        return pprint.pformat(self.__dict__)

    def parse(self, string):
        # Overrides hparams from a comma-separated string of name=value pairs
        if len(string) > 0:
            overrides = [s.split("=") for s in string.split(",")]
            keys, values = zip(*overrides)
            keys = list(map(str.strip, keys))
            values = list(map(str.strip, values))
            for k in keys:
                self.__dict__[k] = ast.literal_eval(values[keys.index(k)])
        return self

    def values(self):
        return pprint.pformat(self.__dict__)


def create_hparams(hparams_string=None, verbose=False):
    """Create model hyperparameters. Parse nondefault from given string."""

    hparams = HParams(
        #
        # Experiment Parameters        #
        #
        epochs=500,
        iters_per_checkpoint=1000,
        seed=1234,
        dynamic_loss_scaling=True,
        fp16_run=False,
        distributed_run=False,
        dist_backend="nccl",
        dist_url="tcp://localhost:54321",
        cudnn_enabled=True,
        cudnn_benchmark=False,
        ignore_layers=['embedding.weight', 'decoder.prenet', 'decoder.linear_projection'],
        #
        # Data Parameters             #
        #
        load_mel_from_disk=False,
        audio_dtype='np.int16',  # Data type of input audio files. If not 'np.int16' ; will be
        # converted to it.
        use_librosa=True,  # If you want to use librosa for loading file and automatically
        # resampling to sampling_rate
        training_files='ultimate_lang_dataset/ds_manifest_train.txt',
        validation_files='ultimate_lang_dataset/ds_manifest_val.txt',
        text_cleaners=['basic_cleaners'],

        #
        # Audio Parameters             #
        #
        max_wav_value=32768.0,
        sampling_rate=22050,
        filter_length=1024,
        hop_length=256,
        win_length=1024,
        n_mel_channels=80,
        mel_fmin=0.0,
        mel_fmax=8000.0,

        #
        # Model Parameters             #
        #
        n_symbols=len(symbols),
        symbols_embedding_dim=256,

        # Encoder parameters
        encoder_kernel_size=5,
        encoder_n_convolutions=3,
        encoder_embedding_dim=256,  # same as symbols_embedding_dim

        # Decoder parameters
        n_frames_per_step=1,  # More than 1 is supported now
        decoder_rnn_dim=256,
        prenet_dim=256,
        max_decoder_steps=1000,
        gate_threshold=0.5,
        p_attention_dropout=0.1,
        p_decoder_dropout=0.1,

        # Attention parameters
        attention_rnn_dim=384,
        attention_dim=128,

        # Location Layer parameters
        attention_location_n_filters=32,
        attention_location_kernel_size=31,

        # Mel-post processing network parameters
        postnet_embedding_dim=384,
        postnet_kernel_size=5,
        postnet_n_convolutions=5,

        #
        # Optimization Hyperparameters
        #
        use_saved_learning_rate=False,
        learning_rate=1e-3,
        anneal=0,  # number of iterations to anneal lr from 0 to 'learning_rate'
        weight_decay=1e-6,
        grad_clip_thresh=1.0,
        batch_size=2,
        mask_padding=True,  # set model's padded outputs to padded values

        #
        # Speaker and Lang Embeddings #
        #
        speaker_embedding_dim=16,
        lang_embedding_dim=3,
        n_langs=2,
        n_speakers=7,

        #
        # Speaker Classifier Params #
        #
        hidden_sc_dim=256,

        #
        # Residual Encoder Params  #
        #
        residual_encoding_dim=32,  # 16 for q(z_l|X) and 16 for q(z_o|X)
        dim_yo=7,  # (==n_speakers) dim(y_{o})
        dim_yl=10,  # K
        mcn=8  # n for monte carlo sampling of q(z_l|X)and q(z_o|X)
    )

    if hparams_string:
        print('Parsing command line hparams: %s', hparams_string)
        hparams.parse(hparams_string)

    if verbose:
        print('Final parsed hparams: %s', hparams.values())

    return hparams
