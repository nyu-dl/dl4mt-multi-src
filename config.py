from collections import OrderedDict

from mcg.utils import (get_enc_dec_ids, get_enc_dec_ids_mSrc, get_paths,
                       get_odict, get_odict_pair, get_val_set_outs,
                       ReadOnlyDict, is_multiSource, p_)

import data_paths as paths


def prototype_config_multiCG_08(cgs):

    enc_ids, dec_ids = get_enc_dec_ids(cgs)

    # Model related
    config = {}
    config['cgs'] = cgs
    config['num_encs'] = len(enc_ids)
    config['num_decs'] = len(dec_ids)
    config['seq_len'] = 50
    config['representation_dim'] = 1200  # joint annotation dimension
    config['enc_nhids'] = get_odict(enc_ids, 1000)
    config['dec_nhids'] = get_odict(dec_ids, 1000)
    config['enc_embed_sizes'] = get_odict(enc_ids, 620)
    config['dec_embed_sizes'] = get_odict(dec_ids, 620)

    # Additional options for the model
    config['take_last'] = True
    config['multi_latent'] = True
    config['readout_dim'] = 1000
    config['representation_act'] = 'linear'  # encoder representation act
    config['lencoder_act'] = 'tanh'  # att-encoder latent space act
    config['ldecoder_act'] = 'tanh'  # att-decoder latent space act
    config['lctxproj_act'] = 'tanh'  # post context projection act
    config['dec_rnn_type'] = 'gru_cond_mCG'
    config['finit_mid_dim'] = 600
    config['finit_code_dim'] = 500
    config['finit_act'] = 'tanh'
    config['att_dim'] = 1200

    # Optimization related
    config['batch_sizes'] = get_odict(cgs, 60)
    config['sort_k_batches'] = 12
    config['step_rule'] = 'uAdam'
    config['learning_rate'] = 2e-4
    config['step_clipping'] = 1
    config['weight_scale'] = 0.01
    config['schedule'] = get_odict(cgs, 1)
    config['save_accumulators'] = True  # algorithms' update step variables
    config['load_accumulators'] = True  # be careful with this
    config['exclude_encs'] = get_odict(enc_ids, False)
    config['min_seq_lens'] = get_odict(cgs, 0)
    config['additional_excludes'] = get_odict(cgs, [])

    # Regularization related
    config['drop_input'] = get_odict(cgs, 0.)
    config['decay_c'] = get_odict(cgs, 0.)
    config['alpha_c'] = get_odict(cgs, 0.)
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary related
    config['src_vocab_sizes'] = get_odict(enc_ids, 30000)
    config['trg_vocab_sizes'] = get_odict(dec_ids, 30000)
    config['src_eos_idxs'] = get_odict(enc_ids, 0)
    config['trg_eos_idxs'] = get_odict(dec_ids, 0)
    config['stream'] = 'multiCG_stream'
    config['unk_id'] = 1

    # Early stopping based on bleu related
    config['normalized_bleu'] = True
    config['track_n_models'] = 3
    config['output_val_set'] = True
    config['beam_size'] = 12

    # Validation set for log probs related
    config['log_prob_freq'] = 2000
    config['log_prob_bs'] = 10

    # Timing related
    config['reload'] = True
    config['save_freq'] = 10000
    config['sampling_freq'] = 17
    config['bleu_val_freq'] = 10000000
    config['val_burn_in'] = 1
    config['finish_after'] = 2000000
    config['incremental_dump'] = True

    # Monitoring related
    config['hook_samples'] = 2
    config['plot'] = False
    config['bokeh_port'] = 3333

    return config


def get_config_single():

    cgs = ['de_en']
    config = prototype_config_multiCG_08(cgs)
    enc_ids, dec_ids = get_enc_dec_ids(cgs)
    config['saveto'] = 'single'

    basedir = ''
    config['batch_sizes'] = OrderedDict([('de_en', 80)])
    config['schedule'] = OrderedDict([('de_en', 12)])
    config['src_vocabs'] = get_paths(enc_ids, paths.src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, paths.trg_vocabs, basedir)
    config['src_datas'] = get_paths(cgs, paths.src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, paths.trg_datas, basedir)
    config['save_freq'] = 5000
    config['val_burn_in'] = 60000
    config['bleu_script'] = basedir + '/multi-bleu.perl'
    config['val_sets'] = get_paths(cgs, paths.val_sets_src, basedir)
    config['val_set_grndtruths'] = get_paths(cgs, paths.val_sets_ref, basedir)
    config['val_set_outs'] = get_val_set_outs(config['cgs'], config['saveto'])
    config['log_prob_sets'] = get_paths(cgs, paths.log_prob_sets, basedir)

    return ReadOnlyDict(config)


def prototype_config_mSrc(cgs):

    enc_ids, dec_ids = get_enc_dec_ids_mSrc(cgs)
    enc_ids_mSrc = [p_(x)[0] for x in cgs if is_multiSource(x)]

    # Model related
    config = {}
    config['cgs'] = cgs
    config['num_encs'] = len(enc_ids)
    config['num_decs'] = len(dec_ids)
    config['seq_len'] = 50
    config['representation_dim'] = 1200  # joint annotation dimension
    config['enc_nhids'] = get_odict(enc_ids, 1000)
    config['dec_nhids'] = get_odict(dec_ids, 1000)
    config['enc_embed_sizes'] = get_odict(enc_ids, 620)
    config['dec_embed_sizes'] = get_odict(dec_ids, 620)

    # Additional options for the model
    config['take_last'] = True
    config['multi_latent'] = True
    config['readout_dim'] = 1000
    config['representation_act'] = 'linear'  # encoder representation act
    config['lencoder_act'] = 'tanh'  # att-encoder latent space act
    config['ldecoder_act'] = 'tanh'  # att-decoder latent space act
    config['lctxproj_act'] = 'tanh'  # post context projection act
    config['dec_rnn_type'] = 'gru_cond_mCG'
    config['finit_mid_dim'] = 600
    config['finit_code_dim'] = 500
    config['finit_act'] = 'tanh'
    config['att_dim'] = 1200

    # specific to multiSource
    config['init_merge_op'] = 'mean'  # initializer mlp merger op
    config['attend_merge_op'] = 'mean'  # post attention merger op
    config['num_encs'] = len(enc_ids)
    config['num_decs'] = len(dec_ids)
    config['init_merge_act'] = 'linear'  # activation after init merge
    config['attend_merge_act'] = 'linear'  # activation after attend merge

    # Optimization related
    config['batch_sizes'] = get_odict(cgs, 60)
    config['sort_k_batches'] = 12
    config['step_rule'] = 'uAdam'
    config['learning_rate'] = 2e-4
    config['step_clipping'] = 1
    config['weight_scale'] = 0.01
    config['schedule'] = get_odict(cgs, 1)
    config['save_accumulators'] = True  # algorithms' update step variables
    config['load_accumulators'] = True  # be careful with this
    config['exclude_encs'] = get_odict(enc_ids, False)
    config['min_seq_lens'] = get_odict(cgs, 0)
    config['additional_excludes'] = get_odict(cgs, [])

    # Regularization related
    config['drop_input'] = get_odict(cgs, 0.)
    config['decay_c'] = get_odict(cgs, 0.)
    config['alpha_c'] = get_odict(cgs, 0.)
    config['weight_noise_ff'] = False
    config['weight_noise_rec'] = False
    config['dropout'] = 1.0

    # Vocabulary related
    config['src_vocab_sizes'] = get_odict(enc_ids, 30000)
    config['trg_vocab_sizes'] = get_odict(dec_ids, 30000)
    config['trg_eos_idxs'] = get_odict(dec_ids, 0)
    config['stream'] = 'multiCG_stream'
    config['unk_id'] = 1

    config['src_eos_idxs'] = get_odict(enc_ids + enc_ids_mSrc, 0)

    # Validation set for log probs related
    config['log_prob_freq'] = 2000
    config['log_prob_bs'] = 10

    # Timing related
    config['reload'] = True
    config['save_freq'] = 10000
    config['sampling_freq'] = 17
    config['val_burn_in'] = 1
    config['finish_after'] = 2000000
    config['incremental_dump'] = True

    # Monitoring related
    config['hook_samples'] = 2
    config['plot'] = False
    config['bokeh_port'] = 3333

    return config


def get_config_EsFr2En_single():
    cgs = ['es_en', 'fr_en']

    enc_ids, dec_ids = get_enc_dec_ids_mSrc(cgs)

    # Model related
    config = prototype_config_mSrc(cgs)
    config['saveto'] = 'esfr2en_single'
    config['batch_sizes'] = get_odict(cgs, 80)

    # Convenience basedirectory
    basedir = 'dl4mt-multi-src/data'

    # Vocabulary/dataset related
    config['src_vocabs'] = get_paths(enc_ids, paths.src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, paths.trg_vocabs, basedir)
    config['src_vocab_sizes'] = get_odict_pair(enc_ids, [20624, 20335])
    config['trg_vocab_sizes'] = get_odict(dec_ids, 20212)

    # Dataset related
    config['src_datas'] = get_paths(cgs, paths.src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, paths.trg_datas, basedir)

    # Early stopping based on bleu related
    config['save_freq'] = 5000
    config['val_burn_in'] = 1

    # Validation set for log probs related
    config['log_prob_sets'] = get_paths(cgs, paths.log_prob_sets, basedir)

    return ReadOnlyDict(config)


def get_config_EsFr2En_mSrc():
    cgs = ['es.fr_en']

    enc_ids, dec_ids = get_enc_dec_ids_mSrc(cgs)

    # Model related
    config = prototype_config_mSrc(cgs)
    config['saveto'] = 'esfr2en_mSrc'
    config['batch_sizes'] = get_odict(cgs, 80)

    # Convenience basedirectory
    basedir = 'dl4mt-multi-src/data'

    # Vocabulary/dataset related
    config['src_vocabs'] = get_paths(enc_ids, paths.src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, paths.trg_vocabs, basedir)
    config['src_vocab_sizes'] = get_odict_pair(enc_ids, [20624, 20335])
    config['trg_vocab_sizes'] = get_odict(dec_ids, 20212)

    # Dataset related
    config['src_datas'] = get_paths(cgs, paths.src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, paths.trg_datas, basedir)

    # Early stopping based on bleu related
    config['save_freq'] = 5000
    config['val_burn_in'] = 1

    # Validation set for log probs related
    config['log_prob_sets'] = get_paths(cgs, paths.log_prob_sets, basedir)

    # specific to multiSource
    config['init_merge_op'] = 'mean'
    config['attend_merge_op'] = 'mean'
    config['init_merge_act'] = 'tanh'
    config['attend_merge_act'] = 'tanh'

    return ReadOnlyDict(config)


def get_config_EsFr2En_single_and_mSrc():
    cgs = ['es_en', 'fr_en', 'es.fr_en']

    enc_ids, dec_ids = get_enc_dec_ids_mSrc(cgs)

    # Model related
    config = prototype_config_mSrc(cgs)
    config['saveto'] = 'esfr2en_single_and_mSrc'
    config['batch_sizes'] = get_odict(cgs, 80)

    # Convenience basedirectory
    basedir = 'dl4mt-multi-src/data'

    # Vocabulary/dataset related
    config['src_vocabs'] = get_paths(enc_ids, paths.src_vocabs, basedir)
    config['trg_vocabs'] = get_paths(dec_ids, paths.trg_vocabs, basedir)
    config['src_vocab_sizes'] = get_odict_pair(enc_ids, [20624, 20335])
    config['trg_vocab_sizes'] = get_odict(dec_ids, 20212)

    # Dataset related
    config['src_datas'] = get_paths(cgs, paths.src_datas, basedir)
    config['trg_datas'] = get_paths(cgs, paths.trg_datas, basedir)

    # Early stopping based on bleu related
    config['save_freq'] = 5000
    config['val_burn_in'] = 1

    # Validation set for log probs related
    config['log_prob_sets'] = get_paths(cgs, paths.log_prob_sets, basedir)

    # specific to multiSource
    config['init_merge_op'] = 'mean'
    config['attend_merge_op'] = 'mean'
    config['init_merge_act'] = 'tanh'
    config['attend_merge_act'] = 'tanh'

    return ReadOnlyDict(config)
