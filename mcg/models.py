import logging
import numpy
import os
import pickle
import theano

from collections import Counter, OrderedDict
from theano import tensor
from toolz import merge

from blocks.graph import ComputationGraph

from .layers import get_layer, linear, tanh, relu
from .utils import (norm_weight, _p, p_, concatenate, init_tparams,
                    get_enc_dec_ids, get_param_idx, is_multiSource,
                    get_enc_ids, get_subdict)

logger = logging.getLogger(__name__)


class MultiSourceEncoder(object):
    """
    Encoder wrapper that computes multiple encoder annotations.
    --------------------------------------------------------
    This class is a wrapper which calls build and sample
    functions of all the given encoders ITERATIVELY, so expect
    increased computational time as these operations are done
    sequentially. Supports any encoder as long as they implement
    build_model and build_sampling_model methods.
    """

    def __init__(self, enc_id, encoders, **kwargs):
        """
        enc_id : str, name of the encoder(s) separated by '.' eg. es.fr
        encoders : dict, {encoder_name: encoder_instance} dictionary
        """
        self.enc_id = enc_id
        self.enc_ids = get_enc_ids(enc_id)
        self.encoders = encoders
        self.n_encs = len(encoders)
        self.params = OrderedDict()
        self.tparams = OrderedDict()

    def init_params(self):
        for i, eid in enumerate(self.enc_ids):
            self.params.update(self.encoders[eid].params)
            self.tparams.update(self.encoders[eid].tparams)

    def build_model(self, xs, x_masks, **kwargs):
        """
        xs : list of theano tensor variables
        x_masks: list of thenao tensor variables
        """
        logger.info(
            " ... MultiSourceEncoder [{}] building training models"
            .format(self.enc_id))
        ctxs = OrderedDict()
        for i, eid in enumerate(self.enc_ids):
            ctxs[eid] = self.encoders[eid].build_model(
                xs[i], x_masks[i], **kwargs)
        return ctxs

    def build_sampling_model(self, xs, **kwargs):
        """
        xs : list of theano tensor variables
        """
        logger.info(
            " ... MultiSourceEncoder [{}] building sampling models"
            .format(self.enc_id))
        ctxs = OrderedDict()
        for i, eid in enumerate(self.enc_ids):
            ctxs[eid] = self.encoders[eid].build_sampling_model(
                xs[i], **kwargs)
        return ctxs

    def get_params(self):
        return self.tparams


class BidirectionalEncoder(object):
    """
    Two recurrent neural networks stitched together
    --------------------------------------------------------
    Wraps two recurrent neural networks, of type rnn_type reading
    input in both directions, using the same word embeddings.
    """

    def __init__(self, n_words, dim_word, dim, rnn_type='gru', enc_id=0,
                 enc_layer_norm=False, **kwargs):
        """
        n_words : vocabulary size
        dim_word : embedding dimension
        dim : number of hidden units in each of the rnns
        rnn_type : layer name to be used by get_layer
        enc_id : string indicating the encoder name
        enc_layer_norm : whether to apply layer normalization
        """
        self.n_words = n_words
        self.dim_word = dim_word
        self.dim = dim
        self.rnn_type = rnn_type
        self.enc_id = enc_id
        self.enc_layer_norm = enc_layer_norm
        self.params = []

    def init_params(self):
        enc_id = self.enc_id
        params = OrderedDict()
        params['Wemb_%s' % self.enc_id] = norm_weight(self.n_words,
                                                      self.dim_word)
        enc_rnn_type = ('ln' + self.rnn_type if self.enc_layer_norm
                        else self.rnn_type)

        params = get_layer(enc_rnn_type)[0](params,
                                            prefix='encoder_%s' % enc_id,
                                            nin=self.dim_word,
                                            dim=self.dim)
        params = get_layer(enc_rnn_type)[0](params,
                                            prefix='encoder_r_%s' % enc_id,
                                            nin=self.dim_word,
                                            dim=self.dim)
        self.params = params
        self.tparams = init_tparams(self.params)

    def build_model(self, x, x_mask, **kwargs):
        """
        x : theano tensor variable
        x_mask : theano tensor variable
        """
        enc_id = self.enc_id
        logger.info(
            " ... BidirectionalEncoder [{}] building training models"
            .format(enc_id))

        xr = x[::-1]
        xr_mask = x_mask[::-1]

        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        enc_rnn_type = ('ln' + self.rnn_type if self.enc_layer_norm
                        else self.rnn_type)
        logger.info(" ... ... using [{}] layer".format(enc_rnn_type))

        emb = self.tparams['Wemb_%s' % self.enc_id][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        proj = get_layer(enc_rnn_type)[1](self.tparams, emb,
                                          prefix='encoder_%s' % enc_id,
                                          mask=x_mask, **kwargs)
        embr = self.tparams['Wemb_%s' % self.enc_id][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.dim_word])
        projr = get_layer(enc_rnn_type)[1](self.tparams, embr,
                                           prefix='encoder_r_%s' % enc_id,
                                           mask=xr_mask, **kwargs)
        ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        return ctx

    def build_sampling_model(self, x, **kwargs):
        """
        x : theano tensor variable
        """
        enc_id = self.enc_id
        logger.info(
            " ... BidirectionalEncoder [{}] building sampling models"
            .format(enc_id))

        xr = x[::-1]
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]

        enc_rnn_type = ('ln' + self.rnn_type if self.enc_layer_norm
                        else self.rnn_type)
        logger.info(" ... ... using [{}] layer".format(enc_rnn_type))

        emb = self.tparams['Wemb_%s' % self.enc_id][x.flatten()]
        emb = emb.reshape([n_timesteps, n_samples, self.dim_word])
        proj = get_layer(enc_rnn_type)[1](self.tparams, emb,
                                          prefix='encoder_%s' % enc_id,
                                          **kwargs)
        embr = self.tparams['Wemb_%s' % self.enc_id][xr.flatten()]
        embr = embr.reshape([n_timesteps, n_samples, self.dim_word])
        projr = get_layer(enc_rnn_type)[1](self.tparams, embr,
                                           prefix='encoder_r_%s' % enc_id,
                                           **kwargs)
        ctx = concatenate([proj[0], projr[0][::-1]], axis=proj[0].ndim-1)
        return ctx

    def get_params(self):
        return self.tparams


class MultiEncoder(object):
    """
    Highest level encoder in the multi-way hierarchy
    --------------------------------------------------------
    A multi-encoder contains multiple encoders of type, MultiSourceEncoder,
    BidirectionalEncoder or any other encoder class that implements
    build_model, build_sampling_model and init_params methods.
    """

    def __init__(self, enc_ids, src_vocab_sizes, enc_embed_sizes,
                 enc_nhids, representation_dim,
                 representation_act='lambda x: x',
                 multi_latent=False, enc_rnn_type='gru', dropout_ctx=0.0,
                 share_encoder=None, **kwargs):
        """
        enc_ids : list of encoder ids, eg. ['es', 'en']
        src_vocab_sizes : dict, mapping enc_id to vocabulary size
        enc_embed_sizes : dict, mapping enc_id to embedding size
        enc_nhids : dict, mapping enc_id to number of hidden units
        representation_dim : dimension of joint encoder space (W_adp)
        representation_act : activation after W_adp
        multi_latent : Legacy, should be True always
        enc_rnn_type : layer name to be used by get_layer
        dropout_ctx : dropout for context embedder
        share_encoder : dictionary for sharing one encoder
        """
        self.enc_ids = enc_ids
        self.src_vocab_sizes = src_vocab_sizes
        self.enc_embed_sizes = enc_embed_sizes
        self.enc_nhids = enc_nhids
        self.representation_dim = representation_dim
        self.representation_act = representation_act
        self.multi_latent = multi_latent
        self.enc_rnn_type = enc_rnn_type
        self.dropout_ctx = dropout_ctx
        self.share_encoder = share_encoder
        self.num_encs = len(enc_ids)  # TODO: check this

        # first initialize regular encoders
        self.encoders = OrderedDict()
        for eid in self.enc_ids:
            if not is_multiSource(eid):
                self.encoders[eid] = BidirectionalEncoder(
                    src_vocab_sizes[eid], enc_embed_sizes[eid], enc_nhids[eid],
                    enc_id=eid, rnn_type=enc_rnn_type, **kwargs)

        # then initialize multisource encoder using individual ones
        for eid in self.enc_ids:
            if is_multiSource(eid):
                eids_this = get_enc_ids(eid)

                # we might have only multi-source cg
                for eid_this in eids_this:
                    if eid_this not in self.encoders:
                        self.encoders[eid_this] = BidirectionalEncoder(
                            src_vocab_sizes[eid_this],
                            enc_embed_sizes[eid_this], enc_nhids[eid_this],
                            enc_id=eid_this, rnn_type=enc_rnn_type, **kwargs)

                self.encoders[eid] = MultiSourceEncoder(
                    encoders=get_subdict(self.encoders, eids_this),
                    enc_id=eid, **kwargs)

    def init_params(self):

        self.params = OrderedDict()
        self.tparams = OrderedDict()
        self.shared_params_map = OrderedDict()

        allocated_encs = []
        initialized_encs = []

        # initialize individual encoders first if single pair cgs exist
        for eid in self.enc_ids:
            if not is_multiSource(eid):
                allocated_encs.append(eid)
                self.params = get_layer('ff')[0](
                    self.params, prefix=_p('ctx_embedder', eid),
                    nin=2 * self.enc_nhids[eid], nout=self.representation_dim,
                    add_bias=True)

        # if we only have multi-source cgs we need to do it again
        for eid in self.enc_ids:
            if is_multiSource(eid):
                eids_this = get_enc_ids(eid)
                for eid_this in eids_this:
                    if eid_this not in allocated_encs:
                        allocated_encs.append(eid_this)
                        self.params = get_layer('ff')[0](
                            self.params, prefix=_p('ctx_embedder', eid_this),
                            nin=2 * self.enc_nhids[eid_this],
                            nout=self.representation_dim, add_bias=True)

        self.tparams = init_tparams(self.params)

        # first initialize individual encoders
        for eid in self.enc_ids:
            if not is_multiSource(eid):
                initialized_encs.append(eid)
                self.encoders[eid].init_params()
                self.params.update(self.encoders[eid].params)
                self.tparams.update(self.encoders[eid].tparams)

        # then initialize multisource models
        for eid in self.enc_ids:
            if is_multiSource(eid):
                eids_this = get_enc_ids(eid)
                for eid_this in eids_this:
                    if eid_this not in initialized_encs:
                        initialized_encs.append(eid_this)
                        self.encoders[eid_this].init_params()
                        self.params.update(self.encoders[eid_this].params)
                        self.tparams.update(self.encoders[eid_this].tparams)

                self.encoders[eid].init_params()
                self.params.update(self.encoders[eid].params)
                self.tparams.update(self.encoders[eid].tparams)

        if self.share_encoder:
            for ref_enc_name, target_enc_names in self.share_encoder.items():

                # fetch reference encoder parameters
                prefix = _p('ctx_embedder', ref_enc_name)
                ref_enc_params = self.encoders[ref_enc_name].tparams
                ref_ctx_emb_W = self.tparams[_p(prefix, 'W')]
                ref_ctx_emb_b = self.tparams[_p(prefix, 'b')]

                for ename in target_enc_names:
                    enc = self.encoders[ename]

                    # first set encoder specific parameters
                    for pname in enc.tparams.keys():
                        ref_pname = pname.replace(
                            '_%s' % ename, _p('', ref_enc_name))
                        if ref_pname in ref_enc_params:
                            logger.info(
                                '...encoder [{}] using [{}] for [{}]'.format(
                                    ename, ref_pname, pname))
                            enc.tparams[pname] = ref_enc_params[ref_pname]
                            self.shared_params_map[pname] = ref_pname

                    # then set ctx embedder specific parameters
                    trg_ctx_emb_Wname = _p(_p('ctx_embedder', ename), 'W')
                    trg_ctx_emb_bname = _p(_p('ctx_embedder', ename), 'b')

                    self.tparams[trg_ctx_emb_Wname] = ref_ctx_emb_W
                    self.tparams[trg_ctx_emb_bname] = ref_ctx_emb_b
                    self.shared_params_map[trg_ctx_emb_Wname] = ref_ctx_emb_W.name
                    self.shared_params_map[trg_ctx_emb_bname] = ref_ctx_emb_b.name

                    logger.info(
                        '...encoder [{}] using [{}] for [{}]'.format(
                            ename, ref_ctx_emb_W.name, trg_ctx_emb_Wname))
                    logger.info(
                        '...encoder [{}] using [{}] for [{}]'.format(
                            ename, ref_ctx_emb_b.name, trg_ctx_emb_bname))

                logger.info('Shared parameters in encoder {}'
                            .format(ref_enc_params.keys() +
                                    [ref_ctx_emb_W.name, ref_ctx_emb_b.name]))

    def build_models(self, x, x_mask, xs=None, x_masks=None, **kwargs):
        """
        x : theano tensor variable
        x_mask : theano tensor variable
        xs : list of theano tensor variables, for multi-source only
        x_masks : list of theano tensor variables, for multi-source only
        """
        logger.info(" MultiEncoder: building training models")

        self.init_params()
        ctx_reps = OrderedDict()
        for eid in self.enc_ids:
            if is_multiSource(eid):
                ctx = self.encoders[eid].build_model(xs, x_masks, **kwargs)
                ctx_reps[eid] = OrderedDict([
                    (enc, get_layer('ff')[1](
                     self.tparams, c_, prefix=_p('ctx_embedder', enc),
                     activ=self.representation_act, add_bias=True))
                    for enc, c_ in ctx.items()])
            else:
                ctx = self.encoders[eid].build_model(x, x_mask, **kwargs)
                ctx_reps[eid] = get_layer('ff')[1](
                    self.tparams, ctx, prefix=_p('ctx_embedder', eid),
                    activ=self.representation_act, add_bias=True)

        # apply ctx-dropout
        if 0. < self.dropout_ctx < 1.:
            for eid in ctx_reps.keys():
                logger.info(
                    " ... ... " +
                    "layer [{}] using ctx dropout with drop ratio:[{}]"
                    .format(_p('ctx_embedder', eid), self.dropout_ctx))
                ctx_reps[eid] = dropout_layer(
                    ctx_reps[eid], self.dropout_ctx, use_noise=True, **kwargs)

        return ctx_reps

    def build_sampling_models(self, x, xs=None, **kwargs):
        """
        x : theano tensor variable
        xs : list of theano tensor variables, for multi-source only
        """
        logger.info(" MultiEncoder: building sampling models")

        ctx_reps = OrderedDict()

        for eid in self.enc_ids:

            if is_multiSource(eid):
                ctx = self.encoders[eid].build_sampling_model(xs, **kwargs)
                ctx_reps[eid] = OrderedDict([
                    (enc, get_layer('ff')[1](
                     self.tparams, c_, prefix=_p('ctx_embedder', enc),
                     activ=self.representation_act, add_bias=True))
                    for enc, c_ in ctx.items()])
            else:
                ctx = self.encoders[eid].build_sampling_model(x, **kwargs)
                ctx_reps[eid] = get_layer('ff')[1](
                    self.tparams, ctx, prefix=_p('ctx_embedder', eid),
                    activ=self.representation_act, add_bias=True)

        return ctx_reps

    def get_params(self, encs_only=False):
        if encs_only:
            params = OrderedDict()
            for enc in self.encoders:
                params.update(enc.get_params())
            return params
        return self.tparams


class MultiDecoder(object):
    """
    Highest level decoder in the multi-way hierarchy
    --------------------------------------------------------
    A multi-decoder contains multiple decoders of type, Decoder, or any other
    decoder class that implements build_model, build_sampling_model,
    init_params, cost and f_log_probs methods.
    """

    def __init__(self, cgs, trg_vocab_sizes, dec_embed_sizes, dec_nhids,
                 representation_dim, num_encs, num_decs, share_att=True,
                 dec_rnn_type='gru_cond_multiEnc', shared_nl=False,
                 **kwargs):
        """
        cgs : list of computational graph ids, we use the following convention,
              encoder and decoder must be separated with '_' like ru_en
              encoder names can ben splitted with '.' for multi-source
        trg_vocab_sizes : dict, mapping dec_id to vocabulary size
        dec_embed_sizes : dict, mapping dec_id to vocabulary size
        dec_nhids : dict, mapping dec_id to number of hidden units
        representation_dim : dimension of joint encoder space (W_adp)
        num_encs : int, number of encoders
        num_decs : int, number of decoders
        share_att : bool, flag to indicate whether to share attention or not
        dec_rnn_type : layer name to be used by get_layer
        shared_nl : bool, flag to indicate whether v09 parameters should be
                    shared or not
        """
        self.cgs = cgs
        self.vocab_sizes = trg_vocab_sizes
        self.embedding_dims = dec_embed_sizes
        self.state_dims = dec_nhids
        self.representation_dim = representation_dim
        self.num_encs = num_encs
        self.num_decs = num_decs
        self.share_att = share_att
        self.dec_rnn_type = dec_rnn_type
        self.shared_nl = shared_nl

        self.enc_ids, self.dec_ids = get_enc_dec_ids(cgs)

        self.decoders = OrderedDict()
        for did in self.dec_ids:
            self.decoders[did] = Decoder(
                vocab_size=trg_vocab_sizes[did],
                embedding_dim=dec_embed_sizes[did],
                state_dim=dec_nhids[did],
                representation_dim=representation_dim,
                num_encs=num_encs, num_decs=num_decs,
                dec_id=did,
                enc_ids=[p_(name)[0] for name in self.cgs
                         if p_(name)[1] == did],
                rnn_type=dec_rnn_type, **kwargs)

    def init_params(self):
        self.params = OrderedDict()
        self.tparams = OrderedDict()
        self.shared_params_map = OrderedDict()

        for did, dec in self.decoders.iteritems():
            dec.init_params()
            self.params.update(dec.params)
            self.tparams.update(dec.tparams)

        if self.share_att and self.num_decs > 1:
            ref_dec_name, ref_dec_params = self._get_shared_params()
            for dname, dec in self.decoders.iteritems():
                if dname != ref_dec_name:
                    for pname in dec.tparams.keys():
                        ref_pname = pname.replace(
                            '_%s' % dname, _p('', ref_dec_name))
                        if ref_pname in ref_dec_params:
                            logger.info(
                                '...decoder [{}] using [{}] for [{}]'.format(
                                    dname, ref_pname, pname))
                            dec.tparams[pname] = ref_dec_params[ref_pname]
                            self.shared_params_map[pname] = ref_pname
            logger.info('Shared parameters in decoder {}'
                        .format(ref_dec_params.keys()))

    def _get_shared_params(self):
        ref_dec_name = self.decoders.keys()[0]
        ref_dec_params = {
            k: v for k, v in self.decoders[ref_dec_name].tparams.iteritems()
            if self._is_shared_param(k)}
        return (ref_dec_name, ref_dec_params)

    def _is_shared_param(self, pname):
        return (
            pname.endswith('_att') or
            pname.endswith('_sel') or
            pname.find('ff_initial_state') > -1 or
            pname.find('ff_logit_src_sel') > -1 or
            pname.endswith('_shared') or
            (pname.endswith('_nl') if self.shared_nl else False))

    def build_models(self, y, y_mask, x_mask, ctx_reps, src_sel_rep,
                     trg_sel_rep, **kwargs):
        """
        y : theano tensor variable
        y_mask : theano tensor variable
        x_mask : theano tensor variable
        ctx_reps : dict, mapping enc_id to encoder annoitations
        src_sel_rep : theano tensor variable, one hot encoding of encoders
        trg_sel_rep : theano tensor variable, one hot encoding of decoders
        """
        logger.info(" MultiDecoder: building training models")
        self.init_params()
        probs = OrderedDict()
        opt_rets = OrderedDict()
        for did in self.dec_ids:
            enc_ids_this = [p_(name)[0] for name in self.cgs
                            if p_(name)[1] == did]
            for eid in enc_ids_this:
                prob, opt_ret = self.decoders[did].build_model(
                    y, y_mask, x_mask, ctx_reps[eid], src_sel_rep, trg_sel_rep,
                    **kwargs)
                probs[_p(eid, did)] = prob
                opt_rets[_p(eid, did)] = opt_ret
        return probs, opt_rets

    def build_sampling_models(self, x, y, src_sel, trg_sel,
                              ctx_reps, prev_state, trng,
                              **kwargs):
        """
        x : theano tensor variable
        y : theano tensor variable
        src_sel : theano tensor variable, one hot encoding of encoders
        trg_sel : theano tensor variable, one hot encoding of decoders
        ctx_reps : dict, mapping enc_id to encoder annoitations
        prev_state : theano tensor variable
        trng : theano random stream
        """
        logger.info(" MultiDecoder: building sampling models")
        f_inits = OrderedDict()
        f_nexts = OrderedDict()
        f_next_states = OrderedDict()
        for did in self.dec_ids:
            enc_ids_this = [p_(name)[0] for name in self.cgs
                            if p_(name)[1] == did]
            for eid in enc_ids_this:
                ret = self.decoders[did].build_sampling_model(
                    x, y, src_sel, trg_sel, ctx_reps[eid], prev_state, eid,
                    trng=trng, **kwargs)
                f_inits[_p(eid, did)] = ret[0]
                f_nexts[_p(eid, did)] = ret[1]
                if len(ret) > 2:  # indicating look generate update
                    f_next_states[_p(eid, did)] = ret[2]
        return f_inits, f_nexts, f_next_states

    def get_params(self):
        return self.tparams

    def costs(self, probs, y, y_mask):
        """
        probs : dict, mapping cg_name to probabilities
        y : theano tensor variable
        y_mask : theano tensor variable
        """
        costs = OrderedDict()
        for cg in self.cgs:
            cost = self.decoders[p_(cg)[1]].cost(probs[cg], y, y_mask)
            cost.name = 'cost_{}'.format(cg)
            costs[cg] = cost
        return costs

    def get_f_log_probs(self, probs, x, x_mask, y, y_mask, src_sel, trg_sel,
                        xs=None, x_masks=None):
        """
        probs : dict, mapping cg_name to probabilities
        x : theano tensor variable
        x_mask : theano tensor variable
        y : theano tensor variable
        y_mask : theano tensor variable
        src_sel : theano tensor variable, one hot encoding of encoders
        trg_sel : theano tensor variable, one hot encoding of decoders
        xs : list of theano tensor variables, used ony for multi-source
        x_masks : list of theano tensor variables, used ony for multi-source
        """
        f_log_probs = OrderedDict()
        for cg in self.cgs:
            if is_multiSource(cg):
                f_log_probs[cg] = self.decoders[p_(cg)[1]].f_log_probs(
                    probs[cg], xs, x_masks, y, y_mask, src_sel, trg_sel,
                    cg=cg)
            else:
                f_log_probs[cg] = self.decoders[p_(cg)[1]].f_log_probs(
                    probs[cg], x, x_mask, y, y_mask, src_sel, trg_sel)
        return f_log_probs


class Decoder(object):
    """
    Lowest level decoder in the multi-way hierarchy
    --------------------------------------------------------
    Decoder carries the heavy burden of the entire multi-way hieararchy. It
    attends, encoder(s).  This class should probably be factorized.
    """

    def __init__(self, vocab_size, embedding_dim, state_dim,
                 representation_dim, num_encs, num_decs, att_dim=None,
                 finit_mid_dim=None, finit_code_dim=None, dec_id=0,
                 enc_ids=None, rnn_type='gru_cond_multiEnc',
                 cond_init_trg=False, deeper_att=False, readout_dim=None,
                 take_last=False, multi_latent=False, readout_nonlin='tanh',
                 representation_act=None, finit_act='linear',
                 lencoder_act='linear', ldecoder_act='tanh',
                 lctxproj_act='tanh', attend_merge_op=None, init_merge_op=None,
                 attend_merge_act='linear', init_merge_act='linear',
                 bundle_states=None, dec_layer_norm=False, **kwargs):
        """
        vocab_size : int, vocabulary size
        embedding_dim : int, embedding dimension
        state_dim : int, number of hidden state of the recurrent net
        representation_dim : int, encoder annotation dimension (W_adp)
        num_encs : int, number of encoders
        num_decs : int, number of decoders
        att_dim : int, internal attention dimension
        finit_mid_dim : int, phi_init dimension
        finit_code_dim : int, psi_init dimension
        dec_id : str, decoder id eg. en
        enc_ids : list of encoder ids
        rnn_type : layer name to be used by get_layer
        cond_init_trg : bool, whether to condition on target selector or not
        deeper_att : bool, adds additional layer to attention module
        readout_dim : int, dimension of the readout layer
        take_last : bool, take the last hidden state of the backward rnn
        multi_latent : bool, should be True (contact me for details)
        readout_nonlin : activation name, to be used by readout layer
        representation_act : activation name, to be applied to encoder ctx
        finit_act : activation name, to be applied to f_init function
        lencoder_act : activation name, to be applied to encoder latent space
        ldecoder_act : activation name, to be applied to decoder latent space
        lctxproj_act : activation name, to be applied to context vector
        attend_merge_op : merge op, for multi-source attention
        attend_merge_act : activation name, to be applied after attend merge op
        init_merge_op : merge op, for multi-source initializer
        init_merge_act : activation name, to be applied after init merge op
        bundle_states : bundle states and concatenate
        dec_layer_norm : apply layer normalization to decoder rnn
        """

        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.state_dim = state_dim
        self.representation_dim = representation_dim
        self.num_encs = num_encs
        self.num_decs = num_decs
        self.att_dim = att_dim
        self.finit_mid_dim = finit_mid_dim
        self.finit_code_dim = finit_code_dim
        self.rnn_type = rnn_type
        self.dec_id = dec_id
        self.enc_ids = enc_ids
        self.cond_init_trg = cond_init_trg
        self.deeper_att = deeper_att
        self.readout_dim = readout_dim
        self.take_last = take_last
        self.multi_latent = multi_latent
        self.readout_nonlin = readout_nonlin
        self.representation_act = representation_act
        self.finit_act = finit_act
        self.lencoder_act = lencoder_act
        self.ldecoder_act = ldecoder_act
        self.lctxproj_act = lctxproj_act
        self.dec_layer_norm = dec_layer_norm

        # used by multiSource decoder
        self.attend_merge_op = attend_merge_op
        self.attend_merge_act = attend_merge_act
        self.init_merge_op = init_merge_op
        self.init_merge_act = init_merge_act
        self.bundle_states = bundle_states
        self._attends_multiSource = (
            True if any([is_multiSource(eid) for eid in self.enc_ids])
            else False)

        if self.readout_dim is None:
            self.readout_dim = embedding_dim

    def init_params(self):
        did = self.dec_id

        params = OrderedDict()
        ctxdim = self.representation_dim

        params[_p('Wemb_dec', did)] = norm_weight(self.vocab_size,
                                                  self.embedding_dim)

        # get the input dimension of the initial state
        initdim = ctxdim

        # init_state, init_cell
        params = get_layer('ff_init')[0](
            params, prefix=_p('ff_init', did),
            nin=initdim, nout=self.state_dim,
            nmid=self.finit_mid_dim,
            ncode=self.finit_code_dim)

        # branch for layer normalization
        dec_rnn_type = self.rnn_type
        if self.dec_layer_norm:
            dec_rnn_type = 'ln' + dec_rnn_type

        # decoder
        if self._attends_multiSource:

            # we are using a special type of decoder initializer
            params = get_layer('ff_init_merger')[0](
                params, prefix=_p('ff_init_merger', did),
                nin=initdim, init_merge_op=self.init_merge_op,
                bundle_states=self.bundle_states)

            # we are using a special type of gru with attention
            params = get_layer(dec_rnn_type + '_mSrc')[0](
                params, prefix=_p('decoder', did), nin=self.embedding_dim,
                dim=self.state_dim, dimctx=ctxdim, dimatt=self.att_dim,
                attend_merge_op=self.attend_merge_op,
                attend_merge_act=self.attend_merge_act,
                bundle_states=self.bundle_states)
        else:
            params = get_layer(dec_rnn_type)[0](
                params, prefix=_p('decoder', did), nin=self.embedding_dim,
                dim=self.state_dim, dimctx=ctxdim, num_encs=self.num_encs,
                num_decs=self.num_decs, deeper_att=self.deeper_att,
                multi_latent=self.multi_latent,
                dimatt=self.att_dim)

        # readout
        readout_dim = self.readout_dim
        state_dim = self.state_dim

        params = get_layer('ff')[0](
            params, prefix=_p('ff_logit_lstm', did), nin=state_dim,
            nout=readout_dim, ortho=False)

        params = get_layer('ff')[0](params, prefix=_p('ff_logit_prev', did),
                                    nin=self.embedding_dim,
                                    nout=readout_dim,
                                    ortho=False)
        params = get_layer('ff')[0](params, prefix=_p('ff_logit_ctx', did),
                                    nin=ctxdim,
                                    nout=readout_dim,
                                    ortho=False)

        # transformation before softmax
        preact_dim = (readout_dim/2
                      if self.readout_nonlin == 'maxout'
                      else readout_dim)
        params = get_layer('ff')[0](params, prefix=_p('ff_logit', did),
                                    nin=preact_dim,
                                    nout=self.vocab_size)
        self.params = params
        self.tparams = init_tparams(self.params)

    def build_models(self, y, y_mask, x_mask,
                     ctx_reps, src_sel_rep, trg_sel_rep,
                     x_masks=None, **kwargs):
        """
        y : theano tensor variable
        y_mask : theano tensor variable
        x_mask : theano tensor variable
        ctx_reps : dict, mapping enc_id to encoder annoitations
        src_sel_rep : theano tensor variable, one hot encoding of encoders
        trg_sel_rep : theano tensor variable, one hot encoding of decoders
        x_masks : list of theano tensor variables, for multi-source
        """
        self.init_params()
        probs = OrderedDict()
        opt_rets = OrderedDict()
        for eid in self.enc_ids:
            logger.info(
                " ... Decoder [{}-{}] building training models"
                .format(eid, self.dec_id))
            prob, opt_ret = self.build_model(
                y, y_mask, x_mask, ctx_reps[eid], src_sel_rep, trg_sel_rep,
                x_masks=x_masks, eid=eid, **kwargs)
            probs[_p(eid, self.dec_id)] = prob
            opt_rets[_p(eid, self.dec_id)] = opt_ret
        return probs, opt_rets

    def build_sampling_models(self, x, y, src_sel, trg_sel,
                              ctx_reps, prev_state, trng,
                              **kwargs):
        """
        x : theano tensor variable
        y : theano tensor variable
        src_sel : theano tensor variable, one hot encoding of encoders
        trg_sel : theano tensor variable, one hot encoding of decoders
        ctx_reps : dict, mapping enc_id to encoder annoitations
        prev_state : theano tensor variable
        trng : theano random stream
        """
        f_inits = OrderedDict()
        f_nexts = OrderedDict()
        f_next_states = OrderedDict()
        for eid in self.enc_ids:
            logger.info(
                " ... Decoder [{}-{}] building sampling models"
                .format(eid, self.dec_id))
            ret = self.build_sampling_model(
                x, y, src_sel, trg_sel, ctx_reps[eid], prev_state, trng, eid,
                **kwargs)
            f_inits[_p(eid, self.dec_id)] = ret[0]
            f_nexts[_p(eid, self.dec_id)] = ret[1]
        return f_inits, f_nexts, f_next_states

    def build_model(self, y, y_mask, x_mask,
                    ctx_rep, src_sel_rep, trg_sel_rep,
                    x_masks=None, eid=None, **kwargs):
        """
        y : theano tensor variable
        y_mask : theano tensor variable
        x_mask : theano tensor variable
        ctx_rep : encoder annotations
        src_sel_rep : theano tensor variable, one hot encoding of encoders
        trg_sel_rep : theano tensor variable, one hot encoding of decoders
        x_masks : list of theano tensor variables, for multi-source
        eid : str, encoder id
        """

        # helpers
        did = self.dec_id
        n_timesteps_trg = y.shape[0]
        n_samples = y.shape[1]

        # for multisource we will have multiple context coming from multiple
        # encoders we will use an mlp to merge initial states
        if isinstance(ctx_rep, dict):  # branch to multisource
            if not self.take_last:
                raise NotImplementedError(
                    'Initializer merger implemented only for take_last!')
            logger.info(
                " ... using initializer merger [{}] for encoders: {}"
                .format(self.init_merge_op, ctx_rep.keys()))
            states_below = [ctx_rep[enc_name][0, :, -self.representation_dim:]
                            for enc_name in ctx_rep.keys()]
            ctx_init = get_layer('ff_init_merger')[1](
                self.tparams, states_below, prefix=_p('ff_init_merger', did),
                init_merge_op=self.init_merge_op,
                init_merge_act=self.init_merge_act,
                bundle_states=self.bundle_states)

        else:  # single pair

            # initial decoder state
            if self.take_last:
                ctx_init = ctx_rep[0, :, -self.representation_dim:]
            else:
                ctx_init = (ctx_rep * x_mask[:, :, None]).sum(0) / \
                    x_mask.sum(0)[:, None]

        init_state = get_layer('ff_init')[1](self.tparams, ctx_init,
                                             prefix=_p('ff_init', did),
                                             activ='tanh',
                                             post_activ=self.finit_act)

        # word embedding (target)
        emb = self.tparams[_p('Wemb_dec', did)][y.flatten()]
        emb = emb.reshape([n_timesteps_trg, n_samples, self.embedding_dim])

        # in order to get bi-gram we need to shift emb one time step ahead
        emb_shifted = tensor.zeros_like(emb)
        emb_shifted = tensor.set_subtensor(emb_shifted[1:], emb[:-1])

        emb = emb_shifted

        # decoder
        # concatenate source and target selector to the context for speed up
        context = ctx_rep
        opt_ret = dict()

        # get the proper layer name for normalization
        dec_rnn_type = ('ln' + self.rnn_type if self.dec_layer_norm
                        else self.rnn_type)

        if isinstance(ctx_rep, dict):  # branch to multisource
            logger.info(
                " ... using post-context merger [{}] for encoders: {}"
                .format(self.attend_merge_op, ctx_rep.keys()))
            dec_rnn_type = dec_rnn_type + '_mSrc'

            proj = get_layer(dec_rnn_type)[1](
                self.tparams, emb, prefix=_p('decoder', did), mask=y_mask,
                contexts=context, context_masks=x_masks, one_step=False,
                init_state=init_state, src_selector=src_sel_rep,
                trg_selector=trg_sel_rep, deeper_att=self.deeper_att,
                lencoder_act=self.lencoder_act, ldecoder_act=self.ldecoder_act,
                lctxproj_act=self.lctxproj_act,
                attend_merge_op=self.attend_merge_op,
                attend_merge_act=self.attend_merge_act,
                bundle_states=self.bundle_states, **kwargs)
        else:
            proj = get_layer(dec_rnn_type)[1](
                self.tparams, emb, prefix=_p('decoder', did), mask=y_mask,
                context=context, context_masks=x_mask, one_step=False,
                init_state=init_state, src_selector=src_sel_rep,
                trg_selector=trg_sel_rep, deeper_att=self.deeper_att,
                lencoder_act=self.lencoder_act, ldecoder_act=self.ldecoder_act,
                lctxproj_act=self.lctxproj_act, **kwargs)

        logger.info(" ... ... using [{}] layer".format(dec_rnn_type))

        proj_h = proj[0]
        ctxs = proj[1]

        if isinstance(ctx_rep, dict):  # branch to multisource
            opt_ret['dec_alphas'] = proj[2:2+len(ctx_rep)]
        else:
            opt_ret['dec_alphas'] = proj[2]

        # compute the preactivation of softmax
        preact = self.get_softmax_preactivation(proj_h, emb, ctxs, src_sel_rep)

        # compute word probabilities
        preact_shp = preact.shape
        probs = tensor.nnet.softmax(
            preact.reshape([preact_shp[0]*preact_shp[1], preact_shp[2]]))
        return probs, opt_ret

    def get_softmax_preactivation(self, proj_h, emb, ctxs, src_sel_rep):
        did = self.dec_id
        logit_lstm = get_layer('ff')[1](self.tparams, proj_h,
                                        prefix=_p('ff_logit_lstm', did),
                                        activ='linear')
        logit_prev = get_layer('ff')[1](self.tparams, emb,
                                        prefix=_p('ff_logit_prev', did),
                                        activ='linear')
        logit_ctx = get_layer('ff')[1](self.tparams, ctxs,
                                       prefix=_p('ff_logit_ctx', did),
                                       activ='linear')

        # apply readout non-linearity
        logit = logit_lstm + logit_prev + logit_ctx

        if self.readout_nonlin == 'maxout':
            logit_shp = logit.shape
            if logit.ndim == 2:  # sampling
                logit = logit.reshape(
                    [logit_shp[0], logit_shp[1]//2, 2]).max(-1)
            elif logit.ndim == 3:  # training
                logit = logit.reshape(
                    [logit_shp[0], logit_shp[1], logit_shp[2]//2, 2]).max(-1)
            else:
                raise ValueError('Readout activation shape unknown')
        else:
            logit = tensor.tanh(logit)

        # preactivation of softmax
        preact = get_layer('ff')[1](self.tparams, logit,
                                    prefix=_p('ff_logit', did),
                                    activ='linear')
        return preact

    def build_sampling_model(self, x, y, src_selector, trg_selector,
                             ctx_rep, prev_state, enc_id, return_alphas=False,
                             xs=None, trng=None, **kwargs):

        # helpers
        did = self.dec_id

        # branch to multisource
        if isinstance(ctx_rep, dict):
            if not self.take_last:
                raise NotImplementedError(
                    'Initializer merger implemented only for take_last!')
            logger.info(
                " ... using initializer merger [{}] for encoders: {}"
                .format(self.init_merge_op, ctx_rep.keys()))
            states_below = [ctx_rep[enc_name][0, :, -self.representation_dim:]
                            for enc_name in ctx_rep.keys()]
            ctx_init = get_layer('ff_init_merger')[1](
                self.tparams, states_below, prefix=_p('ff_init_merger', did),
                init_merge_op=self.init_merge_op,
                init_merge_act=self.init_merge_act,
                bundle_states=self.bundle_states)

        else:  # single pair

            # initial decoder state
            if self.take_last:
                ctx_init = ctx_rep[0, :, -self.representation_dim:]
            else:
                ctx_init = ctx_rep.mean(0)

        func_inputs = (xs[:len(ctx_rep)] if isinstance(ctx_rep, dict) else [x])
        init_state = get_layer('ff_init')[1](self.tparams, ctx_init,
                                             prefix=_p('ff_init', did),
                                             activ='tanh',
                                             post_activ=self.finit_act)

        func_outputs = [init_state] + \
            (ctx_rep.values() if isinstance(ctx_rep, dict) else [ctx_rep])

        logger.info('Building f_init for CG[{}-{}]...'.format(enc_id, did))
        f_init = theano.function(inputs=func_inputs,
                                 outputs=func_outputs,
                                 name=_p('f_init', _p(enc_id, did)),
                                 on_unused_input='warn')

        # if it's the first word, emb should be all zero
        emb = tensor.switch(
            y[:, None] < 0,
            tensor.alloc(0., 1, self.tparams[_p('Wemb_dec', did)].shape[1]),
            self.tparams[_p('Wemb_dec', did)][y])

        # get the proper layer name for normalization
        dec_rnn_type = ('ln' + self.rnn_type if self.dec_layer_norm
                        else self.rnn_type)

        # pass through attention and gru to get the next state and context
        if isinstance(ctx_rep, dict):  # branch to multisource
            logger.info(
                " ... using post-context merger [{}] for encoders: {}"
                .format(self.attend_merge_op, ctx_rep.keys()))
            dec_rnn_type = dec_rnn_type + '_mSrc'

            proj = get_layer(dec_rnn_type)[1](
                self.tparams, emb, prefix=_p('decoder', did), contexts=ctx_rep,
                one_step=True, init_state=prev_state,
                src_selector=src_selector, trg_selector=trg_selector,
                state_before=None, lencoder_act=self.lencoder_act,
                ldecoder_act=self.ldecoder_act, return_alphas=return_alphas,
                attend_merge_op=self.attend_merge_op,
                attend_merge_act=self.attend_merge_act,
                bundle_states=self.bundle_states, trng=trng)
        else:
            proj = get_layer(dec_rnn_type)[1](
                self.tparams, emb, prefix=_p('decoder', did), context=ctx_rep,
                one_step=True, init_state=prev_state,
                src_selector=src_selector, trg_selector=trg_selector,
                state_before=None, lencoder_act=self.lencoder_act,
                ldecoder_act=self.ldecoder_act, return_alphas=return_alphas,
                lctxproj_act=self.lctxproj_act, trng=trng)

        logger.info(" ... ... using [{}] layer".format(dec_rnn_type))

        next_state = proj[0]
        ctxs = proj[1]

        # compute the preactivation of softmax
        preact = self.get_softmax_preactivation(
            next_state, emb, ctxs, src_selector)

        # next word probability
        next_probs = tensor.nnet.softmax(preact)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)

        # define input and outputs of f_next
        logger.info('Building f_next for decoder[{}-{}]..'.format(enc_id, did))

        f_next_inps = [y] + \
            (ctx_rep.values() if isinstance(ctx_rep, dict) else [ctx_rep]) +\
            [prev_state]

        f_next_outs = [next_probs, next_sample, next_state]

        if return_alphas:
            if isinstance(ctx_rep, dict):  # branch to multisource
                num_src = len(ctx_rep)
                f_next_outs += proj[2:2+num_src]
            else:
                f_next_outs.append(proj[2])

        f_next = theano.function(
            inputs=f_next_inps,
            outputs=f_next_outs,
            name=_p('f_next', _p(enc_id, did)),
            on_unused_input='warn')

        f_next_state = None
        return f_init, f_next, f_next_state

    def cost(self, probs, y, y_mask):
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.vocab_size + y_flat
        cost = -tensor.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)
        cost = cost.mean()
        return cost

    def f_log_probs(self, probs, x, x_mask, y, y_mask,
                    src_selector, trg_selector, cg=None):
        y_flat = y.flatten()
        y_flat_idx = tensor.arange(y_flat.shape[0]) * self.vocab_size + y_flat
        cost = -tensor.log(probs.flatten()[y_flat_idx])
        cost = cost.reshape([y.shape[0], y.shape[1]])
        cost = (cost * y_mask).sum(0)
        if self._attends_multiSource and \
                isinstance(x, list) and isinstance(x_mask, list):
            n_attended = len(get_enc_ids(cg))
            func_inps = x[:n_attended] + x_mask[:n_attended] +\
                [y, y_mask, src_selector, trg_selector]
        else:
            func_inps = [x, x_mask, y, y_mask, src_selector, trg_selector]
        return theano.function(
            inputs=func_inps,
            outputs=cost, on_unused_input='warn')

    def get_params(self):
        return self.tparams


class EncoderDecoder(object):
    """
    Highest level abstraction of multi-way encoder decoder
    --------------------------------------------------------
    Stitches any encoder with any decoder, best to be used with a MultiEncoder
    and a MultiDecoder. This class handles the extension related jobs, like
    loading parameters, printing parameter details, printing trained and
    excluded parameters. It is also responsible to build computational graphs,
    computing costs and adding regularizers to the costs.
    """

    def __init__(self, encoder, decoder, **kwargs):
        """
        encoder : class, encoder
        decoder : class, decoder
        """
        self.encoder = encoder
        self.decoder = decoder
        self.probs = OrderedDict()
        self.opt_rets = OrderedDict()

        self.num_encs = 1
        if isinstance(encoder, MultiEncoder):
            self.num_encs = encoder.num_encs

        self.num_decs = 1
        if isinstance(decoder, MultiDecoder):
            self.num_decs = decoder.num_decs

    def build_models(self, x, x_mask, y, y_mask, src_sel, trg_sel, **kwargs):
        """
        x : theano tensor variable
        x_mask : theano tensor variable
        y : theano tensor variable
        y_mask : theano tensor variable
        src_sel : theano tensor variable, one hot encoding of encoders
        trg_sel : theano tensor variable, one hot encoding of decoders
        """
        logger.info(" Encoder-Decoder: building training models")
        ctx_reps = self.encoder.build_models(x=x, x_mask=x_mask, **kwargs)
        probs, opt_rets = self.decoder.build_models(
            y, y_mask, x_mask, ctx_reps, src_sel, trg_sel, **kwargs)
        return probs, opt_rets

    def build_sampling_models(self, x, y, src_sel, trg_sel, prev_state,
                              trng=None, **kwargs):
        """
        x : theano tensor variable
        y : theano tensor variable
        src_sel : theano tensor variable, one hot encoding of encoders
        trg_sel : theano tensor variable, one hot encoding of decoders
        prev_state : theano tensor variable
        trng : theano random stream
        """
        logger.info(" Encoder-Decoder: building sampling models")
        ctx_reps_s = self.encoder.build_sampling_models(x, **kwargs)
        f_inits, f_nexts, f_next_states = self.decoder.build_sampling_models(
            x, y, src_sel, trg_sel, ctx_reps_s, prev_state, trng=trng,
            **kwargs)
        return f_inits, f_nexts, f_next_states

    def get_costs(self, probs, y, y_mask,
                  decay_cs=None, opt_rets=None):
        """
        probs : dict, mapping cg_name to probabilities
        y : theano tensor variable
        y_mask : theano tensor variable
        decay_cs : list of l2 regularization weights
        opt_rets : dict, mapping cg_name to optional returned variables
        """
        costs = self.decoder.costs(probs, y, y_mask)

        if decay_cs is not None:
            for name, cost in costs.iteritems():
                if decay_cs[name] > 0.:
                    decay_c = theano.shared(numpy.float32(decay_cs[name]),
                                            name='decay_c')
                    weight_decay = 0.
                    for pp in ComputationGraph(cost).parameters:
                        weight_decay += (pp ** 2).sum()
                    weight_decay *= decay_c
                    costs[name] += weight_decay
                    costs[name].name = name

        return costs

    def get_computational_graphs(self, costs):
        """
        costs : dict, mapping cg_name to cost
        """
        cgs = OrderedDict()
        for name, cost in costs.iteritems():
            cg = ComputationGraph(cost)
            cgs[name] = cg
        return cgs

    def print_params(self, cgs):
        """
        cgs : list of computational graph names
        """
        for name, cg in cgs.iteritems():
            shapes = [param.get_value().shape for param in cg.parameters]
            logger.info(
                "Parameter shapes for computation graph[{}]".format(name))
            for shape, count in Counter(shapes).most_common():
                logger.info('    {:15}: {}'.format(shape, count))
            logger.info(
                "Total number of parameters for computation graph[{}]: {}"
                .format(name, len(shapes)))

            logger.info(
                "Parameter names for computation graph[{}]: ".format(name))
            for item in cg.parameters:
                logger.info(
                    "    {:15}: {}".format(item.get_value().shape, item.name))
            logger.info(
                "Total number of parameters for computation graph[{}]: {}"
                .format(name, len(cg.parameters)))

    def get_training_params(self, cgs, exclude_encs=None, exclude_decs=None,
                            additional_excludes=None, readout_only=None,
                            train_shared=None):
        """
        cgs : list of computational graph names
        exclude_encs : bool, whether to exclude encoders from training
        exclude_decs : bool, whether to exclude decoders from training
        additional_excludes : list of parameters to be excluded (in addition)
        readout_only : bool, whether to train only readout
        train_shared : bool, whether to include shared components to training
        """

        # By default train all parameters, and exclude none
        excluded_params = OrderedDict({name: list() for name in cgs.keys()})
        training_params = OrderedDict({name: [p for p in cg.parameters]
                                      for name, cg in cgs.items()})

        # Exclude encoder params
        if exclude_encs is not None:
            for name, cg in cgs.iteritems():
                eids = p_(name)[0].split('.')
                for eid in eids:
                    if exclude_encs[eid]:
                        p_enc = self.encoder.encoders[eid].get_params()
                        for pname, pval in p_enc.items():
                            pidx = get_param_idx(training_params[name], pname)
                            if pidx is not None:
                                excluded_params[name].append(
                                    training_params[name].pop(pidx))

        if exclude_decs is not None:
            for name, cg in cgs.iteritems():
                dids = p_(name)[1].split('.')
                for did in dids:
                    if exclude_decs[did]:
                        p_dec = self.decoder.decoders[did].get_params()
                        for pname, pval in p_dec.items():
                            pidx = get_param_idx(training_params[name], pname)
                            if pidx is not None:
                                excluded_params[name].append(
                                    training_params[name].pop(pidx))

        # Filter everything and train only readout
        if readout_only is not None:
            for name, cg in cgs.items():
                if readout_only[name]:
                    for p in cg.parameters:
                        if not p.name.startswith('ff_controller') and \
                                not p.name.startswith('ff_logit') and \
                                'merger' not in p.name:
                            pidx = get_param_idx(training_params[name], p.name)
                            if pidx is not None:
                                excluded_params[name].append(
                                    training_params[name].pop(pidx))

        # Add shared parameters to the training params
        if train_shared is not None:
            shared_params = self.decoder._get_shared_params()[1]
            for name, cg in cgs.items():
                if train_shared[name]:
                    for pname, pval in shared_params.items():
                        pidx = get_param_idx(excluded_params[name], pname)
                        if pidx is not None:
                            training_params[name].append(
                                excluded_params[name].pop(pidx))

        # Put parameter into exclude list
        if additional_excludes is not None:
            for name, cg in cgs.iteritems():
                for p in additional_excludes[name]:
                    pidx_ = get_param_idx(excluded_params[name], p)
                    if pidx_ is not None:
                        logger.warn('parameter [{}] already excluded'
                                    .format(p))
                        continue
                    pidx = get_param_idx(training_params[name], p)
                    if pidx is not None:
                        excluded_params[name].append(
                            training_params[name].pop(pidx))

        # Check if params dispersed correctly among training and exclude
        for name, cg in cgs.items():
            if len(training_params[name]) + len(excluded_params[name]) != \
                    len(cg.parameters):
                raise ValueError(
                    'Parameter numbers not matching for cg[{}]:'.format(name) +
                    ' training[{}] excluded[{}] total[{}]'
                    .format(len(training_params[name]),
                            len(excluded_params[name]),
                            len(cg.parameters)))

        return training_params, excluded_params

    def print_training_params(self, cgs, training_params):

        enc_dec_param_dict = merge(self.encoder.get_params(),
                                   self.decoder.get_params())

        # Print which parameters are excluded
        for k, v in cgs.iteritems():
            excluded_all = list(set(v.parameters) - set(training_params[k]))
            for p in excluded_all:
                logger.info(
                    'Excluding from training of CG[{}]: {}'
                    .format(k, [key for key, val in
                                enc_dec_param_dict.iteritems()
                                if val == p][0]))
            logger.info(
                'Total number of excluded parameters for CG[{}]: [{}]'
                .format(k, len(excluded_all)))

        for k, v in training_params.iteritems():
            for p in v:
                logger.info('Training parameter from CG[{}]: {}'
                            .format(k, p.name))
            logger.info(
                'Total number of parameters will be trained for CG[{}]: [{}]'
                .format(k, len(v)))

    def build_f_log_probs(self, probs, x, x_mask, y, y_mask, src_sel, trg_sel,
                          xs=None, x_masks=None):
        """
        probs : dict, mapping cg_name to probabilities
        x: theano tensor variable
        x_mask: theano tensor variable
        y: theano tensor variable
        y_mask: theano tensor variable
        src_sel : theano tensor variable, one hot encoding of encoders
        trg_sel : theano tensor variable, one hot encoding of decoders
        xs : list of theano tensor variables, for multi-source
        x_masks : list of theano tensor variables, for multi-source
        """
        return self.decoder.get_f_log_probs(
            probs, x, x_mask, y, y_mask, src_sel, trg_sel,
            xs=xs, x_masks=x_masks)

    def init_params(self):
        self.encoder.init_params()
        self.decoder.init_params()

    def get_params(self):
        return merge(self.encoder.get_params(),
                     self.decoder.get_params())

    def load_params(self, saveto):
        try:
            logger.info(" ...loading model parameters")
            params_all = numpy.load(saveto)
            params_this = self.get_params()
            missing = set(params_this) - set(params_all)
            for pname in params_this.keys():
                if pname in params_all:
                    val = params_all[pname]
                    self._set_param_value(params_this[pname], val, pname)
                elif self.num_decs > 1 and self.decoder.share_att and \
                        pname in self.decoder.shared_params_map:
                    val = params_all[self.decoder.shared_params_map[pname]]
                    self._set_param_value(params_this[pname], val, pname)
                elif self.encoder.share_encoder and \
                        pname in self.encoder.shared_params_map:
                    val = params_all[self.encoder.shared_params_map[pname]]
                    self._set_param_value(params_this[pname], val, pname)
                else:
                    logger.warning(
                        " Parameter does not exist: {}".format(pname))

            logger.info(
                " Number of params loaded: {}"
                .format(len(params_this) - len(missing)))
        except Exception as e:
            logger.error(" Error {0}".format(str(e)))

    def _set_param_value(self, param, val, pname):
        if param.get_value().shape != val.shape:
            logger.warning(
                " Dimension mismatch {}-{} for {}"
                .format(param.get_value().shape, val.shape, pname))
        param.set_value(val)
        logger.info(" Loaded {:15}: {}".format(val.shape, pname))
