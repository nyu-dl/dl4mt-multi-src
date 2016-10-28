import logging
import numpy
import theano

from collections import OrderedDict

from .utils import _p, concatenate, norm_weight, ortho_weight
from theano import tensor

logger = logging.getLogger(__name__)
floatX = theano.config.floatX

layers = {
    'ff': ('param_init_fflayer',
           'fflayer'),
    'lnff': ('param_init_lnfflayer',
             'lnfflayer'),
    'ff_init': ('param_init_ffinit_layer',
                'ffinit_layer'),
    'ff_init_merger': ('param_init_ff_initial_state_merger',
                       'ff_initial_state_merger_layer'),
    'ff_attend_merger': ('param_init_ff_attend_merger',
                         'ff_attend_merger_layer'),
    'gru': ('param_init_gru',
            'gru_layer'),
    'lngru': ('param_init_lngru',
              'lngru_layer'),
    'gru_cond_mCG': ('param_init_gru_cond_mCG',
                     'gru_cond_mCG_layer'),
    'lngru_cond_mCG': ('param_init_lngru_cond_mCG',
                       'lngru_cond_mCG_layer'),
    'gru_cond_mCG_mSrc': ('param_init_gru_cond_mCG_mSrc',
                          'gru_cond_mCG_mSrc_layer'),
    'lngru_cond_mCG_mSrc': ('param_init_lngru_cond_mCG_mSrc',
                            'lngru_cond_mCG_mSrc_layer')
}


# lateral normalization
def ln(x, b, s):
    _eps = 1e-5
    output = (x - x.mean(1)[:, None]) / tensor.sqrt((x.var(1)[:, None] + _eps))
    output = s[None, :] * output + b[None, :]
    return output


def get_layer(name):
    fns = layers[name]
    return (eval(fns[0]), eval(fns[1]))


def tanh(x):
    return tensor.tanh(x)


def linear(x):
    return x


def relu(x):
    return tensor.maximum(0.0, x)


def logistic(x):
    return tensor.nnet.sigmoid(x)


# feedforward layer: affine transformation + point-wise nonlinearity
def param_init_fflayer(params, prefix='ff', nin=None, nout=None, ortho=True,
                       add_bias=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if add_bias:
        params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')
    return params


def fflayer(tparams, state_below, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', add_bias=True, **kwargs):
    preact = tensor.dot(state_below, tparams[_p(prefix, 'W')])
    if add_bias:
        preact += tparams[_p(prefix, 'b')]
    return eval(activ)(preact)


# feedforward layer but layer normalized
def param_init_lnfflayer(params, prefix='ff', nin=None, nout=None, ortho=True,
                       add_bias=True):
    params[_p(prefix, 'W')] = norm_weight(nin, nout, scale=0.01, ortho=ortho)
    if add_bias:
        params[_p(prefix, 'b')] = numpy.zeros((nout,)).astype('float32')

    # LN parameters
    scale_add = 0.0
    scale_mul = 1.0
    params[_p(prefix, 'b1')] = scale_add * numpy.ones((2*nout)).astype(floatX)
    params[_p(prefix, 's1')] = scale_mul * numpy.ones((2*nout)).astype(floatX)
    return params


def lnfflayer(tparams, state_below, prefix='rconv',
            activ='lambda x: tensor.tanh(x)', add_bias=True, **kwargs):
    preact = tensor.dot(state_below, tparams[_p(prefix, 'W')])
    if add_bias:
        preact += tparams[_p(prefix, 'b')]
    b1 = tparams[_p(prefix, 'b1')]
    s1 = tparams[_p(prefix, 's1')]
    preact = ln(preact, b1, s1)
    return eval(activ)(preact)


# feedforward decoder initializer layer:
# two affine transformations (first one is applied by the encoder already,
#   the second one is shared across decoders) +
# point-wise nonlinearity +
# two affine transformations back (first one is shared across decoders,
#   the second one is decoder specific)
def param_init_ffinit_layer(params, prefix='ff_init', nin=None, nmid=None,
                            ncode=None, nout=None, ortho=True):

    if nmid is None:
        nmid = nin
    if ncode is None:
        ncode = nin

    # from encoder specific context embedding to code
    params[_p(prefix, 'W_shared')] = norm_weight(nin, ncode, scale=0.01,
                                                 ortho=ortho)
    # from code to decoder specific embedding
    params[_p(prefix, 'U_shared')] = norm_weight(ncode, nmid, scale=0.01,
                                                 ortho=ortho)

    # decoder specific embedding
    params[_p(prefix, 'U')] = norm_weight(nmid, nout, scale=0.01, ortho=ortho)
    params[_p(prefix, 'c')] = numpy.zeros((nout,)).astype('float32')
    return params


def ffinit_layer(tparams, state_below, prefix='ff_init',
                 activ='lambda x: tensor.tanh(x)',
                 post_activ='lambda x: x', **kwargs):

    preact = tensor.dot(state_below, tparams[_p(prefix, 'W_shared')])
    code = eval(activ)(preact)
    act = tensor.dot(code, tparams[_p(prefix, 'U_shared')])

    act = tensor.dot(act, tparams[_p(prefix, 'U')])
    act += tparams[_p(prefix, 'c')]
    return eval(post_activ)(act)


# GRU layer
def param_init_gru(params, prefix='gru', nin=None, dim=None, hiero=False):
    if not hiero:
        W = numpy.concatenate([norm_weight(nin, dim),
                               norm_weight(nin, dim)], axis=1)
        params[_p(prefix, 'W')] = W
        params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    return params


def gru_layer(tparams, state_below, prefix='gru', mask=None, **kwargs):
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(
        state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(
        state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step_slice(m_, x_, xx_, h_, U, Ux):
        preact = tensor.dot(h_, U)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    rval, _ = theano.scan(_step,
                          sequences=seqs,
                          outputs_info=[tensor.alloc(0., n_samples, dim)],
                          non_sequences=[tparams[_p(prefix, 'U')],
                                         tparams[_p(prefix, 'Ux')]],
                          name=_p(prefix, '_layers'),
                          n_steps=nsteps,
                          strict=True)
    rval = [rval]
    return rval


# Conditional GRU layer with Attention for multiple encoders - version 0.8
def param_init_gru_cond_mCG(params, nin, dim, dimctx, dimatt,
                            prefix='gru_cond_mCG', **kwargs):
    """Note that, all the parameters with _att suffix are shared across
    decoders."""

    params = param_init_gru(params, prefix, nin=nin, dim=dim)

    # context to LSTM [C]
    Wc = norm_weight(dimctx, dim*2)
    params[_p(prefix, 'Wc')] = Wc

    # context to LSTM [Cz, Cr]
    Wcx = norm_weight(dimctx, dim)
    params[_p(prefix, 'Wcx')] = Wcx

    # attention: prev -> hidden [Wi_dec]
    Wi_dec = norm_weight(nin, dimctx)
    params[_p(prefix, 'Wi_dec')] = Wi_dec

    # attention: LSTM -> hidden [Wa]
    Wd_att = norm_weight(dim, dimctx)
    params[_p(prefix, 'Wd_dec')] = Wd_att

    # attention: hidden bias [ba]
    b_att = numpy.zeros((dimatt,)).astype('float32')
    params[_p(prefix, 'b_att')] = b_att

    # attention: [va]
    U_att = norm_weight(dimatt, 1)
    params[_p(prefix, 'U_att')] = U_att

    # attention: encoder latent space embedder, [Le_att]
    Le_att = norm_weight(dimctx, dimatt).astype('float32')
    params[_p(prefix, 'Le_att')] = Le_att

    # attention: decoder latent space embedder, [Ld_att]
    Ld_att = norm_weight(dimctx, dimatt).astype('float32')
    params[_p(prefix, 'Ld_att')] = Ld_att

    # attention: weighted averages to gru input, post weight [Wp_att]
    Wp_att = norm_weight(dimctx, dimctx, ortho=False).astype('float32')
    params[_p(prefix, 'Wp_att')] = Wp_att

    return params


def gru_cond_mCG_layer(
        tparams, state_below, mask=None, context=None, one_step=False,
        init_state=None, context_mask=None, prefix='gru_cond_mCG',
        lencoder_act='tanh', ldecoder_act='tanh', lctxproj_act='tanh',
        **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3d: #annotation x #sample x dim + num_encs + num_decs'

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # projected context
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_dec')])

    # embed encoders latent space into attention outside of the scan
    enc_lspace = eval(lencoder_act)(
        context.dot(tparams[_p(prefix, 'Le_att')]) +
        tparams[_p(prefix, 'b_att')])

    # m_     : mask
    # x_     : state_below_ from target embeddings
    # xx_    : state_belowx from target embeddings
    # xc_    : state_belowc from source context
    # h_     : previous hidden state
    # ctx_   : previous weighted averages
    # alpha_ : previous weights
    # enc_ls : enc_lspace + b_att - encoder latent space
    # cc_    : context
    #              |    sequences   |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_,
                    enc_ls, cc_,
                    U, Wc, Wd_dec, U_att, Ux, Wcx, Ld_att, Wp_att):
        # attention
        # previous gru hidden state s_{i-1}
        pstate_ = tensor.dot(h_, Wd_dec)
        dec_lspace = eval(ldecoder_act)(xc_ + pstate_)

        # transform decoder latent space with shared attention parameters
        dec_lspace = dec_lspace.dot(Ld_att)

        # combine encoder and decoder latent spaces and compute alignments
        pctx__ = tensor.tanh(dec_lspace[None, :, :] + enc_ls)
        alpha = tensor.dot(pctx__, U_att)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

        # stabilize energies first
        max_alpha = alpha.max(axis=0)
        alpha = alpha - max_alpha
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        # project current context using a shared post projection
        ctx_ = eval(lctxproj_act)(ctx_.dot(Wp_att))

        # gru unit for final hidden state
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_dec')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'Ld_att')],
                   tparams[_p(prefix, 'Wp_att')]]

    pselectors = []

    if one_step:
        rval = _step(*(seqs +
                       [init_state] +
                       [enc_lspace, context] + shared_vars + pselectors))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state, None, None],
            non_sequences=[enc_lspace, context] + shared_vars + pselectors,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True)
    return rval


def param_init_ff_attend_merger(params, dimctx=None, prefix='ff_attend_merger',
                                nin=None, nout=None, ortho=True, add_bias=True,
                                attend_merge_op=None, bundle_states=None,
                                **kwargs):
    if attend_merge_op == 'gru':
        pass
    elif attend_merge_op == 'att':
        params[_p(prefix, 'U')] = norm_weight(dimctx, 1)
        params[_p(prefix, 'W')] = norm_weight(dimctx, dimctx)
        params[_p(prefix, 'b')] = numpy.zeros((dimctx,)).astype('float32')
    elif attend_merge_op == 'mean-concat':
        numSrc = len(list(set(bundle_states)))
        params[_p(prefix, 'W')] = norm_weight(dimctx*numSrc, dimctx)
        params[_p(prefix, 'b')] = numpy.zeros((dimctx,)).astype('float32')
    return params


def ff_attend_merger_layer(tparams, states_below, prefix='ff_attend_merger',
                           attend_merge_act='lambda x: x', add_bias=True,
                           attend_merge_op='mean', bundle_states=None,
                           **kwargs):
    alpha = None
    if attend_merge_op == 'mean':
        preact = states_below[0]
        for i in range(1, len(states_below)):
            preact += states_below[i]
        preact /= tensor.cast(len(states_below), theano.config.floatX)
    elif attend_merge_op == 'sum':
        preact = states_below[0]
        for i in range(1, len(states_below)):
            preact += states_below[i]
    elif attend_merge_op == 'att':
        tbd = tensor.stack(states_below)  # time x batch x dim
        ptbd = tensor.tanh(
            tensor.dot(tbd, tparams[_p(prefix, 'W')]) +
            tparams[_p(prefix, 'b')])
        alpha = tensor.dot(ptbd, tparams[_p(prefix, 'U')])
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])
        alpha = tensor.exp(alpha)
        alpha = alpha / alpha.sum(0, keepdims=True)
        preact = (tbd * alpha[:, :, None]).sum(0)  # weighted average
    elif attend_merge_op == 'max':
        preact = states_below[0]
        for i in range(1, len(states_below)):
            preact = tensor.maximum(preact, states_below[i])

    elif attend_merge_op == 'mean-concat':

        if bundle_states is None:
            raise ValueError('bundle_states cannot be None!')

        def take_mean(states):
            _preact = states[0]
            for i in range(1, len(states)):
                _preact += states[i]
            _preact /= tensor.cast(len(states), theano.config.floatX)
            return _preact

        uniq = list(set(bundle_states))
        bundle_states = numpy.asarray(bundle_states)
        mean_bundled_states = OrderedDict()

        # take the mean over bundles
        for bidx in uniq:
            mean_bundled_states[bidx] = \
                take_mean([states_below[ii]
                           for ii, bb in enumerate(bundle_states==bidx)])

        # concatenate each bundle
        concatenated = concatenate(mean_bundled_states.values(),
                                   axis=states_below[0].ndim-1)

        # project
        preact = get_layer('ff')[1](
            tparams, concatenated, prefix=prefix,
            activ='linear', add_bias=True)

    else:
        raise ValueError('Unrecognized merge op!')
    return eval(attend_merge_act)(preact), alpha


def param_init_ff_initial_state_merger(
        params, prefix='ff_init_state_merger', nin=None, nout=None, ortho=True,
        add_bias=True, init_merge_op=None, bundle_states=None, **kwargs):
    if init_merge_op == 'mean-concat':
        numSrc = len(list(set(bundle_states)))
        params[_p(prefix, 'W')] = norm_weight(nin*numSrc, nin)
        params[_p(prefix, 'b')] = numpy.zeros((nin,)).astype('float32')
    return params


def ff_initial_state_merger_layer(
        tparams, states_below, prefix='ff_init_state_merger',
        init_merge_act='lambda x: x', add_bias=True, init_merge_op='mean',
        bundle_states=None, **kwargs):
    if init_merge_op == 'mean':
        preact = states_below[0]
        for i in range(1, len(states_below)):
            preact += states_below[i]
        preact /= tensor.cast(len(states_below), theano.config.floatX)
    elif init_merge_op == 'sum':
        preact = states_below[0]
        for i in range(1, len(states_below)):
            preact += states_below[i]
    elif init_merge_op == 'max':
        preact = states_below[0]
        for i in range(1, len(states_below)):
            preact = tensor.maximum(preact, states_below[i])
    elif init_merge_op == 'mean-concat':

        if bundle_states is None:
            raise ValueError('bundle_states cannot be None!')

        def take_mean(states):
            _preact = states[0]
            for i in range(1, len(states)):
                _preact += states[i]
            _preact /= tensor.cast(len(states), theano.config.floatX)
            return _preact

        uniq = list(set(bundle_states))
        bundle_states = numpy.asarray(bundle_states)
        mean_bundled_states = OrderedDict()

        # take the mean over bundles
        for bidx in uniq:
            mean_bundled_states[bidx] = \
                take_mean([states_below[ii]
                           for ii, bb in enumerate(bundle_states==bidx)])

        # concatenate each bundle
        concatenated = concatenate(mean_bundled_states.values(),
                                   axis=states_below[0].ndim-1)

        # project
        preact = get_layer('ff')[1](
            tparams, concatenated, prefix=prefix,
            activ='linear', add_bias=True)

    else:
        raise ValueError('Unrecognized merge op!')
    return eval(init_merge_act)(preact)


# LN-GRU layer
def param_init_lngru(params, prefix='lngru', nin=None, dim=None):
    """
    Gated Recurrent Unit (GRU) with LN
    """
    W = numpy.concatenate([norm_weight(nin, dim),
                           norm_weight(nin, dim)], axis=1)
    params[_p(prefix, 'W')] = W
    params[_p(prefix, 'b')] = numpy.zeros((2 * dim,)).astype('float32')
    U = numpy.concatenate([ortho_weight(dim),
                           ortho_weight(dim)], axis=1)
    params[_p(prefix, 'U')] = U

    Wx = norm_weight(nin, dim)
    params[_p(prefix, 'Wx')] = Wx
    Ux = ortho_weight(dim)
    params[_p(prefix, 'Ux')] = Ux
    params[_p(prefix, 'bx')] = numpy.zeros((dim,)).astype('float32')

    # LN parameters
    scale_add = 0.0
    scale_mul = 1.0
    params[_p(prefix, 'b1')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b2')] = scale_add * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 'b3')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b4')] = scale_add * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 's1')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's2')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 's3')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's4')] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    return params


def lngru_layer(tparams, state_below, prefix='lngru', mask=None, **kwargs):
    """
    Feedforward pass through GRU with LN
    """
    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    dim = tparams[_p(prefix, 'Ux')].shape[1]

    init_state = tensor.alloc(0., n_samples, dim)

    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    state_below_ = tensor.dot(
        state_below, tparams[_p(prefix, 'W')]) + tparams[_p(prefix, 'b')]
    state_belowx = tensor.dot(
        state_below, tparams[_p(prefix, 'Wx')]) + tparams[_p(prefix, 'bx')]

    def _step_slice(m_, x_, xx_, h_,
                    U, Ux,
                    b1, b2, b3, b4,
                    s1, s2, s3, s4,
                    *args):

        x_ = ln(x_, b1, s1)
        xx_ = ln(xx_, b2, s2)

        preact = tensor.dot(h_, U)
        preact = ln(preact, b3, s3)
        preact += x_

        r = tensor.nnet.sigmoid(_slice(preact, 0, dim))
        u = tensor.nnet.sigmoid(_slice(preact, 1, dim))

        preactx = tensor.dot(h_, Ux)
        preactx = ln(preactx, b4, s4)
        preactx = preactx * r
        preactx = preactx + xx_

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h

    seqs = [mask, state_below_, state_belowx]
    _step = _step_slice

    non_seqs = [tparams[_p(prefix, 'U')],
                tparams[_p(prefix, 'Ux')]]
    non_seqs += [tparams[_p(prefix, 'b1')],
                 tparams[_p(prefix, 'b2')],
                 tparams[_p(prefix, 'b3')],
                 tparams[_p(prefix, 'b4')]]
    non_seqs += [tparams[_p(prefix, 's1')],
                 tparams[_p(prefix, 's2')],
                 tparams[_p(prefix, 's3')],
                 tparams[_p(prefix, 's4')]]

    rval, updates = theano.scan(_step,
                                sequences=seqs,
                                outputs_info=[init_state],
                                non_sequences=non_seqs,
                                name=_p(prefix, '_layers'),
                                n_steps=nsteps,
                                profile=False,
                                strict=True)
    rval = [rval]
    return rval


def param_init_gru_cond_mCG_mSrc(params, nin, dim, dimctx, dimatt,
                                 prefix='grun_cond_mCG_mSrc',
                                 attend_merge_op='mean', **kwargs):
    params = param_init_gru_cond_mCG(
        params, nin, dim, dimctx, dimatt, prefix=prefix, **kwargs)

    params = get_layer('ff_attend_merger')[0](
        params, dimctx, prefix=_p(prefix, 'ff_attend_merger'),
        attend_merge_op=attend_merge_op, **kwargs)
    return params


def gru_cond_mCG_mSrc_layer(
        tparams, state_below, mask=None, contexts=None, one_step=False,
        init_state=None, context_masks=None, prefix='gru_cond_mCG_mSrc',
        lencoder_act='tanh', ldecoder_act='tanh', lctxproj_act='tanh',
        attend_merge_op=None, **kwargs):

    assert contexts, 'Contexts must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected context
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_dec')])

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # embed encoders latent space into attention outside of the scan
    enc_lspaces = [eval(lencoder_act)(
        contexts[eid].dot(tparams[_p(prefix, 'Le_att')]) +
        tparams[_p(prefix, 'b_att')]) for eid in contexts.keys()]

    # m_     : mask
    # x_     : state_below_ from target embeddings
    # xx_    : state_belowx from target embeddings
    # xc_    : state_belowc from source context
    # h_     : previous hidden state
    # ctx_   : previous weighted averages
    # alpha_ : previous weights
    # enc_ls : enc_lspace + b_att - encoder latent space
    # cc_    : context
    #              |    sequences   |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_,
                    U, Wc, Wd_dec, U_att, Ux, Wcx, Ld_att, Wp_att,
                    *args):

        # attention
        # previous gru hidden state s_{i-1}
        pstate_ = tensor.dot(h_, Wd_dec)
        dec_lspace = eval(ldecoder_act)(xc_ + pstate_)

        # transform decoder latent space with shared attention parameters
        dec_lspace = dec_lspace.dot(Ld_att)

        # iterate over encoder latent spaces
        alphas = []
        weighted_averages = []
        num_encs = len(contexts)
        for i in range(num_encs):

            # combine encoder and decoder latent spaces and compute alignments
            enc_ls = args[i]
            cc_ = args[i + num_encs]
            pctx__ = tensor.tanh(dec_lspace[None, :, :] + enc_ls)
            alpha = tensor.dot(pctx__, U_att)
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

            # stabilize energies first
            max_alpha = alpha.max(axis=0)
            alpha = alpha - max_alpha
            alpha = tensor.exp(alpha)
            if context_masks is not None:
                alpha = alpha * context_masks[i]
            alpha = alpha / alpha.sum(0, keepdims=True)
            ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

            # project current context using a shared post projection
            ctx_ = eval(lctxproj_act)(ctx_.dot(Wp_att))
            weighted_averages.append(ctx_)
            alphas.append(alpha.T)

        # our guy, the merger layer
        ctx_, merger_alpha = get_layer('ff_attend_merger')[1](
            tparams, weighted_averages, prefix=_p(prefix, 'ff_attend_merger'),
            attend_merge_op=attend_merge_op, **kwargs)

        # gru unit for final hidden state
        preact = tensor.dot(h_, U)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return (h, ctx_) + tuple(alphas)

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_dec')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'Ld_att')],
                   tparams[_p(prefix, 'Wp_att')]]

    pselectors = []
    non_seqs = (shared_vars + pselectors +
                enc_lspaces + contexts.values())
    if attend_merge_op == 'att':
        non_seqs += [tparams[_p(prefix, 'ff_attend_merger_U')],
                     tparams[_p(prefix, 'ff_attend_merger_W')],
                     tparams[_p(prefix, 'ff_attend_merger_b')]]
    elif attend_merge_op == 'mean-concat':
        non_seqs += [tparams[_p(prefix, 'ff_attend_merger_W')],
                     tparams[_p(prefix, 'ff_attend_merger_b')]]

    outputs_info = [init_state, None]
    outputs_info += [None for _ in contexts.values()]

    if one_step:
        rval = _step(*(seqs +
                       [init_state] +
                       non_seqs))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True)
    return rval


# Conditional GRU layer with Attention for multiple encoders - version 0.8
def param_init_lngru_cond_mCG(params, nin, dim, dimctx, dimatt,
                              prefix='lngru_cond_mCG', **kwargs):
    """Note that, all the parameters with _att suffix are shared across
    decoders."""

    params = param_init_gru_cond_mCG(
        params, nin, dim, dimctx, dimatt, prefix=prefix, **kwargs)

    # LN parameters
    scale_add = 0.0
    params[_p(prefix, 'b1')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b2')] = scale_add * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 'b3')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 'b4')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 'b5_att')] = scale_add * numpy.ones((1*dimatt)).astype(floatX)
    params[_p(prefix, 'b6_att')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 'b7')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b8')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b9')] = scale_add * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 'b10')] = scale_add * numpy.ones((1*dim)).astype(floatX)

    scale_mul = 1.0
    params[_p(prefix, 's1')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's2')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 's3')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 's4')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 's5_att')] = scale_mul * numpy.ones((1*dimatt)).astype(floatX)
    params[_p(prefix, 's6_att')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 's7')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's8')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's9')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 's10')] = scale_mul * numpy.ones((1*dim)).astype(floatX)

    return params


def lngru_cond_mCG_layer(
        tparams, state_below, mask=None, context=None, one_step=False,
        init_state=None, context_mask=None, prefix='gru_cond_mCG',
        lencoder_act='tanh', ldecoder_act='tanh', lctxproj_act='tanh',
        **kwargs):

    assert context, 'Context must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    # projected context
    assert context.ndim == 3, \
        'Context must be 3d: #annotation x #sample x dim + num_encs + num_decs'

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # projected context
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_dec')])

    # embed encoders latent space into attention outside of the scan
    # TODO: find out a way to apply ln here
    enc_lspace = eval(lencoder_act)(
        context.dot(tparams[_p(prefix, 'Le_att')]) +
        tparams[_p(prefix, 'b_att')])

    # m_     : mask
    # x_     : state_below_ from target embeddings
    # xx_    : state_belowx from target embeddings
    # xc_    : state_belowc from source context
    # h_     : previous hidden state
    # ctx_   : previous weighted averages
    # alpha_ : previous weights
    # enc_ls : enc_lspace + b_att - encoder latent space
    # cc_    : context
    #              |    sequences   |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_,
                    enc_ls, cc_,
                    U, Wc, Wd_dec, U_att, Ux, Wcx, Ld_att, Wp_att,
                    b1, b2, b3, b4, b5_att, b6_att, b7, b8, b9, b10,
                    s1, s2, s3, s4, s5_att, s6_att, s7, s8, s9, s10):

        x_ = ln(x_, b1, s1)
        xx_ = ln(xx_, b2, s2)
        xc_ = ln(xc_, b3, s3)

        # attention
        # previous gru hidden state s_{i-1}
        pstate_ = tensor.dot(h_, Wd_dec)
        pstate_ = ln(pstate_, b4, s4)
        dec_lspace = eval(ldecoder_act)(xc_ + pstate_)

        # transform decoder latent space with shared attention parameters
        dec_lspace = dec_lspace.dot(Ld_att)
        dec_lspace = ln(dec_lspace, b5_att, s5_att)

        # combine encoder and decoder latent spaces and compute alignments
        pctx__ = tensor.tanh(dec_lspace[None, :, :] + enc_ls)
        alpha = tensor.dot(pctx__, U_att)
        alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

        # stabilize energies first
        max_alpha = alpha.max(axis=0)
        alpha = alpha - max_alpha
        alpha = tensor.exp(alpha)
        if context_mask:
            alpha = alpha * context_mask
        alpha = alpha / alpha.sum(0, keepdims=True)
        ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

        # project current context using a shared post projection
        ctx_ = ctx_.dot(Wp_att)
        ctx_ = ln(ctx_, b6_att, s6_att)
        ctx_ = eval(lctxproj_act)(ctx_)

        # gru unit for final hidden state
        preact = tensor.dot(h_, U)
        preact = ln(preact, b7, s7)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = ln(preact, b8, s8)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx = ln(preactx, b9, s9)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)
        preactx = ln(preactx, b10, s10)

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return h, ctx_, alpha.T

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_dec')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'Ld_att')],
                   tparams[_p(prefix, 'Wp_att')]]

    shared_vars += [tparams[_p(prefix, 'b1')],
                    tparams[_p(prefix, 'b2')],
                    tparams[_p(prefix, 'b3')],
                    tparams[_p(prefix, 'b4')],
                    tparams[_p(prefix, 'b5_att')],
                    tparams[_p(prefix, 'b6_att')],
                    tparams[_p(prefix, 'b7')],
                    tparams[_p(prefix, 'b8')],
                    tparams[_p(prefix, 'b9')],
                    tparams[_p(prefix, 'b10')]]

    shared_vars += [tparams[_p(prefix, 's1')],
                    tparams[_p(prefix, 's2')],
                    tparams[_p(prefix, 's3')],
                    tparams[_p(prefix, 's4')],
                    tparams[_p(prefix, 's5_att')],
                    tparams[_p(prefix, 's6_att')],
                    tparams[_p(prefix, 's7')],
                    tparams[_p(prefix, 's8')],
                    tparams[_p(prefix, 's9')],
                    tparams[_p(prefix, 's10')]]

    pselectors = []

    if one_step:
        rval = _step(*(seqs +
                       [init_state] +
                       [enc_lspace, context] + shared_vars + pselectors))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=[init_state, None, None],
            non_sequences=[enc_lspace, context] + shared_vars + pselectors,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True)
    return rval


def param_init_lngru_cond_mCG_mSrc(params, nin, dim, dimctx, dimatt,
                                   prefix='grun_cond_mCG_mSrc',
                                   attend_merge_op='mean', **kwargs):
    params = param_init_gru_cond_mCG(
        params, nin, dim, dimctx, dimatt, prefix=prefix, **kwargs)

    params = get_layer('ff_attend_merger')[0](
        params, dimctx, prefix=_p(prefix, 'ff_attend_merger'),
        attend_merge_op=attend_merge_op, **kwargs)

    # LN parameters
    scale_add = 0.0
    params[_p(prefix, 'b1')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b2')] = scale_add * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 'b3')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 'b4')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 'b5_att')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 'b6_att')] = scale_add * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 'b7')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b8')] = scale_add * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 'b9')] = scale_add * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 'b10')] = scale_add * numpy.ones((1*dim)).astype(floatX)

    scale_mul = 1.0
    params[_p(prefix, 's1')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's2')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 's3')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 's4')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 's5_att')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 's6_att')] = scale_mul * numpy.ones((1*dimctx)).astype(floatX)
    params[_p(prefix, 's7')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's8')] = scale_mul * numpy.ones((2*dim)).astype(floatX)
    params[_p(prefix, 's9')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    params[_p(prefix, 's10')] = scale_mul * numpy.ones((1*dim)).astype(floatX)
    return params


def lngru_cond_mCG_mSrc_layer(
        tparams, state_below, mask=None, contexts=None, one_step=False,
        init_state=None, context_masks=None, prefix='gru_cond_mCG_mSrc',
        lencoder_act='tanh', ldecoder_act='tanh', lctxproj_act='tanh',
        attend_merge_op=None, attend_merge_act='linear', bundle_states=None,
        **kwargs):

    assert contexts, 'Contexts must be provided'

    if one_step:
        assert init_state, 'previous state must be provided'

    nsteps = state_below.shape[0]
    if state_below.ndim == 3:
        n_samples = state_below.shape[1]
    else:
        n_samples = 1

    # mask
    if mask is None:
        mask = tensor.alloc(1., state_below.shape[0], 1)

    dim = tparams[_p(prefix, 'Wcx')].shape[1]

    # initial/previous state
    if init_state is None:
        init_state = tensor.alloc(0., n_samples, dim)

    def _slice(_x, n, dim):
        if _x.ndim == 3:
            return _x[:, :, n*dim:(n+1)*dim]
        return _x[:, n*dim:(n+1)*dim]

    # projected context
    state_belowc = tensor.dot(state_below, tparams[_p(prefix, 'Wi_dec')])

    # projected x
    state_belowx = tensor.dot(state_below, tparams[_p(prefix, 'Wx')]) +\
        tparams[_p(prefix, 'bx')]
    state_below_ = tensor.dot(state_below, tparams[_p(prefix, 'W')]) +\
        tparams[_p(prefix, 'b')]

    # embed encoders latent space into attention outside of the scan
    enc_lspaces = [eval(lencoder_act)(
        contexts[eid].dot(tparams[_p(prefix, 'Le_att')]) +
        tparams[_p(prefix, 'b_att')]) for eid in contexts.keys()]

    # m_     : mask
    # x_     : state_below_ from target embeddings
    # xx_    : state_belowx from target embeddings
    # xc_    : state_belowc from source context
    # h_     : previous hidden state
    # ctx_   : previous weighted averages
    # alpha_ : previous weights
    # enc_ls : enc_lspace + b_att - encoder latent space
    # cc_    : context
    #              |    sequences   |  outputs-info   | non-seqs ...
    def _step_slice(m_, x_, xx_, xc_, h_,
                    U, Wc, Wd_dec, U_att, Ux, Wcx, Ld_att, Wp_att,
                    b1, b2, b3, b4, b5_att, b6_att, b7, b8, b9, b10,
                    s1, s2, s3, s4, s5_att, s6_att, s7, s8, s9, s10,
                    *args):

        x_ = ln(x_, b1, s1)
        xx_ = ln(xx_, b2, s2)
        xc_ = ln(xc_, b3, s3)

        # attention
        # previous gru hidden state s_{i-1}
        pstate_ = tensor.dot(h_, Wd_dec)
        pstate_ = ln(pstate_, b4, s4)
        dec_lspace = eval(ldecoder_act)(xc_ + pstate_)

        # transform decoder latent space with shared attention parameters
        dec_lspace = dec_lspace.dot(Ld_att)
        dec_lspace = ln(dec_lspace, b5_att, s5_att)

        # iterate over encoder latent spaces
        alphas = []
        weighted_averages = []
        num_encs = len(contexts)
        for i in range(num_encs):

            # combine encoder and decoder latent spaces and compute alignments
            enc_ls = args[i]
            cc_ = args[i + num_encs]
            pctx__ = tensor.tanh(dec_lspace[None, :, :] + enc_ls)
            alpha = tensor.dot(pctx__, U_att)
            alpha = alpha.reshape([alpha.shape[0], alpha.shape[1]])

            # stabilize energies first
            max_alpha = alpha.max(axis=0)
            alpha = alpha - max_alpha
            alpha = tensor.exp(alpha)
            if context_masks is not None:
                alpha = alpha * context_masks[i]
            alpha = alpha / alpha.sum(0, keepdims=True)
            ctx_ = (cc_ * alpha[:, :, None]).sum(0)  # current context

            # project current context using a shared post projection
            ctx_ = ctx_.dot(Wp_att)
            ctx_ = ln(ctx_, b6_att, s6_att)
            ctx_ = eval(lctxproj_act)(ctx_)
            weighted_averages.append(ctx_)
            alphas.append(alpha.T)

        # our guy, the merger layer
        # TODO: add ln to merger layer
        ctx_, merger_alpha = get_layer('ff_attend_merger')[1](
            tparams, weighted_averages, prefix=_p(prefix, 'ff_attend_merger'),
            attend_merge_op=attend_merge_op, bundle_states=bundle_states)

        # gru unit for final hidden state
        preact = tensor.dot(h_, U)
        preact = ln(preact, b7, s7)
        preact += x_
        preact += tensor.dot(ctx_, Wc)
        preact = ln(preact, b8, s8)
        preact = tensor.nnet.sigmoid(preact)

        r = _slice(preact, 0, dim)
        u = _slice(preact, 1, dim)

        preactx = tensor.dot(h_, Ux)
        preactx = ln(preactx, b9, s9)
        preactx *= r
        preactx += xx_
        preactx += tensor.dot(ctx_, Wcx)
        preactx = ln(preactx, b10, s10)

        h = tensor.tanh(preactx)

        h = u * h_ + (1. - u) * h
        h = m_[:, None] * h + (1. - m_)[:, None] * h_

        return (h, ctx_) + tuple(alphas)

    seqs = [mask, state_below_, state_belowx, state_belowc]
    _step = _step_slice

    shared_vars = [tparams[_p(prefix, 'U')],
                   tparams[_p(prefix, 'Wc')],
                   tparams[_p(prefix, 'Wd_dec')],
                   tparams[_p(prefix, 'U_att')],
                   tparams[_p(prefix, 'Ux')],
                   tparams[_p(prefix, 'Wcx')],
                   tparams[_p(prefix, 'Ld_att')],
                   tparams[_p(prefix, 'Wp_att')]]

    # add all the ln-biases
    shared_vars += [tparams[_p(prefix, 'b1')],
                    tparams[_p(prefix, 'b2')],
                    tparams[_p(prefix, 'b3')],
                    tparams[_p(prefix, 'b4')],
                    tparams[_p(prefix, 'b5_att')],
                    tparams[_p(prefix, 'b6_att')],
                    tparams[_p(prefix, 'b7')],
                    tparams[_p(prefix, 'b8')],
                    tparams[_p(prefix, 'b9')],
                    tparams[_p(prefix, 'b10')]]

    # add all the ln-scales
    shared_vars += [tparams[_p(prefix, 's1')],
                    tparams[_p(prefix, 's2')],
                    tparams[_p(prefix, 's3')],
                    tparams[_p(prefix, 's4')],
                    tparams[_p(prefix, 's5_att')],
                    tparams[_p(prefix, 's6_att')],
                    tparams[_p(prefix, 's7')],
                    tparams[_p(prefix, 's8')],
                    tparams[_p(prefix, 's9')],
                    tparams[_p(prefix, 's10')]]

    pselectors = []
    non_seqs = (shared_vars + pselectors +
                enc_lspaces + contexts.values())
    if attend_merge_op == 'att':
        non_seqs += [tparams[_p(prefix, 'ff_attend_merger_U')],
                     tparams[_p(prefix, 'ff_attend_merger_W')],
                     tparams[_p(prefix, 'ff_attend_merger_b')]]
    elif attend_merge_op == 'mean-concat':
        non_seqs += [tparams[_p(prefix, 'ff_attend_merger_W')],
                     tparams[_p(prefix, 'ff_attend_merger_b')]]

    outputs_info = [init_state, None]
    outputs_info += [None for _ in contexts.values()]

    if one_step:
        rval = _step(*(seqs +
                       [init_state] +
                       non_seqs))
    else:
        rval, updates = theano.scan(
            _step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=_p(prefix, '_layers'),
            n_steps=nsteps,
            strict=True)
    return rval
