import logging
import copy
import numpy
import os
import time

from blocks.extensions import SimpleExtension

from collections import OrderedDict

from .utils import _p, is_multiSource, get_enc_ids

logger = logging.getLogger(__name__)


# generate sample for multi-source ensemble - only the target
# is matching among all the models, sources may differ.
def gen_sample_ensemble_mSrc(
        f_init_list, f_next_list, seqs_list, k=1, maxlen=30, minlen=-1,
        stochastic=True, argmax=False, eos_idx=0, ignore_unk=False, unk_idx=1,
        arithmetic=False, **kwargs):

    # Input x in multi-source is a list of inputs, note that the ordering
    # in that list, xs, is crucial and should match with training ordering.
    # This ordering is determined by enc_ids list and alphabetical.
    # So, seqs should be an ordered dictionary with the correct ordering.
    xs_list = [
        [numpy.array(seq).reshape([len(seq), 1]) for seq in seqs.values()]
        for seqs in seqs_list]
    enc_ids_list = [seqs.keys() for seqs in seqs_list]

    # number of models in ensemble
    tot_model = len(f_init_list)

    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    # start from one empty hyp
    sample = []
    sample_score = []
    sample_decalphas_list = [OrderedDict([(eid, list()) for eid in enc_ids])
                             for enc_ids in enc_ids_list]

    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]]  # live_k  don't do * like this!! pointers will be used!
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states_list = [[] for mi in range(tot_model)]

    # for different models, we only have one hyp
    ret_list = []
    for mi in range(tot_model):
        # call f_init first
        init_inps = xs_list[mi]
        ret_list.append(f_init_list[mi](*init_inps))

    ret_index = 0
    next_state_list = [amodel[ret_index] for amodel in ret_list]
    ret_index += 1

    ctx0_list = [amodel[ret_index:] for amodel in ret_list]
    ret_index += 1

    # we only have one hyp in the very beginning
    next_w_list = [-1 * numpy.ones((1,)).astype('int64')
                   for _ in range(tot_model)]

    for ii in xrange(maxlen):
        ctx_list = [[numpy.tile(ctx0, [live_k, 1]) for ctx0 in ctxs0]
                    for ctxs0 in ctx0_list]

        inps_list = [[next_w] + ctxs + [next_state]
                     for next_w, ctxs, next_state in
                     zip(next_w_list, ctx_list, next_state_list)]

        ret_list = [f_next(*inps) for inps, f_next in
                    zip(inps_list, f_next_list)]

        # ret_list: model first, tot_model * outputs
        ret_index = 0
        next_p_list = [ret[ret_index] for ret in ret_list]
        ret_index += 1
        next_w_list = [ret[ret_index] for ret in ret_list]
        ret_index += 1
        next_state_list = [ret[ret_index] for ret in ret_list]
        ret_index += 1

        # Handle ensembles here
        num_models = len(next_p_list)
        if arithmetic:
            log_next_p = numpy.log(numpy.array(next_p_list).mean(axis=0))
        else:
            log_next_p = numpy.log(numpy.array(next_p_list)).sum(axis=0) / \
                num_models

        if stochastic:  # TODO complete this part latter
            if argmax:
                nw = next_p_list[0][0].argmax()
            else:
                nw = next_w_list[0][0]
            sample.append(nw)
            sample_score -= numpy.log(next_p_list[0][0, nw])
            if nw == 0:
                break
        else:
            """
            cand_scores = hyp_scores[:, None] -\
                numpy.log(next_p_list[0])/len(ret_list)
            for next_p in next_p_list[1:]:
                # each next_p is a matrix = beam_size * voc_size
                cand_scores -= (numpy.log(next_p)/len(ret_list))
            """
            cand_scores = hyp_scores[:, None] - log_next_p
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p_list[0].shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []  # translations
            # scores for all translations
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states_list = [[] for mi in range(tot_model)]

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])

                ###
                for mi in range(tot_model):
                    new_hyp_states_list[mi].append(
                        copy.copy(next_state_list[mi][ti]))

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states_list = [[] for mi in range(tot_model)]

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == 0:
                    if len(new_hyp_samples[idx]) >= minlen:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    for mi in range(tot_model):
                        hyp_states_list[mi].append(
                            new_hyp_states_list[mi][idx])

            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w_list = [numpy.array([w[-1] for w in hyp_samples])
                           for mi in range(tot_model)]
            next_state_list = [numpy.array(hyp_states)
                               for hyp_states in hyp_states_list]

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in xrange(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
    sample_decalphas_list = None
    return sample, sample_score, sample_decalphas_list


def gen_sample(f_init, f_next, x, k=1, maxlen=30, stochastic=True,
               argmax=False, eos_idx=0, cond_init_trg=False, ignore_unk=False,
               minlen=1, unk_idx=1, f_next_state=None, aux_lm=None,
               return_alphas=False, xs=None):
    if k > 1:
        assert not stochastic, \
            'Beam search does not support stochastic sampling'

    sample = []
    sample_score = []
    sample_decalphas = []
    if stochastic:
        sample_score = 0

    live_k = 1
    dead_k = 0

    hyp_samples = [[]] * live_k
    hyp_decalphas = []
    hyp_scores = numpy.zeros(live_k).astype('float32')
    hyp_states = []

    # multi-source
    if xs is not None:
        inp_xs = xs
    else:
        inp_xs = [x]

    init_inps = inp_xs

    ret = f_init(*init_inps)
    next_state, ctx0 = ret[0], ret[1]
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in range(maxlen):
        ctx = numpy.tile(ctx0, [live_k, 1])

        inps = [next_w, ctx, next_state]

        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]

        if return_alphas:
            next_decalpha = ret.pop(0)

        if stochastic:
            if argmax:
                nw = next_p[0].argmax()
            else:
                nw = next_w[0]
            sample.append(nw)
            sample_score -= numpy.log(next_p[0, nw])
            if nw == eos_idx:
                break
        else:
            log_probs = numpy.log(next_p)

            # Adjust log probs according to search restrictions
            if ignore_unk:
                log_probs[:, unk_idx] = -numpy.inf
            if ii < minlen:
                log_probs[:, eos_idx] = -numpy.inf

            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(k-dead_k)]

            voc_size = next_p.shape[1]
            trans_indices = ranks_flat / voc_size
            word_indices = ranks_flat % voc_size
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
            new_hyp_states = []
            new_hyp_decalphas = []

            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append(copy.copy(next_state[ti]))

                if return_alphas:
                    tmp_decalphas = []
                    if ii > 0:
                        tmp_decalphas = copy.copy(hyp_decalphas[ti])
                    tmp_decalphas.append(next_decalpha[ti])
                    new_hyp_decalphas.append(tmp_decalphas)

            # check the finished samples
            new_live_k = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []
            hyp_decalphas = []

            for idx in range(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == eos_idx:
                    sample.append(new_hyp_samples[idx])
                    sample_score.append(new_hyp_scores[idx])
                    if return_alphas:
                        sample_decalphas.append(new_hyp_decalphas[idx])
                    dead_k += 1
                else:
                    new_live_k += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])
                    if return_alphas:
                        hyp_decalphas.append(new_hyp_decalphas[idx])
            hyp_scores = numpy.array(hyp_scores)
            live_k = new_live_k

            if new_live_k < 1:
                break
            if dead_k >= k:
                break

            next_w = numpy.array([w[-1] for w in hyp_samples])
            next_state = numpy.array(hyp_states)

    if not stochastic:
        # dump every remaining one
        if live_k > 0:
            for idx in range(live_k):
                sample.append(hyp_samples[idx])
                sample_score.append(hyp_scores[idx])
                if return_alphas:
                    sample_decalphas.append(hyp_decalphas[idx])

    if not return_alphas:
        return numpy.array(sample), numpy.array(sample_score)
    return numpy.array(sample), numpy.array(sample_score), \
        numpy.array(sample_decalphas)


class SamplingBase(object):

    def _get_attr_rec(self, obj, attr):
        return self._get_attr_rec(getattr(obj, attr), attr) \
            if hasattr(obj, attr) else obj

    def _get_true_length(self, seq, eos_idx):
        try:
            return seq.tolist().index(eos_idx) + 1
        except ValueError:
            return len(seq)

    def _oov_to_unk(self, seq):
        return [x if x < self.src_vocab_size else self.unk_idx
                for x in seq]

    def _parse_input(self, line, eos_idx):
        seqin = line.split()
        seqlen = len(seqin)
        seq = numpy.zeros(seqlen+1, dtype='int64')
        for idx, sx in enumerate(seqin):
            seq[idx] = self.vocab.get(sx, self.unk_idx)
            if seq[idx] >= self.src_vocab_size:
                seq[idx] = self.unk_idx
        seq[-1] = eos_idx
        return seq

    def _idx_to_word(self, seq, ivocab):
        return " ".join([ivocab.get(idx, "<UNK>") for idx in seq])

    def _idx2word(self, seq, ivocab, eos_idx):
        input_length = self._get_true_length(seq, eos_idx)
        return self._idx_to_word(seq[:input_length], ivocab)

    def _get_true_seq(self, seq, eos_idx):
        return seq[:self._get_true_length(seq, eos_idx)]

    def _make_matrix(self, arr):
        if arr.ndim >= 2:
            return arr
        return arr[None, :]


class Sampler(SimpleExtension, SamplingBase):
    """Samples from computation graph

        Does not use peeked batches
    """

    def __init__(self, f_init, f_next, data_stream, num_samples=1,
                 src_vocab=None, trg_vocab=None, src_ivocab=None,
                 trg_ivocab=None, enc_id=0, dec_id=0, src_eos_idx=-1,
                 trg_eos_idx=-1, cond_init_trg=False, f_next_state=None,
                 **kwargs):
        super(Sampler, self).__init__(**kwargs)
        self.f_init = f_init
        self.f_next = f_next
        self.f_next_state = f_next_state
        self.data_stream = data_stream
        self.num_samples = num_samples
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_ivocab = src_ivocab
        self.trg_ivocab = trg_ivocab
        self.src_eos_idx = src_eos_idx
        self.trg_eos_idx = trg_eos_idx
        self.cond_init_trg = cond_init_trg
        self.enc_id = enc_id
        self.dec_id = dec_id
        self._synced = False
        if is_multiSource(enc_id):
            self.sampling_fn = gen_sample_ensemble_mSrc
            self.src_vocabs = None
        else:
            self.sampling_fn = gen_sample

    def _do_singleSource(self, *args):
        batch = args[0]

        # Get current model parameters
        if not self._synced:
            sources = self._get_attr_rec(
                self.main_loop.data_stream.streams[_p(self.enc_id,
                                                      self.dec_id)],
                'data_stream')
            self.sources = sources
            self._synced = True

        batch = self.main_loop.data_stream\
            .get_batch_with_stream_id(_p(self.enc_id, self.dec_id))

        batch_size = batch['source'].shape[1]

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocab:
            self.src_vocab = self.sources.data_streams[0].dataset.dictionary
        if not self.trg_vocab:
            self.trg_vocab = self.sources.data_streams[1].dataset.dictionary
        if not self.src_ivocab:
            self.src_ivocab = {v: k for k, v in self.src_vocab.items()}
            self.src_ivocab[self.src_eos_idx] = '</S>'
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
            self.trg_ivocab[self.trg_eos_idx] = '</S>'

        sample_idx = numpy.random.choice(
            batch_size, self.num_samples, replace=False)
        src_batch = batch['source']
        trg_batch = batch['target']

        input_ = src_batch[:, sample_idx]
        target_ = trg_batch[:, sample_idx]

        # Sample
        outputs = [list() for _ in sample_idx]
        costs = [list() for _ in sample_idx]

        for i, idx in enumerate(sample_idx):
            outputs[i], costs[i] = self.sampling_fn(
                self.f_init, self.f_next, eos_idx=self.trg_eos_idx,
                x=self._get_true_seq(input_[:, i], self.src_eos_idx)[:, None],
                k=1, maxlen=30, stochastic=True, argmax=False,
                cond_init_trg=self.cond_init_trg,
                f_next_state=self.f_next_state)

        print ""
        logger.info("Sampling from computation graph[{}-{}]"
                    .format(self.enc_id, self.dec_id))
        for i in range(len(outputs)):
            input_length = self._get_true_length(input_[:, i],
                                                 self.src_eos_idx)
            target_length = self._get_true_length(target_[:, i],
                                                  self.trg_eos_idx)
            sample_length = self._get_true_length(outputs[i],
                                                  self.trg_eos_idx)

            print "Input : ", self._idx_to_word(input_[:, i][:input_length],
                                                self.src_ivocab)
            print "Target: ", self._idx_to_word(target_[:, i][:target_length],
                                                self.trg_ivocab)
            print "Sample: ", self._idx_to_word(outputs[i][:sample_length],
                                                self.trg_ivocab)
            print "Sample cost: ", costs[i].sum()
            print ""

    def _do_multiSource(self, *args):
        batch = args[0]

        # Get current model parameters
        if not self._synced:
            sources = self._get_attr_rec(
                self.main_loop.data_stream.streams[_p(self.enc_id,
                                                      self.dec_id)],
                'data_stream')
            self.sources = sources
            self._synced = True

        batch = self.main_loop.data_stream\
            .get_batch_with_stream_id(_p(self.enc_id, self.dec_id))

        batch_size = batch['source0'].shape[1]
        enc_ids = get_enc_ids(self.enc_id)

        # Load vocabularies and invert if necessary
        # WARNING: Source and target indices from data stream
        #  can be different
        if not self.src_vocabs:
            self.src_vocabs = OrderedDict()
            self.src_ivocabs = OrderedDict()
            for eidx, eid in enumerate(enc_ids):
                self.src_vocabs[eid] =\
                    self.sources.data_streams[eidx].dataset.dictionary
                self.src_ivocabs[eid] = {v: k for k, v in
                                         self.src_vocabs[eid].items()}
                self.src_ivocabs[eid][self.src_eos_idx] = '</S>'

        if not self.trg_vocab:
            self.trg_vocab = self.sources.data_streams[-1].dataset.dictionary
        if not self.trg_ivocab:
            self.trg_ivocab = {v: k for k, v in self.trg_vocab.items()}
            self.trg_ivocab[self.trg_eos_idx] = '</S>'

        sample_idx = numpy.random.choice(
            batch_size, self.num_samples, replace=False)

        inputs = OrderedDict()
        for eidx, eid in enumerate(enc_ids):
            inputs[eid] = batch['source%d' % eidx][:, sample_idx]

        target_ = batch['target'][:, sample_idx]

        # Sample
        outputs = [list() for _ in sample_idx]
        costs = [list() for _ in sample_idx]

        for i, idx in enumerate(sample_idx):
            seqs = OrderedDict(
                [(eid, self._get_true_seq(
                    inp[:, i], self.src_eos_idx)[:, None])
                 for eid, inp in inputs.items()])
            maxlen = max([len(val) for key, val in seqs.items()])
            outputs[i], costs[i], alphas = self.sampling_fn(
                [self.f_init], [self.f_next], [seqs],
                eos_idx=self.trg_eos_idx, k=1, maxlen=maxlen*2,
                stochastic=True, argmax=False,
                cond_init_trg=self.cond_init_trg)

        print ""
        logger.info("Sampling from computation graph[{}-{}]"
                    .format(self.enc_id, self.dec_id))
        for i in range(len(outputs)):
            for eid in enc_ids:
                print "Input[{}]: ".format(eid), self._idx2word(
                      inputs[eid][:, i], self.src_ivocabs[eid],
                      self.src_eos_idx)

            print "Target   : ", self._idx2word(
                  target_[:, i], self.trg_ivocab, self.trg_eos_idx)
            print "Sample   : ", self._idx2word(
                  numpy.asarray(outputs[i]), self.trg_ivocab,
                  self.trg_eos_idx)
            print "Sample cost: ", costs[i].sum()
            print ""

    def do(self, which_callback, *args):
        if is_multiSource(self.enc_id):
            self._do_multiSource(*args)
        else:
            self._do_singleSource(*args)


class ModelInfo:
    def __init__(self, bleu_score, path=None, enc_id=None, dec_id=None):
        self.bleu_score = bleu_score
        self.enc_id = enc_id if enc_id is not None else ''
        self.dec_id = dec_id if dec_id is not None else ''
        self.path = self._generate_path(path) if path else None

    def _generate_path(self, path):
        return os.path.join(
            path, 'best_bleu_model{}_{}_{}_BLEU{:.2f}.npz'.format(
                self.enc_id, self.dec_id, int(time.time()), self.bleu_score))
