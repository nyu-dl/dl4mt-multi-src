import argparse
import copy
import logging
import numpy
import os
import re
import pprint
import theano
import time

from collections import OrderedDict
from itertools import izip
from multiprocessing import Process, Queue

from theano import tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import config as configuration

from mcg.models import EncoderDecoder, MultiEncoder, MultiDecoder
from mcg.sampling import gen_sample_ensemble_mSrc
from mcg.utils import (get_enc_dec_ids, get_enc_dec_ids_mSrc, p_,
                       seqs2words, words2seqs_multi_irregular,
                       is_multiSource, load_vocab, invert_vocab)

from multiprocessing import Process, Queue
from subprocess import Popen, PIPE

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('translate')


def calculate_bleu(bleu_script, trans, gold):
    multibleu_cmd = ['perl', bleu_script, gold, '<']
    mb_subprocess = Popen(multibleu_cmd, stdin=PIPE, stdout=PIPE)
    print >> mb_subprocess.stdin, '\n'.join(trans)
    mb_subprocess.stdin.flush()
    mb_subprocess.stdin.close()
    stdout = mb_subprocess.stdout.readline()
    logger.info(stdout)
    out_parse = re.match(r'BLEU = [-.0-9]+', stdout)
    assert out_parse is not None
    bleu_score = float(out_parse.group()[6:])
    mb_subprocess.terminate()
    return bleu_score


def _translate(seqs_list, f_init_list, f_next_list, trg_eos_idx,
               k, **kwargs):

    maxlen = max([max([len(el) for el in seq.values()]) for seq in seqs_list])
    sample, score, _ = gen_sample_ensemble_mSrc(
        f_init_list, f_next_list, seqs_list=seqs_list, eos_idx=trg_eos_idx,
        k=k, maxlen=2*maxlen, stochastic=False, argmax=False, arithmetic=True,
        **kwargs)

    # length normalization
    lengths = numpy.array([len(s) for s in sample])
    score = score / lengths
    sidx = numpy.argmin(score)

    return sample[sidx], score[sidx]


def translate_model(queue, rqueue, pid, f_inits_list, f_nexts_list,
                    trg_eos_idx, k, **kwargs):

    while True:
        req = queue.get()
        if req is None:
            break

        idx, seqs_list = req[0], req[1]
        print pid, '-', idx
        seq, scores = _translate(
            seqs_list, f_inits_list, f_nexts_list, trg_eos_idx, k)

        rqueue.put((idx, seq, scores))

    return


def add_single_pairs(enc_ids):
    enc_ids_n = copy.deepcopy(enc_ids)
    for enc_id in enc_ids:
        for eid in enc_id.split('.'):
            if eid not in enc_ids:
                enc_ids_n.append(eid)
    return enc_ids_n


def main(configs, models, val_sets, output_file, n_process=5, chr_level=False,
         cgs_to_translate=None, gold_file=None, bleu_script=None, beam_size=7):

    # Translate only the chosen cgs if they are valid
    if cgs_to_translate is None:
        raise ValueError('cgs-to-translate cannot be None')

    # Check if computational graphs are valid
    for cidx, config in enumerate(configs):
        enc_ids, dec_ids = get_enc_dec_ids(config['cgs'])
        enc_ids = add_single_pairs(enc_ids)
        for cg_name in cgs_to_translate.values()[cidx]:
            if cg_name not in config['cgs']:
                eids = p_(cg_name)[0].split('.')
                dids = p_(cg_name)[1].split('.')
                if not all([eid in enc_ids for eid in eids]) or \
                        not all([did in dec_ids for did in dids]):
                    raise ValueError(
                        'Zero shot is NOT valid for [{}]'.format(cg_name))
                logger.info('Zero shot is valid for [{}]'.format(cg_name))
                config['cgs'].append(cg_name)
            else:
                logger.info('Translation is valid for [{}]'.format(cg_name))

    # Create Theano variables
    floatX = theano.config.floatX
    src_sel = tensor.matrix('src_selector', dtype=floatX)
    trg_sel = tensor.matrix('trg_selector', dtype=floatX)
    x_sampling = tensor.matrix('source', dtype='int64')
    y_sampling = tensor.vector('target', dtype='int64')
    prev_state = tensor.matrix('prev_state', dtype=floatX)

    # for multi source - maximum is 10 for now
    xs_sampling = [tensor.matrix('source%d' % i, dtype='int64')
                   for i in range(10)]

    # Iterate over multiple models
    enc_dec_dict = OrderedDict()
    f_init_dict = OrderedDict()
    f_next_dict = OrderedDict()
    enc_ids_dict = OrderedDict()
    dec_ids_dict = OrderedDict()
    for mi, (model_id, model_path, config) in enumerate(
            zip(cgs_to_translate.keys(), models, configs)):

        # Helper variables
        cgs = config['cgs']
        trng = RandomStreams(config['seed'] if 'seed' in config else 1234)
        enc_ids, dec_ids = get_enc_dec_ids_mSrc(cgs)
        enc_ids_dict[model_id] = enc_ids
        dec_ids_dict[model_id] = dec_ids

        # Create encoder-decoder architecture
        logger.info('Creating encoder-decoder for model [{}]'.format(mi))
        enc_dec = EncoderDecoder(
            encoder=MultiEncoder(enc_ids=get_enc_dec_ids(cgs)[0], **config),
            decoder=MultiDecoder(**config))

        # Allocate parameters
        enc_dec.init_params()

        # Build sampling models
        logger.info('Building sampling models for model [{}]'.format(mi))
        f_inits, f_nexts, f_next_states = enc_dec.build_sampling_models(
            x_sampling, y_sampling, src_sel, trg_sel, prev_state,
            trng=trng, xs=xs_sampling)

        # Load parameters
        logger.info('Loading parameters for model [{}]'.format(mi))
        enc_dec.load_params(model_path)

        enc_dec_dict[model_id] = enc_dec
        f_init_dict[model_id] = f_inits
        f_next_dict[model_id] = f_nexts

    # Output translation file names to be returned
    translations = {}

    # Fetch necessary functions and variables from all models
    f_inits_list = []
    f_nexts_list = []
    src_vocabs_list = []
    src_vocabs_sizes_list = []

    source_files = OrderedDict()

    for midx, (model_id, cg_names) in enumerate(cgs_to_translate.items()):
        for cg_name in cg_names:

            config = configs[midx]
            enc_name = p_(cg_name)[0]
            dec_name = p_(cg_name)[1]
            enc_ids = enc_name.split('.')

            f_inits_list.append(f_init_dict[model_id][cg_name])
            f_nexts_list.append(f_next_dict[model_id][cg_name])

            if is_multiSource(cg_name):
                source_files.update(val_sets[cg_name])
            else:
                source_files[enc_name] = val_sets[cg_name]

            src_vocabs = OrderedDict()
            src_vocab_sizes = OrderedDict()

            # This ordering will be abided all the way
            for eid in p_(cg_name)[0].split('.'):
                src_vocabs[eid] = load_vocab(
                    config['src_vocabs'][eid], 0, config['src_eos_idxs'][eid],
                    config['unk_id'])
                src_vocab_sizes[eid] = config['src_vocab_sizes'][eid]
            src_vocabs_list.append(src_vocabs)
            src_vocabs_sizes_list.append(src_vocab_sizes)

    saveto = output_file

    # Skip if outputs exist
    if os.path.exists(saveto):
        logger.info('Outputs exist:')
        logger.info(' ... {}'.format(saveto))
        logger.info(' ... skipping')
        return

    logger.info('Output file: [{}]'.format(saveto))

    # Prepare target vocabs and files, make sure special tokens are there
    trg_vocab = load_vocab(
        configs[0]['trg_vocabs'][dec_name], 0,
        configs[0]['trg_eos_idxs'][dec_name], configs[0]['unk_id'])

    # Invert dictionary
    trg_ivocab = invert_vocab(trg_vocab)

    # Actual translation here
    for eid, fname in source_files.items():
        logger.info('Translating from [{}]-[{}]...'.format(eid, fname))
    logger.info('Using [{}] processes...'.format(n_process))
    val_start_time = time.time()

    # helper functions for multi-process
    def _send_jobs(source_fnames, source_files,
                   src_vocabs_list, src_vocabs_sizes_list):
        for idx, rows in enumerate(izip(*source_fnames)):
            lines = OrderedDict(zip(source_files.keys(), rows))
            seqs_list = [
                words2seqs_multi_irregular(
                    lines, src_vocabs_list[ii], src_vocabs_sizes_list[ii])
                for ii, _ in enumerate(src_vocabs_list)]
            queue.put((idx, seqs_list))
        return idx+1

    def _finish_processes():
        for midx in xrange(n_process):
            queue.put(None)

    def _retrieve_jobs(n_samples):
        trans = [None] * n_samples
        scores = [None] * n_samples
        for idx in xrange(n_samples):
            resp = rqueue.get()
            trans[resp[0]] = resp[1]
            scores[resp[0]] = resp[2]
            if numpy.mod(idx, 10) == 0:
                print 'Sample ', (idx+1), '/', n_samples, ' Done'
        return trans, scores

    # Open files with the correct ordering
    source_fnames = [open(source_files[eid], "r")
                     for eid in source_files.keys()]

    if n_process == 1:

        trans = []
        scores = []

        # Process each line for each source simultaneuosly
        for idx, rows in enumerate(izip(*source_fnames)):
            if idx % 100 == 0 and idx != 0:
                logger.info('...translated [{}] lines'.format(idx))
            lines = OrderedDict(zip(source_files.keys(), rows))
            seqs_list = [
                words2seqs_multi_irregular(
                    lines, src_vocabs_list[ii], src_vocabs_sizes_list[ii])
                for ii, _ in enumerate(src_vocabs_list)]
            _t, _s = _translate(
                seqs_list, f_inits_list, f_nexts_list, trg_vocab['</S>'],
                beam_size)
            trans.append(_t)
            scores.append(_s)

    else:

        # Create queues
        queue = Queue()
        rqueue = Queue()
        processes = [None] * n_process
        for midx in xrange(n_process):
            processes[midx] = Process(
                target=translate_model,
                args=(queue, rqueue, midx, f_inits_list, f_nexts_list,
                      trg_vocab['</S>'], beam_size),
                kwargs={'f_next_state': f_next_states})
            processes[midx].start()

        n_samples = _send_jobs(source_fnames, source_files,
                               src_vocabs_list, src_vocabs_sizes_list)
        trans, scores = _retrieve_jobs(n_samples)
        _finish_processes()

    logger.info("Validation Took: {} minutes".format(
        float(time.time() - val_start_time) / 60.))

    # Prepare translation outputs and calculate BLEU if necessary
    # Note that, translations are post processed for BPE here
    trans = seqs2words(trans, trg_vocab, trg_ivocab)
    trans = [tt.replace('@@ ', '') for tt in trans]
    if gold_file is not None and os.path.exists(gold_file):
        bleu_score = calculate_bleu(
            bleu_script=bleu_script, trans=trans,
            gold=gold_file)
        saveto += '{}'.format(bleu_score)

    # Write to file
    with open(saveto, 'w') as f:
        print >>f, '\n'.join(trans)

    translations[cg_name] = saveto
    return translations, saveto


def get_parser():

    def dict_type(ss):
        return dict([map(str.strip, s.split(':'))
                     for s in ss.split(',')])

    def odict2list_type(ss):
        ret = OrderedDict()
        for m2c in ss.split('@'):
            mid, cgs = m2c.split(':')
            ret[mid] = cgs.split(',')
        return ret

    def dict_type_mSrc(ss):
        aa = dict_type(ss)
        for k, v in aa.items():
            if is_multiSource(k):
                aa[k] = dict([el.split('=') for el in aa[k].split('@')])
        return aa

    parser = argparse.ArgumentParser()
    parser.add_argument('--num-process', '-p', type=int, default=5)
    parser.add_argument('--beam-size', type=int, default=7)
    parser.add_argument('--protos', type=str, nargs='+',
                        help='List of prototypes (for Aiur!)')
    parser.add_argument('--char-level', '-c', action="store_true",
                        default=False)
    parser.add_argument('--cgs-to-translate', type=odict2list_type,
                        help='Dictionary with list values, \
                        eg. --cgs-to-translate=model0:fr_en,es_en@model1:fr_en')
    parser.add_argument('--source-files', type=dict_type_mSrc,
                        help="Source files (optional), \
                        eg. --source-files=fi_en:file1,de_en:file2")
    parser.add_argument('--output-file', type=str,
                        help="Translation output file")
    parser.add_argument('--gold-file', type=str,
                        help="Translation reference file")
    parser.add_argument('--bleu-script', type=str, help="Path to bleu script")
    parser.add_argument("--changes", type=dict_type,
                        help="Changes/additions to config")
    parser.add_argument('-m', '--models', nargs='+', type=str)
    return parser


if __name__ == "__main__":

    args = get_parser().parse_args()

    # Set source files in config
    val_sets = OrderedDict()
    for cg_name, s_file in args.source_files.items():
        logger.info('val_sets[{}]:{}'.format(cg_name, s_file))
        val_sets[cg_name] = s_file
    logger.info('output_file:{}'.format(args.output_file))

    # Set translation output files in config
    configs = []
    for mi in range(len(args.models)):
        config = getattr(configuration, args.protos[mi])().copy()
        if args.changes is not None:
            config.update(args.changes)
        configs.append(config)

    logger.info("Model options:\n{}".format(pprint.pformat(config)))
    main(configs, args.models, val_sets, args.output_file,
         n_process=args.num_process, chr_level=args.char_level,
         cgs_to_translate=args.cgs_to_translate,
         gold_file=args.gold_file, bleu_script=args.bleu_script,
         beam_size=args.beam_size)
