"""

Aligns three-way data with a reference file in the following format:

    <reference_sentence>|||<line_number_in_source1>|||<line_number_in_source2>

Well, the filename is a misnomer :)

"""
import argparse
import logging

from collections import OrderedDict


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('alignnway')


def get_argparser():
    def t_type(s):
        return tuple(s.split(':'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref', type=t_type, nargs='+')
    parser.add_argument('--inputs', type=t_type, nargs='+')
    parser.add_argument('--outputs', type=t_type, nargs='+')
    return parser


def main(ref, inputs, outputs):

    # Open files to read
    logger.info(" Opening files to read ...")
    inp_fds = {key: open(value, "r")
               for key, value in inputs.items()}

    # Load source files to memory !!!
    logger.info(" Loading source files to memory ...")
    lines = OrderedDict(
        [(lang, fd.readlines()) for lang, fd in inp_fds.items()])

    # Open output files
    logger.info(" Opening files to write ...")
    out_fnames = {key: open(value, "w") for key, value in outputs.items()}

    # Process each line for each source simultaneuosly
    logger.info(" Starting processing ...")
    refl, reff = ref.items()[0]
    src1, src2 = inputs.keys()
    with open(reff, 'r') as ins:
        for idx, row in enumerate(ins):
            ret = row.split('|||')
            ref_sent, idx1, idx2 = ret[0], int(ret[1]), int(ret[2])
            if idx % 10000 == 0:
                print ".",
            if idx % 100000 == 0:
                print idx,
            out_fnames[refl].write(ref_sent.strip() + '\n')
            out_fnames[src1].write(lines[src1][idx1 - 1].strip() + '\n')
            out_fnames[src2].write(lines[src2][idx2 - 1].strip() + '\n')

    print ""

    # Close input files
    logger.info(" Closing files ...")
    for k, f in inp_fds.items():
        f.close()
    for k, f in out_fnames.items():
        f.close()

    logger.info(" Done!")


if __name__ == "__main__":
    args = get_argparser().parse_args()
    main(OrderedDict(args.ref),
         OrderedDict(args.inputs),
         OrderedDict(args.outputs))
