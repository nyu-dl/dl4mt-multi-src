"""
TODO: write me
"""

# multi-source data paths
src_vocabs = {

    # spanish
    'es': 'europarl-v7.es-en.es.tok.bpe20k.vocab.pkl',

    # french
    'fr': 'europarl-v7.fr-en.fr.tok.bpe20k.vocab.pkl',

}
trg_vocabs = {

    # english
    'en': 'europarl-v7.fr-en.en.tok.bpe20k.vocab.pkl',

}
src_datas = {

    # spanish-english single pair data
    'es_en': 'europarl-v7.es-en.es.tok.bpe20k',

    # french-english single pair data
    'fr_en': 'europarl-v7.fr-en.fr.tok.bpe20k',

    # spanish + french - english multi-source data
    'es.fr_en': {
        'es': 'europarl-v7.esfr-en.es.tok.bpe20k',
        'fr': 'europarl-v7.esfr-en.fr.tok.bpe20k'},

}
trg_datas = {

    # spanish-english single pair data
    'es_en': 'europarl-v7.es-en.en.tok.bpe20k',

    # french-english single pair data
    'fr_en': 'europarl-v7.fr-en.en.tok.bpe20k',

    # spanish + french - english multi-source data
    'es.fr_en': {
        'en': 'europarl-v7.esfr-en.en.tok.bpe20k'},

}
log_prob_sets = {

    # spanish-english
    'es_en': ['dev/newstest2011.es.tok.bpe20k',
              'dev/newstest2011.en.tok.bpe20k'],

    # french-english
    'fr_en': ['dev/newstest2011.fr.tok.bpe20k',
              'dev/newstest2011.en.tok.bpe20k'],

    # spanish + french -english multi-source
    'es.fr_en': {
        'es': 'dev/newstest2011.es.tok.bpe20k',
        'fr': 'dev/newstest2011.fr.tok.bpe20k',
        'en': 'dev/newstest2011.en.tok.bpe20k'},
}
