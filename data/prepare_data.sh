#!/bin/bash

# Creates directory
create_dir () {
    if [ ! -d "${1}" ]; then
        mkdir -p ${1} 
    fi 
}

# ============================================================================
# Download europarl-v7 for es-en and fr-en
# Download newstest 

if [ ! -e "es-en.tgz" ]; then 
    echo "Downloading es-en.tgz"
    wget http://www.statmt.org/europarl/v7/es-en.tgz
else
    echo "[es-en.tgz] exists, skipping download" 
fi

if [ ! -e "fr-en.tgz" ]; then 
    echo "Downloading fr-en.tgz"
    wget http://www.statmt.org/europarl/v7/fr-en.tgz
else
    echo "[fr-en.tgz] exists, skipping download" 
fi


if [[ ! -e "dev.tgz" ]]; then 
    echo "Downloading dev.tgz"
    wget http://www.statmt.org/wmt13/dev.tgz
else
    echo "[dev.tgz] exists, skipping download" 
fi


# ============================================================================
# Extract 

if [ ! -e "europarl-v7.es-en.en" ]; then 
    echo "Extracting es-en.tgz"
    tar -xvzf es-en.tgz
else
    echo "[europarl-v7.es-en.en] exists, skipping extraction" 
fi

if [ ! -e "europarl-v7.fr-en.en" ]; then 
    echo "Extracting fr-en.tgz"
    tar -xvzf fr-en.tgz
else
    echo "[europarl-v7.fr-en.en] exists, skipping extraction" 
fi

if [ ! -e "dev/newstest2011.fr" ] || [ ! -e "dev/newstest2011.es" ] || \
   [ ! -e "dev/newstest2011.en" ] || [ ! -e "dev/newstest2012.fr" ] || \
   [ ! -e "dev/newstest2012.es" ] || [ ! -e "dev/newstest2012.en" ] ; 
then 
    echo "Extracting dev.tgz"
    tar -xvzf dev.tgz
fi


# ============================================================================
# Extract 3-way parallel dataset

if [ ! -e "3way.esfren" ]; then 
    echo "Extracting 3-way data"
    awk 'FNR==NR{l[$0]=NR; next}; $0 in l{printf "%s|||%s|||%s\n", $0, l[$0], FNR}' \
        europarl-v7.es-en.en europarl-v7.fr-en.en > 3way.esfren
else
    echo "[3way.esfren] exists, skipping extraction" 
fi

if [ ! -e "europarl-v7.esfr-en.fr" ] || \
   [ ! -e "europarl-v7.esfr-en.en" ] || \
   [ ! -e "europarl-v7.esfr-en.es" ] ; 
then 
    echo "Aligning Es-Fr-En"
    python align_3way.py \
        --ref en:3way.esfren \
        --inputs es:europarl-v7.es-en.es \
                 fr:europarl-v7.fr-en.fr \
        --outputs es:europarl-v7.esfr-en.es \
                  fr:europarl-v7.esfr-en.fr \
                  en:europarl-v7.esfr-en.en
fi


# ============================================================================
# Tokenize using moses tokenizer

# utility function 
moses_tokenizer=tokenizer.perl
tokenize () {
    inp=$1
    out=$2
    lang=$3
    if [ -e "${out}" ]; then
        echo "[${out}] exists"
    else 
        echo "...tokenizing $inp with $lang tokenizer"
        perl ${moses_tokenizer} -l ${lang} -threads 8 < ${inp} > ${out} 
    fi 
}


# tokenize english
tokenize europarl-v7.esfr-en.en europarl-v7.esfr-en.en.tok en 
tokenize europarl-v7.es-en.en europarl-v7.es-en.en.tok en 
tokenize europarl-v7.fr-en.en europarl-v7.fr-en.en.tok en 
tokenize dev/newstest2011.en dev/newstest2011.en.tok en 
tokenize dev/newstest2012.en dev/newstest2012.en.tok en 

# tokenize french
tokenize europarl-v7.esfr-en.fr europarl-v7.esfr-en.fr.tok fr 
tokenize europarl-v7.fr-en.fr europarl-v7.fr-en.fr.tok fr 
tokenize dev/newstest2011.fr dev/newstest2011.fr.tok fr  
tokenize dev/newstest2012.fr dev/newstest2012.fr.tok fr  

# tokenize spanish
tokenize europarl-v7.esfr-en.es europarl-v7.esfr-en.es.tok es
tokenize europarl-v7.es-en.es europarl-v7.es-en.es.tok es 
tokenize dev/newstest2011.es dev/newstest2011.es.tok es 
tokenize dev/newstest2012.es dev/newstest2012.es.tok es 

##############################################################################
# Learn byte pair encodings with 20k codes

# Wrapper to call learn bpe
learn_bpe=${HOME}/git/subword-nmt/learn_bpe.py
learn_bpe_call () {
    inp=$1
    out=$2
    n_sym=$3
    if [ -e "${out}" ]; then
        echo "[${out}] exists"
    else 
        echo "...learning bpe with ${n_sym} symbols using ${inp}"
        python ${learn_bpe} -s ${n_sym} < ${inp} > ${out}
    fi 
}

# first clone bpe repo if does not exist
if [ ! -e "${learn_bpe}" ]; then
    create_dir "${HOME}/git"
    git clone https://github.com/rsennrich/subword-nmt "${HOME}/git/subword-nmt"
fi

# code files
en_codes=europarl-v7.fr-en.en.tok.code20k 
es_codes=europarl-v7.es-en.es.tok.code20k 
fr_codes=europarl-v7.fr-en.fr.tok.code20k 

# learn codes 
learn_bpe_call europarl-v7.fr-en.en.tok ${en_codes} 20000    
learn_bpe_call europarl-v7.fr-en.fr.tok ${fr_codes} 20000    
learn_bpe_call europarl-v7.es-en.es.tok ${es_codes} 20000    


##############################################################################
# Encode all the training, dev and test files using the learned codes

# Encodes a given text file using code
apply_bpe=${HOME}/git/subword-nmt/apply_bpe.py
encode () {
    code=$1
    inp=$2
    out=$3
    if [ -e "${out}" ]; then
        echo "[${out}] exists"
    else 
        echo "...encoding ${inp}"
        python ${apply_bpe} -c ${code} < ${inp} > ${out}
    fi 
}

# encode english
encode ${en_codes} europarl-v7.esfr-en.en.tok europarl-v7.esfr-en.en.tok.bpe20k
encode ${en_codes} europarl-v7.fr-en.en.tok europarl-v7.fr-en.en.tok.bpe20k
encode ${en_codes} europarl-v7.es-en.en.tok europarl-v7.es-en.en.tok.bpe20k
encode ${en_codes} dev/newstest2011.en.tok dev/newstest2011.en.tok.bpe20k
encode ${en_codes} dev/newstest2012.en.tok dev/newstest2012.en.tok.bpe20k

# encode french 
encode ${fr_codes} europarl-v7.esfr-en.fr.tok europarl-v7.esfr-en.fr.tok.bpe20k
encode ${fr_codes} europarl-v7.fr-en.fr.tok europarl-v7.fr-en.fr.tok.bpe20k
encode ${fr_codes} dev/newstest2011.fr.tok dev/newstest2011.fr.tok.bpe20k
encode ${fr_codes} dev/newstest2012.fr.tok dev/newstest2012.fr.tok.bpe20k

# encode spanish 
encode ${es_codes} europarl-v7.esfr-en.es.tok europarl-v7.esfr-en.es.tok.bpe20k
encode ${es_codes} europarl-v7.es-en.es.tok europarl-v7.es-en.es.tok.bpe20k
encode ${es_codes} dev/newstest2011.es.tok dev/newstest2011.es.tok.bpe20k
encode ${es_codes} dev/newstest2012.es.tok dev/newstest2012.es.tok.bpe20k


##############################################################################
# Extract dictionaries

dictionary=preprocess.py
extract_dict () {
    inp=$1
    out=$1.vocab.pkl
    if [ -e "${out}" ]; then
        echo "[${out}] exists"
    else 
        echo "...extracting dictionary ${inp}"
        python ${dictionary} ${inp} -d ${out}
    fi 
}

extract_dict europarl-v7.fr-en.en.tok.bpe20k
extract_dict europarl-v7.fr-en.fr.tok.bpe20k
extract_dict europarl-v7.es-en.es.tok.bpe20k

echo "PRE-PROCESSING DONE"
