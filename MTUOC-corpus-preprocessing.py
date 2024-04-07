#    MTUOC-corpus-preprocessing
#    Copyright (C) 2024  Antoni Oliver
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.


import sys
from datetime import datetime
import os
import codecs
import importlib
import re

import pickle

from shutil import copyfile

import yaml
from yaml import load, dump

from itertools import (takewhile,repeat)


try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
    
def file_len(fname):
    num_lines = sum(1 for line in open(fname))
    return(num_lines)
    
def findEMAILs(string): 
    email=re.findall('\S+@\S+', string)   
    return email

'''
def findURLs(string): 
    regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
    url = re.findall(regex,string)       
    return [x[0] for x in url] 
'''
def findURLs(text):
    # Regular expression for identifying URLs
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    # Find all matches using the regular expression
    matches = re.findall(url_pattern, text)
    return(matches)
    
def replace_EMAILs(string,code="@EMAIL@"):
    EMAILs=findEMAILs(string)
    cont=0
    for EMAIL in EMAILs:
        string=string.replace(EMAIL,code)
    return(string)

def replace_URLs(string,code="@URL@"):
    URLs=findURLs(string)
    cont=0
    for URL in URLs:
        string=string.replace(URL,code)
    return(string)

def rawincount(filename):
    f = open(filename, 'rb')
    bufgen = takewhile(lambda x: x, (f.raw.read(1024*1024) for _ in repeat(None)))
    return sum( buf.count(b'\n') for buf in bufgen )

stream = open('config-corpus-preprocessing.yaml', 'r',encoding="utf-8")
config=yaml.load(stream, Loader=yaml.FullLoader)

MTUOC=config["MTUOC"]
sys.path.append(MTUOC)

from MTUOC_train_truecaser import TC_Trainer
from MTUOC_truecaser import Truecaser
from MTUOC_splitnumbers import splitnumbers

import sentencepiece as spm

from MTUOC_sentencepiece import sentencepiece_train
from MTUOC_sentencepiece import sentencepiece_encode
from MTUOC_subwordnmt import subwordnmt_train
from MTUOC_subwordnmt import subwordnmt_encode

preprocess_type=config["preprocess_type"]

DELETE_TEMP=config["DELETE_TEMP"]

corpus=config["corpus"]
valsize=int(config["valsize"])
evalsize=int(config["evalsize"])
SLcode3=config["SLcode3"]
SLcode2=config["SLcode2"]
TLcode3=config["TLcode3"]
TLcode2=config["TLcode2"]

from_train_val=config["from_train_val"]
train_corpus=config["train_corpus"]
val_corpus=config["val_corpus"]

#VERBOSE
VERBOSE=config["VERBOSE"]
LOGFILE=config["LOG_FILE"]

REPLACE_EMAILS=config["REPLACE_EMAILS"]
EMAIL_CODE=config["EMAIL_CODE"]
REPLACE_URLS=config["REPLACE_URLS"]
URL_CODE=config["URL_CODE"]


TRAIN_SL_TRUECASER=config["TRAIN_SL_TRUECASER"]
TRAIN_SL_TRUECASER_MAXLINES=int(config["TRAIN_SL_TRUECASER_MAXLINES"])
SL_DICT=config["SL_DICT"]
TRUECASE_SL=config["TRUECASE_SL"]
SL_TC_MODEL=config["SL_TC_MODEL"]
if SL_TC_MODEL=="auto":
    SL_TC_MODEL="tc."+SLcode2

TRAIN_TL_TRUECASER=config["TRAIN_TL_TRUECASER"]
TRAIN_TL_TRUECASER_MAXLINES=int(config["TRAIN_TL_TRUECASER_MAXLINES"])
TL_DICT=config["TL_DICT"]
TRUECASE_TL=config["TRUECASE_TL"]
TL_TC_MODEL=config["TL_TC_MODEL"]
if TL_TC_MODEL=="auto":
    TL_TC_MODEL="tc."+TLcode2
    
SL_TOKENIZER=config["SL_TOKENIZER"]
if SL_TOKENIZER=="None":
    SL_TOKENIZER=None
TL_TOKENIZER=config["TL_TOKENIZER"]
if TL_TOKENIZER=="None":
    TL_TOKENIZER=None
TOKENIZE_SL=config["TOKENIZE_SL"]
TOKENIZE_TL=config["TOKENIZE_TL"]

if SL_TOKENIZER==None: TOKENIZE_SL=False
if TL_TOKENIZER==None: TOKENIZE_TL=False


CLEAN=config["CLEAN"]
MIN_TOK=config["MIN_TOK"]
MAX_TOK=config["MAX_TOK"]

MIN_CHAR=config["MIN_CHAR"]
MAX_CHAR=config["MAX_CHAR"]

#SENTENCE PIECE
SP_MODEL_PREFIX=config["SP_MODEL_PREFIX"]
MODEL_TYPE=config["MODEL_TYPE"]
#one of unigram, bpe, char, word
JOIN_LANGUAGES=config["JOIN_LANGUAGES"]
VOCAB_SIZE=config["VOCAB_SIZE"]
CHARACTER_COVERAGE=config["CHARACTER_COVERAGE"]
CHARACTER_COVERAGE_SL=config["CHARACTER_COVERAGE_SL"]
CHARACTER_COVERAGE_TL=config["CHARACTER_COVERAGE_TL"]
VOCABULARY_THRESHOLD=config["VOCABULARY_THRESHOLD"]
INPUT_SENTENCE_SIZE=config["INPUT_SENTENCE_SIZE"]
CONTROL_SYMBOLS=config["CONTROL_SYMBOLS"]
USER_DEFINED_SYMBOLS=config["USER_DEFINED_SYMBOLS"]

BOS=config["bos"]
EOS=config["eos"]

#SUBWORD NMT
LEARN_BPE=config["LEARN_BPE"]
JOINER=config["JOINER"]
SPLIT_DIGITS=config["SPLIT_DIGITS"]
NUM_OPERATIONS=config["NUM_OPERATIONS"]
APPLY_BPE=config["APPLY_BPE"]
BPE_DROPOUT=config["BPE_DROPOUT"]
BPE_DROPOUT_P=config["BPE_DROPOUT_P"]

if VERBOSE:
    logfile=codecs.open(LOGFILE,"w",encoding="utf-8")

if not from_train_val:
    #SPLITTING CORPUS
    corpussize=rawincount(corpus)
    trainsize=corpussize-valsize-evalsize
    print(corpus,corpussize,"VAL:",valsize,"EVAL:",evalsize,"TRAIN:",trainsize)
    trainCorpus="train-"+SLcode3+"-"+TLcode3+".txt"
    valCorpus="val-"+SLcode3+"-"+TLcode3+".txt"
    
    trainPreCorpus="train-pre-"+SLcode3+"-"+TLcode3+".txt"
    valPreCorpus="val-pre-"+SLcode3+"-"+TLcode3+".txt"
    
    evalCorpus="eval-"+SLcode3+"-"+TLcode3+".txt"
    entrada=codecs.open(corpus,"r",encoding="utf-8")
    sortidaTrain=codecs.open(trainCorpus,"w",encoding="utf-8")
    sortidaVal=codecs.open(valCorpus,"w",encoding="utf-8")
    sortidaEval=codecs.open(evalCorpus,"w",encoding="utf-8")
    cont=0
    for linia in entrada:
        if cont < valsize:
            sortidaVal.write(linia)
        elif cont < valsize+evalsize:
            sortidaEval.write(linia)
        else:
            sortidaTrain.write(linia)
        cont+=1
    sortidaTrain.close()
    sortidaVal.close()
    sortidaEval.close()
    entrada=codecs.open(evalCorpus,"r",encoding="utf-8")
    evalSL="eval."+SLcode2
    evalTL="eval."+TLcode2
    sortidaSL=codecs.open(evalSL,"w",encoding="utf-8")
    sortidaTL=codecs.open(evalTL,"w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
else:
    trainCorpus=train_corpus
    valCorpus=val_corpus
    trainPreCorpus="train-pre-"+SLcode3+"-"+TLcode3+".txt"
    valPreCorpus="val-pre-"+SLcode3+"-"+TLcode3+".txt"


if VERBOSE:
    cadena="Start of process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")

#TRAIN

entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortidaSL=codecs.open("trainSL.temp","w",encoding="utf-8")
sortidaTL=codecs.open("trainTL.temp","w",encoding="utf-8")
sortidaW=codecs.open("trainW.temp","w",encoding="utf-8")
for linia in entrada:
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        sortidaSL.write(camps[0]+"\n")
        sortidaTL.write(camps[1]+"\n")
        if len(camps)>=3:
            sortidaW.write(camps[2]+"\n")
        else:
            sortidaW.write("\n")
entrada.close()
sortidaSL.close()
sortidaTL.close()
sortidaW.close()

if TRAIN_SL_TRUECASER:
    if VERBOSE:
        cadena="Training SL Truecaser: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
        logfile.flush()
    SLTrainer=TC_Trainer(MTUOC, SL_TC_MODEL, "trainSL.temp", SL_DICT, SL_TOKENIZER,maxlines=TRAIN_SL_TRUECASER_MAXLINES)
    SLTrainer.train_truecaser()

if TRAIN_TL_TRUECASER:
    if VERBOSE:
        cadena="Training TL Truecaser: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
        logfile.flush()
    TLTrainer=TC_Trainer(MTUOC, TL_TC_MODEL, "trainTL.temp", TL_DICT, TL_TOKENIZER,maxlines=TRAIN_TL_TRUECASER_MAXLINES)
    TLTrainer.train_truecaser()    

if TRUECASE_SL:
    truecaserSL=Truecaser()
    truecaserSL.set_MTUOCPath(MTUOC)
    truecaserSL.set_tokenizer(SL_TOKENIZER)
    truecaserSL.set_tc_model(SL_TC_MODEL)

if TRUECASE_TL:
    truecaserTL=Truecaser()
    truecaserTL.set_MTUOCPath(MTUOC)
    truecaserTL.set_tokenizer(TL_TOKENIZER)
    truecaserTL.set_tc_model(TL_TC_MODEL)


if not SL_TOKENIZER==None:
    SL_TOKENIZER=MTUOC+"/"+SL_TOKENIZER
    if not SL_TOKENIZER.endswith(".py"): SL_TOKENIZER=SL_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', SL_TOKENIZER)
    tokenizerSLmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerSLmod)
    tokenizerSL=tokenizerSLmod.Tokenizer()

if not TL_TOKENIZER==None: 
    TL_TOKENIZER=MTUOC+"/"+TL_TOKENIZER   
    if not TL_TOKENIZER.endswith(".py"): TL_TOKENIZER=TL_TOKENIZER+".py"
    spec = importlib.util.spec_from_file_location('', TL_TOKENIZER)
    tokenizerTLmod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tokenizerTLmod)
    tokenizerTL=tokenizerTLmod.Tokenizer()



if VERBOSE:
    cadena="Preprocessing train corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")
    logfile.flush()


entrada=codecs.open(trainCorpus,"r",encoding="utf-8")
sortida=codecs.open(trainPreCorpus,"w",encoding="utf-8")
sortidaW=codecs.open("train.weights","w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        if len(camps)>=3:
            weight=camps[2]
        else:
            weight=1
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_SL:
            toksl=tokenizerSL.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_TL:
            toktl=tokenizerTL.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_SL:
                toksl=truecaserSL.truecase(toksl)
            if TRUECASE_TL:
                toktl=truecaserTL.truecase(toktl)
            
            cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            
            sortida.write(cadena+"\n")
            sortidaW.write(str(weight)+"\n")
    
entrada.close()
sortida.close()
sortidaW.close()
#Val CORPUS
if VERBOSE:
    cadena="Preprocessing val corpus: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")
    logfile.flush()

entrada=codecs.open(valCorpus,"r",encoding="utf-8")
sortida=codecs.open(valPreCorpus,"w",encoding="utf-8")
sortidaW=codecs.open("val.weights","w",encoding="utf-8")

for linia in entrada:
    toWrite=True
    linia=linia.rstrip()
    camps=linia.split("\t")
    if len(camps)>=2:
        l1=camps[0]
        l2=camps[1]
        if len(camps)>=3:
            weight=camps[2]
        else:
            weight=1
        lensl=len(l1)
        lentl=len(l2)
        if TOKENIZE_SL:
            toksl=tokenizerSL.tokenize(l1)
        else:
            toksl=l1
        if TOKENIZE_TL:
            toktl=tokenizerTL.tokenize(l2)
        else:
            toktl=l2
        lentoksl=len(toksl.split(" "))
        lentoktl=len(toktl.split(" "))
        if CLEAN and lensl<MIN_CHAR: toWrite=False
        if CLEAN and lentl<MIN_CHAR: toWrite=False
        if CLEAN and lensl>MAX_CHAR: toWrite=False
        if CLEAN and lentl>MAX_CHAR: toWrite=False
        
        if CLEAN and lentoksl<MIN_TOK: toWrite=False
        if CLEAN and lentoktl<MIN_TOK: toWrite=False
        if CLEAN and lentoksl>MAX_TOK: toWrite=False
        if CLEAN and lentoktl>MAX_TOK: toWrite=False
        
        if toWrite:
            if REPLACE_EMAILS:
                toksl=replace_EMAILs(toksl,EMAIL_CODE)
                toktl=replace_EMAILs(toktl,EMAIL_CODE)
            if REPLACE_URLS:
                toksl=replace_URLs(toksl)
                toktl=replace_URLs(toktl)
            if TRUECASE_SL:
                toksl=truecaserSL.truecase(toksl)
            if TRUECASE_TL:
                toktl=truecaserTL.truecase(toktl)
            
            cadena=" ".join(toksl.split())+"\t"+" ".join(toktl.split())+"\t"+str(weight)
            
            sortida.write(cadena+"\n")
            sortidaW.write(str(weight)+"\n")
    
entrada.close()
sortida.close()
sortidaW.close()

if preprocess_type=="sentencepiece":
    ###sentencepiece is default if no smt or subword-nmt is selected
    if VERBOSE:
        cadena="Start of sentencepiece process: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
        logfile.flush()

    if VERBOSE:
        cadena="Start of sentencepiece training: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
        logfile.flush()

    entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("trainPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("trainPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("trainPreW.temp","w",encoding="utf-8")

    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")

        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
    sortidaW.close()
            
    entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("valPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("valPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("valPreW.temp","w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
        
    if VERBOSE:
        cadena="Training sentencepiece: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
        logfile.flush()
    bosSP=True
    eosSP=True
    if BOS=="None": bosSP=False
    if EOS=="None": eosSP=False
    sentencepiece_train("trainPreSL.temp","trainPreTL.temp",SLcode2=SLcode2,TLcode2=TLcode2,JOIN_LANGUAGES=JOIN_LANGUAGES,SP_MODEL_PREFIX=SP_MODEL_PREFIX,MODEL_TYPE=MODEL_TYPE,VOCAB_SIZE=VOCAB_SIZE,CHARACTER_COVERAGE=CHARACTER_COVERAGE,INPUT_SENTENCE_SIZE=INPUT_SENTENCE_SIZE,SPLIT_DIGITS=SPLIT_DIGITS,CONTROL_SYMBOLS=CONTROL_SYMBOLS,USER_DEFINED_SYMBOLS=USER_DEFINED_SYMBOLS)
    
    if VERBOSE:
        cadena="Encoding corpora with sentencepiece: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
        logfile.flush()
    
    if JOIN_LANGUAGES:
        SP_MODEL=SP_MODEL_PREFIX+".model"
    else:
        SP_MODEL=SP_MODEL_PREFIX+"-"+SLcode2+".model"
    
    outfile="train.sp."+SLcode2
    vocabulary_file="vocab_file."+SLcode2
    sentencepiece_encode("trainPreSL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)
    outfile="val.sp."+SLcode2
    vocabulary_file="vocab_file."+SLcode2
    sentencepiece_encode("valPreSL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)
    
    if JOIN_LANGUAGES:
        SP_MODEL=SP_MODEL_PREFIX+".model"
    else:
        SP_MODEL=SP_MODEL_PREFIX+"-"+TLcode2+".model"
    
    outfile="train.sp."+TLcode2
    vocabulary_file="vocab_file."+TLcode2
    sentencepiece_encode("trainPreTL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)
    outfile="val.sp."+TLcode2
    sentencepiece_encode("valPreTL.temp",OUTFILE=outfile, SP_MODEL=SP_MODEL,VOCABULARY=vocabulary_file,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,BOS=bosSP,EOS=eosSP)
         
elif preprocess_type=="subwordnmt":
    print("SUBWORD NMT BPE")
    #####################
    print("Starting BPE training",datetime.now())

    entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("trainPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("trainPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("trainPreW.temp","w",encoding="utf-8")

    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()
            
    entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
    sortidaSL=codecs.open("valPreSL.temp","w",encoding="utf-8")
    sortidaTL=codecs.open("valPreTL.temp","w",encoding="utf-8")
    sortidaW=codecs.open("valPreW.temp","w",encoding="utf-8")

    for linia in entrada:
        linia=linia.rstrip()
        camps=linia.split("\t")
        if len(camps)>=2:
            sortidaSL.write(camps[0]+"\n")
            sortidaTL.write(camps[1]+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        else:
            print("ERROR",camps)
    entrada.close()
    sortidaSL.close()
    sortidaTL.close()

    if LEARN_BPE: 
        if VERBOSE:
            print("Learning BPE",datetime.now())
        if JOIN_LANGUAGES: 
            if VERBOSE: print("JOINING LANGUAGES",datetime.now())
            subwordnmt_train("trainPreSL.temp trainPreTL.temp",SLcode2=SLcode2,TLcode2=TLcode2,NUM_OPERATIONS=NUM_OPERATIONS,CODES_file="codes_file")

        else:
            print("**************NOT JOINING LANGUAGES")
            if VERBOSE: print("SL",datetime.now())
            subwordnmt_train("trainPreSL.temp",SLcode2=SLcode2,TLcode2="",NUM_OPERATIONS=NUM_OPERATIONS,CODES_file="codes_file."+SLcode2)
           
            if VERBOSE: print("TL",datetime.now())
            subwordnmt_train("trainPreTL.temp",SLcode2=TLcode2,TLcode2="",NUM_OPERATIONS=NUM_OPERATIONS,CODES_file="codes_file."+TLcode2)
           


    if APPLY_BPE: 
        if VERBOSE:
            print("Applying BPE",datetime.now())
        if JOIN_LANGUAGES:
            BPESL="codes_file"
            BPETL="codes_file"
        if not JOIN_LANGUAGES:
            BPESL="codes_file."+SLcode2
            BPETL="codes_file."+TLcode2
        
        subwordnmt_encode("trainPreSL.temp","train.bpe."+SLcode2,CODES_FILE=BPESL,VOCAB_FILE="vocab_BPE."+SLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
        subwordnmt_encode("trainPreTL.temp","train.bpe."+TLcode2,CODES_FILE=BPETL,VOCAB_FILE="vocab_BPE."+TLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
        
        subwordnmt_encode("valPreSL.temp","val.bpe."+SLcode2,CODES_FILE=BPESL,VOCAB_FILE="vocab_BPE."+SLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
        subwordnmt_encode("valPreTL.temp","val.bpe."+TLcode2,CODES_FILE=BPETL,VOCAB_FILE="vocab_BPE."+TLcode2,VOCABULARY_THRESHOLD=VOCABULARY_THRESHOLD,JOINER=JOINER,BPE_DROPOUT=BPE_DROPOUT,BPE_DROPOUT_P=BPE_DROPOUT_P,SPLIT_DIGITS=SPLIT_DIGITS,BOS=BOS,EOS=EOS)
       
    
    
    #####################


elif preprocess_type=="smt":
    #train
    entrada=codecs.open(trainPreCorpus,"r",encoding="utf-8")
    nomsl="train.smt."+SLcode2
    nomtl="train.smt."+TLcode2
    sortidaSL=codecs.open(nomsl,"w",encoding="utf-8")
    sortidaTL=codecs.open(nomtl,"w",encoding="utf-8")
    sortidaW=codecs.open("trainPreW.temp","w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        try:
            camps=linia.split("\t")
            SLsegment=camps[0]
            TLsegment=camps[1]
            sortidaSL.write(SLsegment+"\n")
            sortidaTL.write(TLsegment+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        except:
            pass
            
    #val
    entrada=codecs.open(valPreCorpus,"r",encoding="utf-8")
    nomsl="val.smt."+SLcode2
    nomtl="val.smt."+TLcode2
    sortidaSL=codecs.open(nomsl,"w",encoding="utf-8")
    sortidaTL=codecs.open(nomtl,"w",encoding="utf-8")
    sortidaW=codecs.open("valPreW.temp","w",encoding="utf-8")
    for linia in entrada:
        linia=linia.rstrip()
        try:
            camps=linia.split("\t")
            SLsegment=camps[0]
            TLsegment=camps[1]
            sortidaSL.write(SLsegment+"\n")
            sortidaTL.write(TLsegment+"\n")
            if len(camps)>=3:
                sortidaW.write(camps[2]+"\n")
            else:
                sortidaW.write("\n")
        except:
            pass

if VERBOSE:
    cadena="End of process: "+str(datetime.now())
    print(cadena)
    logfile.write(cadena+"\n")
    logfile.flush()

#DELETE TEMPORAL FILES


if DELETE_TEMP:
    if VERBOSE:
        cadena="Deleting temporal files: "+str(datetime.now())
        print(cadena)
        logfile.write(cadena+"\n")
            
        valPreTL.temp
    todeletetemp=["trainPreSL.temp","trainPreW.temp","trainTL.temp","valPreSL.temp","valPreW.temp","trainPreTL.temp","trainSL.temp","trainW.temp","valPreTL.temp"]
    for td in todeletetemp:
        try:
            os.remove(td)
        except:
            pass

