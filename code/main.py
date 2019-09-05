#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import pickle
import sys, re, os
import multiprocessing
import numpy as np
from collections import OrderedDict
from gensim.models import KeyedVectors
from gensim.models import KeyedVectors
from collections import defaultdict
from random import shuffle

import data_loader
import score
import evaluation
import data_processing

def saveObjToFile(FileName,obj):
	fw = open(FileName,"wb")
	pickle.dump(obj,fw,protocol=2)
	fw.close()

#system settings
parser = argparse.ArgumentParser()

args = parser.parse_args()
parser.add_argument('--target', type=str, default="wt2g", help='what trec dataset to retrieval: wt2g,ap8889,robust04,wt10g,blog06')
parser.add_argument('--function', type=str, default="constant", help='what function to score: constant, linear function, gaussin function')
parser.add_argument('--window_width', type=int, default=30, help='text window width for documents')
parser.add_argument('--alpha', type=float, default=0.1, help='infulence factor to balance semantic interactions weighting')
parser.add_argument('--beta', type=float, default=0.05, help='infulence factor to balance BM25 method weighting')
parser.add_argument('--C', type=int, default=15, help='constant to balance parameter co')
parser.add_argument('--cons_a', type=int, default=9, help='constant for window width function')
parser.add_argument('--cons_b', type=int, default=1, help='constant for window width function')
args = parser.parse_args()

print '************** Loading Data**************'

data_directory = '../data'
w2v_file = data_directory+'/vector_bins/glove.6B.50d.g.wt2g.newdocs.txt'#glove.6B.50d.g.txt'
query_path = data_directory+'/queries/'+args.target
# doc_path = '/home/qiyy/data_4_ir/WT2G/new/docs/'
doc_path =data_directory+'/'+args.target+'/new/docs/'
qrels_path = data_directory+'/qrels/qrels.'+args.target
qrets_path = data_directory+'/qrets/'+args.target+'.res'
print w2v_file
w2v = data_loader.load_w2v(w2v_file)
print data_directory+'/results/'+args.target+'_win{0}_alpha{1}_bm25_beta{2}.eval'.format(args.window_width, args.alpha, args.beta)
print '*****************Scoring **************'

wind = args.window_width
prediction_data = data_loader.load_qrets(qrets_path)
prediction_score ={}
count=0
if args.function == 'constant':#alpha=0.1,beta=0.15,wind=30,c=15
	print args.function
	for qid,value in prediction_data.items():
		count += 1
		print 'processing qid %s: %d / %d'%(qid,count,len(prediction_data))
		y_pred =[]
		for docno in value['docno']:
			doc_scores = score.cw_c(w2v,query_path, qid, doc_path, docno, wind, args.alpha, args.C)
			y_pred.append(doc_scores)
		if qid not in prediction_score:
			prediction_score[qid] = {'docno':[],'y_pred':[]}
		prediction_score[qid]['docno']=value['docno']
		prediction_score[qid]['y_pred'].append(y_pred)
elif args.function == 'linear function':#alpha=0.15,beta=0.25,cons_a=10,cons_b=1,c=35/45
	for qid,value in prediction_data.items():
		count += 1
		print 'processing qid %s: %d / %d'%(qid,count,len(prediction_data))
		y_pred =[]
		for docno in value['docno']:
			doc_scores = score.cw_lf(w2v,query_path, qid, doc_path, docno, args.cons_a, args.cons_b, args.alpha, args.C)
			y_pred.append(doc_scores)
		if qid not in prediction_score:
			prediction_score[qid] = {'docno':[],'y_pred':[]}
		prediction_score[qid]['docno']=value['docno']
		prediction_score[qid]['y_pred'].append(y_pred)
elif args.function == 'gaussin function':#alpha=0.1,beta=0.15,cons_a=5,cons_b=0,c=15
	for qid,value in prediction_data.items():
		count += 1
		print 'processing qid %s: %d / %d'%(qid,count,len(prediction_data))
		y_pred =[]
		for docno in value['docno']:
			doc_scores = score.cw_gf(w2v,query_path, qid, doc_path, docno, args.cons_a, args.cons_b, args.alpha, args.C)
			y_pred.append(doc_scores)
		if qid not in prediction_score:
			prediction_score[qid] = {'docno':[],'y_pred':[]}
		prediction_score[qid]['docno']=value['docno']
		prediction_score[qid]['y_pred'].append(y_pred)

print '************** Combining with BM25 *************************'
score_bm25 = data_loader.load_qrets(qrets_path)
golden_data = data_loader.load_qrels(qrels_path)
res_dict = data_processing.lc_bm25(prediction_score, score_bm25, args.beta, topk=1000)


print '************** BM25 Evaluation  **************'
res_dict_bm25 = {'questions': []}
for qid ,value in score_bm25.items():
	docnos = value['docno']
	bm25_scores = score_bm25[qid]['bm25']
	retr_scores = list(zip(docnos, bm25_scores))
	shuffle(retr_scores) # Shuffle docs to make sure re-ranker works.
	sorted_retr_scores = sorted(retr_scores, key=lambda x: x[1], reverse=True)
	res_dict_bm25['questions'].append({'id': qid, 'documents': [d[0] for d in sorted_retr_scores], 'score':[d[1] for d in sorted_retr_scores]})
evaluation.trec(golden_data, res_dict_bm25, data_directory+'/results/'+args.target+'_bm25_0.35.eval')

print '************** Combining BM25 Evaluation **************'
evalfile_name = data_directory+'/results/'+args.target+'_{0}_win{1}_alpha{2}_consa{3}_consb{4}_bm25_beta{5}.eval'.format(args.function, wind, args.alpha, args.cons_a, args.cons_b, args.beta)#
evaluation.trec(golden_data, res_dict, evalfile_name)
