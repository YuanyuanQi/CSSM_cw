#!/usr/bin/python
# -*- coding: UTF-8 -*-
import numpy as np
import pickle
import data_processing
import os
import data_loader
import math

def rolling_max(A, window, num_max):
	'''Computes roling maximum of 2D array.
	A is the array, window is the length of the rolling window and num_max is the number of maximums to return for each window.
	The output is an array of size (D,N,num_max) where D is the number of 
	columns in A and N is the number of rows.
	'''
	shape = (A.shape[1], np.max([A.shape[0]-window+1, 1]), np.min([window, A.shape[0]]))
	strides = (A.strides[-1],) + (A.strides[-1]*A.shape[1],) + (A.strides[-1]*A.shape[1],)
	b = np.lib.stride_tricks.as_strided(A, shape=shape, strides=strides)
	return np.sort(b, axis=2)[:,:,::-1][:,:,:num_max]

def cw_c(w2v,query_path, qid, doc_path, docno, wind, alpha, C):
	
	Stopwords = data_loader.load_sw()
	qry_list = pickle.load(open(os.path.join(query_path, qid)))
	qry_list = list(set(qry_list)-set(Stopwords))
	query_np = data_processing.term2vector(w2v,qry_list)

	sents = pickle.load(open(os.path.join(doc_path, docno)))
	words = sum(sents, [])
	doc_np = data_processing.term2vector(w2v,words)
	co = len(set(qry_list)&set(words))
	Doc_Score =[]
	
	try:
		query_np = data_processing.query_tv(query_np)#term vectors weighting
		qd_cos = doc_np.dot(query_np.T)/np.outer(np.linalg.norm(doc_np, axis=1),np.linalg.norm(query_np, axis=1))#doc_len*query_len

		if wind > doc_np.shape[0]:
			length = doc_np.shape[0]
		else:
			length = wind

		Top_K = int(math.log(length))+1
		Con_Maxs = rolling_max(qd_cos, length, Top_K)#[query_len*(doc_len-45+1)*2] 
		Con_Score = np.sum(Con_Maxs, axis=0)#[(doc_len-45+1)*2]
		score_al = np.max(Con_Score[:,0] + alpha*np.mean(Con_Score,axis=1))*math.log(co+C)
		Doc_Score.append(score_al)
	except:
		Doc_Score.append(0.0)

	return Doc_Score


def cw_lf(w2v,query_path, qid, doc_path, docno, a, b, alpha, C):
	
	Stopwords = data_loader.load_sw()
	qry_list = pickle.load(open(os.path.join(query_path, qid)))
	qry_list = list(set(qry_list)-set(Stopwords))
	query_np = data_processing.term2vector(w2v,qry_list)

	sents = pickle.load(open(os.path.join(doc_path, docno)))
	words = sum(sents, [])
	doc_np = data_processing.term2vector(w2v,words)
	co = len(set(qry_list)&set(words))
	Doc_Score =[]
	wind = (a+1)*int(len(qry_list))+b

	try:
		query_np = data_processing.query_tv(query_np)#term vectors weighting
		qd_cos = doc_np.dot(query_np.T)/np.outer(np.linalg.norm(doc_np, axis=1),np.linalg.norm(query_np, axis=1))#doc_len*query_len

		if wind > doc_np.shape[0]:
			length = doc_np.shape[0]
		else:
			length = wind

		Top_K = int(math.log(length))+1
		Con_Maxs = rolling_max(qd_cos, length, Top_K)#[query_len*(doc_len-45+1)*2] 
		Con_Score = np.sum(Con_Maxs, axis=0)#[(doc_len-45+1)*2]
		score_al = np.max(Con_Score[:,0] + alpha*np.mean(Con_Score,axis=1))*math.log(co+C)
		# score_al = np.max(np.array(Con_Score)[:,0] + alpha*(np.array(np.mean(np.array(Con_Score),axis=1))))*math.log(co+15)
		Doc_Score.append(score_al)
	except:
		Doc_Score.append(0.0)

	return Doc_Score



def cw_gf(w2v,query_path, qid, doc_path, docno, a, b, alpha, C):
	
	
	
	Stopwords = data_loader.load_sw()
	qry_list = pickle.load(open(os.path.join(query_path, qid)))
	qry_list = list(set(qry_list)-set(Stopwords))
	query_np = data_processing.term2vector(w2v,qry_list)

	sents = pickle.load(open(os.path.join(doc_path, docno)))
	words = sum(sents, [])
	doc_np = data_processing.term2vector(w2v,words)
	co = len(set(qry_list)&set(words))
	Doc_Score =[]
	
	temp = query_np.dot(query_np.T)/np.outer(np.linalg.norm(query_np, axis=1),np.linalg.norm(query_np, axis=1))#query_np*query_len
	if len(qry_list) > 2:
		wind = (a+1)*int(round(np.mean(temp)/(np.var(temp)*math.sqrt(2))))+b
		# wind = len(qry_list)*(int(round(np.mean(temp)/(np.std(temp)*math.sqrt(2))))+Cons_2)#
	else:
		wind = len(qry_list)*15
	del temp
	try:
		query_np = data_processing.query_tv(query_np)#term vectors weighting
		qd_cos = doc_np.dot(query_np.T)/np.outer(np.linalg.norm(doc_np, axis=1),np.linalg.norm(query_np, axis=1))#doc_len*query_len

		if wind > doc_np.shape[0]:
			length = doc_np.shape[0]
		else:
			length = wind

		Top_K = int(math.log(length))+1
		Con_Maxs = rolling_max(qd_cos, length, Top_K)#[query_len*(doc_len-45+1)*2] 
		Con_Score = np.sum(Con_Maxs, axis=0)#[(doc_len-45+1)*2]
		score_al = np.max(Con_Score[:,0] + alpha*np.mean(Con_Score,axis=1))*math.log(co+C)
		# score_al = np.max(np.array(Con_Score)[:,0] + alpha*(np.array(np.mean(np.array(Con_Score),axis=1))))*math.log(co+15)
		Doc_Score.append(score_al)
	except:
		Doc_Score.append(0.0)

	return Doc_Score
