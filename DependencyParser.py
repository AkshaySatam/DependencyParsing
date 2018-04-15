import collections
import tensorflow as tf
import numpy as np
import pickle
import math
from progressbar import ProgressBar

from DependencyTree import DependencyTree
from ParsingSystem import ParsingSystem
from Configuration import Configuration
import Config
import Util

"""
This script defines a transition-based dependency parser which makes
use of a classifier powered by a neural network. The neural network
accepts distributed representation inputs: dense, continuous
representations of words, their part of speech tags, and the labels
which connect words in a partial dependency parse.

This is an implementation of the method described in

Danqi Chen and Christopher Manning. A Fast and Accurate Dependency Parser Using Neural Networks. In EMNLP 2014.

Author: Danqi Chen, Jon Gauthier
Modified by: Heeyoung Kwon (2017)
Modified by: Jun S. Kang (2018 Mar)
"""


class DependencyParserModel(object):

    def __init__(self, graph, embedding_array, Config):

        self.build_graph(graph, embedding_array, Config)

    def build_graph(self, graph, embedding_array, Config):
        """

        :param graph:
        :param embedding_array:
        :param Config:
        :return:
        """

        with graph.as_default():
            self.embeddings = tf.Variable(embedding_array, dtype=tf.float32,trainable=False)

            """
            ===================================================================

            Define the computational graph with necessary variables.
            
            1) You may need placeholders of:
                - Many parameters are defined at Config: batch_size, n_Tokens, etc
                - # of transitions can be get by calling parsing_system.numTransitions()
                
            
            ...self.train_inputs = 
            self.train_labels = 
            self.test_inputs =
            
                
            2) Call forward_pass and get predictions
            
            ...
            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)


            3) Implement the loss function described in the paper
             - lambda is defined at Config.lam
            
            ...
            self.loss =
            
            ===================================================================
            """


            #Defining the placeholders. Need to check on the sizes
            self.train_inputs = tf.placeholder(tf.int32, shape=(Config.batch_size,Config.n_Tokens))
            self.train_labels = tf.placeholder(tf.float32, shape=(Config.batch_size,parsing_system.numTransitions()))
            self.test_inputs = tf.placeholder(tf.int32, shape=(Config.n_Tokens))

            #Defining the parameters of the model as variables
            
            #weights_input = tf.Variable(tf.truncated_normal([Config.n_Tokens*Config.embedding_size,Config.hidden_size],stddev=1.0/math.sqrt(Config.embedding_size)))

	    #two layer changes
            #weights_input = tf.Variable(tf.random_normal([Config.n_Tokens*Config.embedding_size+Config.hidden_size,Config.hidden_size],stddev=0.1))
            #weights_input = tf.Variable(tf.random_normal([Config.n_Tokens*Config.embedding_size+1000,Config.hidden_size],stddev=0.1))
            weights_input = tf.Variable(tf.random_normal([Config.n_Tokens*Config.embedding_size,Config.hidden_size],stddev=0.1))
            
            #embed = tf.Variable(tf.truncated_normal(Config.batch_size,Config.n_Tokens*Config.embedding_size))            
            e = tf.nn.embedding_lookup(self.embeddings, self.train_inputs)
            embed = tf.reshape(e,[Config.batch_size,Config.n_Tokens * Config.embedding_size])
	    #two layer changes
            #biases_input = tf.Variable(tf.zeros(2 * Config.hidden_size))
            #biases_input = tf.Variable(tf.zeros(Config.hidden_size+1000))
            biases_input = tf.Variable(tf.zeros(Config.hidden_size))
            
            #weights_output = tf.Variable(tf.truncated_normal([parsing_system.numTransitions(),Config.hidden_size],stddev=1.0/math.sqrt(Config.embedding_size)))
            #Needs to be changed
            #weights_output = tf.Variable(tf.random_normal([parsing_system.numTransitions(),1000],stddev=0.1))
            weights_output = tf.Variable(tf.random_normal([parsing_system.numTransitions(),Config.hidden_size],stddev=0.1))

            self.prediction = self.forward_pass(embed, weights_input, biases_input, weights_output)

            #Changes to remove -1
            condition = tf.equal(self.train_labels, -1)
            case_true = tf.reshape(tf.zeros([Config.batch_size * parsing_system.numTransitions()], tf.float32),[Config.batch_size, parsing_system.numTransitions()]);
	    	#case_true = (tf.ones([Config.batch_size*parsing_system.numTransitions()],tf.int32),0)
	    	
	    	# case_true = tf.zeros([Config.batch_size,parsing_system.numTransitions()],tf.int32)
	    case_false = self.train_labels
	    newLabels = tf.where(condition, case_true, case_false)
			
            l2 = tf.nn.l2_loss(weights_input) + tf.nn.l2_loss(weights_output)+ tf.nn.l2_loss(biases_input) 
	    l2 = Config.lam / 2 * l2
	    ce = tf.nn.softmax_cross_entropy_with_logits_v2(_sentinel=None,labels=newLabels,logits=self.prediction,dim=-1,name=None)

            self.loss = tf.reduce_mean(ce+l2)

            optimizer = tf.train.GradientDescentOptimizer(Config.learning_rate)
            grads = optimizer.compute_gradients(self.loss)
            clipped_grads = [(tf.clip_by_norm(grad, 5), var) for grad, var in grads]
            self.app = optimizer.apply_gradients(clipped_grads)
            #self.app = optimizer.apply_gradients(grads)

            # For test data, we only need to get its prediction
            test_embed = tf.nn.embedding_lookup(self.embeddings, self.test_inputs)
            test_embed = tf.reshape(test_embed, [1, -1])
            self.test_pred = self.forward_pass(test_embed, weights_input, biases_input, weights_output)

            # intializer
            self.init = tf.global_variables_initializer()

    def train(self, sess, num_steps):
        """

        :param sess:
        :param num_steps:
        :return:
        """
        self.init.run()
        print "Initailized"

        average_loss = 0
        for step in range(num_steps):
            start = (step * Config.batch_size) % len(trainFeats)
            end = ((step + 1) * Config.batch_size) % len(trainFeats)
            if end < start:
                start -= end
                end = len(trainFeats)
            batch_inputs, batch_labels = trainFeats[start:end], trainLabels[start:end]

            feed_dict = {self.train_inputs: batch_inputs, self.train_labels: batch_labels}

            _, loss_val = sess.run([self.app, self.loss], feed_dict=feed_dict)
            average_loss += loss_val

            if step % Config.display_step == 0:
                if step > 0:
                    average_loss /= Config.display_step
                print "Average loss at step ", step, ": ", average_loss
                average_loss = 0
            if step % Config.validation_step == 0 and step != 0:
                print "\nTesting on dev set at step ", step
                predTrees = []
                for sent in devSents:
                    numTrans = parsing_system.numTransitions()

                    c = parsing_system.initialConfiguration(sent)
                    while not parsing_system.isTerminal(c):
                        feat = getFeatures(c)
                        pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                        optScore = -float('inf')
                        optTrans = ""

                        for j in range(numTrans):
                            if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                                optScore = pred[0, j]
                                optTrans = parsing_system.transitions[j]

                        c = parsing_system.apply(c, optTrans)

                    predTrees.append(c.tree)
                result = parsing_system.evaluate(devSents, predTrees, devTrees)
                print result

        print "Train Finished."

    def evaluate(self, sess, testSents):
        """

        :param sess:
        :return:
        """

        print "Starting to predict on test set"
        predTrees = []
        for sent in testSents:
            numTrans = parsing_system.numTransitions()

            c = parsing_system.initialConfiguration(sent)
            while not parsing_system.isTerminal(c):
                # feat = getFeatureArray(c)
                feat = getFeatures(c)
                pred = sess.run(self.test_pred, feed_dict={self.test_inputs: feat})

                optScore = -float('inf')
                optTrans = ""

                for j in range(numTrans):
                    if pred[0, j] > optScore and parsing_system.canApply(c, parsing_system.transitions[j]):
                        optScore = pred[0, j]
                        optTrans = parsing_system.transitions[j]

                c = parsing_system.apply(c, optTrans)

            predTrees.append(c.tree)
        print "Saved the test results."
        Util.writeConll('result_test.conll', testSents, predTrees)


	
    def forward_pass(self, embed, weights_input, biases_input, weights_output):
	embedArray = embed
	prod = tf.matmul(embedArray,weights_input)
	#Cube activation function
	t = tf.pow(tf.add(prod,biases_input, name = None),3)
	p = tf.matmul(weights_output,tf.transpose(t))
	return tf.transpose(p) 

	
    """def forward_pass(self, embed, weights_input, biases_input, weights_output):
	wordsEmbedding = embed[:,0:18*50]
	posEmbedding = embed[:,18*50:36*50]
	labelEmbedding = embed[:,36*50:]

	wordsWeights = weights_input[0:18*50,:]
	posWeights = weights_input[18*50:36*50,:]
	labelWeights = weights_input[36*50:,:]
	
	wordsBias = biases_input[0:200]
	posBias = biases_input[200:400]
	labelBias = biases_input[400:]
	
	prod1 = tf.matmul(wordsEmbedding,wordsWeights)
	prod2 = tf.matmul(posEmbedding,posWeights)
	prod3 = tf.matmul(labelEmbedding,labelWeights)

	t1 = tf.pow(tf.add(prod1,wordsBias, name = None),3)
	t2 = tf.pow(tf.add(prod2,posBias, name = None),3)
	t3 = tf.pow(tf.add(prod3,labelBias, name = None),3)

	print (t1)
	print t2, t3	
	t = tf.concat([t1, t2, t3],axis=1)
	print t	
	#t = np.concatenate((t4, t3))
	
	p = tf.matmul(weights_output,tf.transpose(t))
	print p
	return tf.transpose(p)


    def forward_pass(self, embed, weights_input, biases_input, weights_output):
	embedArray = embed
	w1 = weights_input[0:48*50,:]
	w2 = weights_input[48*50:3400,:]
	w3 = weights_input[3400:4400,:]
	b1 = biases_input[0:200]
	b2 = biases_input[200:1200]
	b3 = biases_input[1200:]

	prod1 = tf.matmul(embedArray,w1) #10000,200
	#Cube activation function
	t1 = tf.pow(tf.add(prod1,b1, name = None),3) #10000,200
	t2 = tf.matmul(t1,tf.transpose(w2)) #10000,1000
	t3 = tf.pow(tf.add(t2,b2, name = None),3) #10000,1000
	t4 = tf.matmul(t3,w3) #10000,200 		

	p = tf.matmul(t4,tf.transpose(weights_output))
	#return tf.transpose(p)		
	return p		
    	
    def forward_pass(self, embed, weights_input, biases_input, weights_output):
	embedArray = embed
	w1 = weights_input[0:48*50,:]
	w2 = weights_input[48*50:,:]
	b1 = biases_input[0:200]
	b2 = biases_input[200:]

	prod1 = tf.matmul(embedArray,w1)
	#Cube activation function
	t1 = tf.pow(tf.add(prod1,b1, name = None),3)
	t2 = tf.matmul(t1,tf.transpose(w2))
	t3 = tf.pow(tf.add(t2,b2, name = None),3)		

	p = tf.matmul(t3,tf.transpose(weights_output))
	#return tf.transpose(p)		
	return p
	"""	



def genDictionaries(sents, trees):
    word = []
    pos = []
    label = []
    for s in sents:
        for token in s:
            word.append(token['word'])
            pos.append(token['POS'])

    rootLabel = None
    for tree in trees:
        for k in range(1, tree.n + 1):
            if tree.getHead(k) == 0:
                rootLabel = tree.getLabel(k)
            else:
                label.append(tree.getLabel(k))

    if rootLabel in label:
        label.remove(rootLabel)

    index = 0
    wordCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    wordCount.extend(collections.Counter(word))
    for word in wordCount:
        wordDict[word] = index
        index += 1

    posCount = [Config.UNKNOWN, Config.NULL, Config.ROOT]
    posCount.extend(collections.Counter(pos))
    for pos in posCount:
        posDict[pos] = index
        index += 1

    labelCount = [Config.NULL, rootLabel]
    labelCount.extend(collections.Counter(label))
    for label in labelCount:
        labelDict[label] = index
        index += 1

    return wordDict, posDict, labelDict


def getWordID(s):
    if s in wordDict:
        return wordDict[s]
    else:
        return wordDict[Config.UNKNOWN]


def getPosID(s):
    if s in posDict:
        return posDict[s]
    else:
        return posDict[Config.UNKNOWN]


def getLabelID(s):
    if s in labelDict:
        return labelDict[s]
    else:
        return labelDict[Config.UNKNOWN]

def addElement(elementList,s1Word,i):
    elementList[i] = s1Word

def getFeatures(c):

    """
    =================================================================

    Implement feature extraction described in
    "A Fast and Accurate Dependency Parser using Neural Networks"(2014)

    =================================================================
    """
    elementList =[]
    i=0

    s1 = c.getStack(0)
    elementList.append(s1)
    s2 = c.getStack(1)
    elementList.append(s2)
    s3 = c.getStack(2)
    elementList.append(s3)

    b1 = c.getBuffer(0)
    elementList.append(b1)
    b2 = c.getBuffer(1)
    elementList.append(b2)
    b3 = c.getBuffer(2)
    elementList.append(b3)


    # s1 data
    s1_lc1 = c.getLeftChild(s1,1)
    elementList.append(s1_lc1)
    s1_rc1 = c.getRightChild(s1,1)
    elementList.append(s1_rc1)
    s1_lc2 = c.getLeftChild(s1,2)
    elementList.append(s1_lc2)
    s1_rc2 = c.getRightChild(s1,2)
    elementList.append(s1_rc2)

    # s2 data
    s2_lc1 = c.getLeftChild(s2,1)
    elementList.append(s2_lc1)
    s2_rc1 = c.getRightChild(s2,1)
    elementList.append(s2_rc1)
    s2_lc2 = c.getLeftChild(s2,2)
    elementList.append(s2_lc2)
    s2_rc2 = c.getRightChild(s2,2)
    elementList.append(s2_rc2)

    # leftmost/rightmost of rightmost children
    s1_lc1_lc1 = c.getLeftChild(s1_lc1,1)
    elementList.append(s1_lc1_lc1)
    s1_rc1_rc1 = c.getRightChild(s1_rc1,1)
    elementList.append(s1_rc1_rc1)

    s2_lc1_lc1 = c.getLeftChild (s2_lc1,1)
    elementList.append(s2_lc1_lc1)
    s2_rc1_rc1 = c.getRightChild (s2_rc1,1)
    elementList.append(s2_rc1_rc1)


    wordIDs = []
    labelIDs = []
    tagIDs  = []
    mergedList = []
   # print elementList

    for i in elementList:
        mergedList.append(getWordID(c.getWord(i)))

    for i in elementList:
        mergedList.append(getPosID(c.getPOS(i)))

    for i in elementList[6:]:
	#print "element: ",i
        mergedList.append(getLabelID(c.getLabel(i)))

    #mergedList = wordIDs + tagIDs + labelIDs
    # Method 1
    return mergedList

    
    """
    #Method 2
    row = 48, col = 50;
    j=0
    Matrix = [[0 for x in range(col)] for y in range(row)]

    for x in range (row):
        Matrix[x] = embedding_array[mergedList[j]]
        j+=1

    return Matrix
    """

    """
    #Words of the selected elements
    s1Word = c.getWord(s1)
    addElement(elementList,getWordID(s1Word),i)
    i+=1  
    s2Word = c.getWord(s2)
    addElement(elementList,getWordID(s2Word),i)
    i+=1
    s3Word = c.getWord(s3)
    addElement(elementList,getWordID(s3Word),i)
    i+=1

    b1Word = c.getWord(b1)
    addElement(elementList,getWordID(b1Word),i)
    i+=1
    b2Word = c.getWord(b2)
    addElement(elementList,getWordID(b2Word),i)
    i+=1
    b3Word = c.getWord(b3)
    addElement(elementList,getWordID(b3Word),i)
    i+=1

    s1_lc1Word = c.getWord(s1_lc1)
    addElement(elementList,getWordID(s1_lc1Word),i)
    i+=1
    s1_rc1Word = c.getWord(s1_rc1)
    addElement(elementList,getWordID(s1_rc1Word),i)
    i+=1
    s1_lc2Word = c.getWord(s1_lc2)
    addElement(elementList,getWordID(s1_lc2Word),i)
    i+=1
    s1_rc2Word = c.getWord(s1_rc2)
    addElement(elementList,getWordID(s1_rc2Word),i)
    i+=1

    s2_lc1Word = c.getWord(s2_lc1)
    addElement(elementList,getWordID(s2_lc1Word),i)
    i+=1
    s2_rc1Word = c.getWord(s2_rc1)
    addElement(elementList,getWordID(s2_rc1Word),i)
    i+=1
    s2_lc2Word = c.getWord(s2_lc2)
    addElement(elementList,getWordID(s2_lc2Word),i)
    i+=1
    s2_rc2Word = c.getWord(s2_rc2)
    addElement(elementList,getWordID(s2_rc2Word),i)
    i+=1

    s1_lc1_lc1Word = c.getWord(s1_lc1_lc1)
    addElement(elementList,getWordID(s1_lc1_lc1Word),i)
    i+=1
    s1_rc1_rc1Word = c.getWord(s1_rc1_rc1)
    addElement(elementList,getWordID(s1_rc1_rc1Word),i)
    i+=1
    s2_lc1_lc1Word = c.getWord(s2_lc1_lc1)
    addElement(elementList,getWordID(s2_lc1_lc1Word),i)
    i+=1
    s2_rc1_rc1Word = c.getWord(s2_rc1_rc1)
    addElement(elementList,getWordID(s2_rc1_rc1Word),i)
    i+=1


    #POS tags of the selected elements
    s1POS = c.getPOS(s1)
    addElement(elementList,getPosID(s1POS),i)
    i+=1
    s2POS = c.getPOS(s2)
    addElement(elementList,getPosID(s2POS),i)
    i+=1
    s3POS = c.getPOS(s3)
    addElement(elementList,getPosID(s3POS),i)
    i+=1

    b1POS = c.getPOS(b1)
    addElement(elementList,getPosID(b1POS),i)
    i+=1
    b2POS = c.getPOS(b2)
    addElement(elementList,getPosID(b2POS),i)
    i+=1
    b3POS = c.getPOS(b3)
    addElement(elementList,getPosID(b3POS),i)
    i+=1

    s1_lc1POS = c.getPOS(s1_lc1)
    addElement(elementList,getPosID(s1_lc1POS),i)
    i+=1
    s1_rc1POS = c.getPOS(s1_rc1)
    addElement(elementList,getPosID(s1_rc1POS),i)
    i+=1
    s1_lc2POS = c.getPOS(s1_lc2)
    addElement(elementList,getPosID(s1_lc2POS),i)
    i+=1
    s1_rc2POS = c.getPOS(s1_rc2)
    addElement(elementList,getPosID(s1_rc2POS),i)
    i+=1

    s2_lc1POS = c.getPOS(s2_lc1)
    addElement(elementList,getPosID(s2_lc1POS),i)
    i+=1
    s2_rc1POS = c.getPOS(s2_rc1)
    addElement(elementList,getPosID(s2_rc1POS),i)
    i+=1
    s2_lc2POS = c.getPOS(s2_lc2)
    addElement(elementList,getPosID(s2_lc2POS),i)
    i+=1
    s2_rc2POS = c.getPOS(s2_rc2)
    addElement(elementList,getPosID(s2_rc2POS),i)
    i+=1

    s1_lc1_lc1POS = c.getPOS(s1_lc1_lc1)
    addElement(elementList,getPosID(s1_lc1_lc1POS),i)
    i+=1
    s1_rc1_rc1POS = c.getPOS(s1_rc1_rc1)
    addElement(elementList,getPosID(s1_rc1_rc1POS),i)
    i+=1
    s2_lc1_lc1POS = c.getPOS(s2_lc1_lc1)
    addElement(elementList,getPosID(s2_lc1_lc1POS),i)
    i+=1
    s2_rc1_rc1POS = c.getPOS(s2_rc1_rc1)
    addElement(elementList,getPosID(s2_rc1_rc1POS),i)
    i+=1
    


    #Labels of the selected elements
    s1_lc1Label = c.getLabel(s1_lc1)
    addElement(elementList,getLabelID(s1_lc1Label)s,i)
    i+=1
    s1_rc1Label = c.getLabel(s1_rc1)
    addElement(elementList,getLabelID(s1_rc1Label),i)
    i+=1
    s1_lc2Label = c.getLabel(s1_lc2)
    addElement(elementList,getLabelID(s1_lc2Label),i)
    i+=1
    s1_rc2Label = c.getLabel(s1_rc2)
    addElement(elementList,getLabelID(s1_rc2Label),i)
    i+=1

    s2_lc1Label = c.getLabel(s2_lc1)
    addElement(elementList,getLabelID(s2_lc1Label),i)
    i+=1
    s2_rc1Label = c.getLabel(s2_rc1)
    addElement(elementList,getLabelID(s2_rc1Label),i)
    i+=1
    s2_lc2Label = c.getLabel(s2_lc2)
    addElement(elementList,getLabelID(s2_lc2Label),i)
    i+=1
    s2_rc2Label = c.getLabel(s2_rc2)
    addElement(elementList,getLabelID(s2_rc2Label),i)
    i+=1

    s1_lc1_lc1Label = c.getLabel(s1_lc1_lc1)
    addElement(elementList,getLabelID(s1_lc1_lc1Label),i)
    i+=1
    s1_rc1_rc1Label = c.getLabel(s1_rc1_rc1)
    addElement(elementList,getLabelID(s1_rc1_rc1Label),i)
    i+=1
    s2_lc1_lc1Label = c.getLabel(s2_lc1_lc1)
    addElement(elementList,getLabelID(s2_lc1_lc1Label),i)
    i+=1
    s2_rc1_rc1Label = c.getLabel(s2_rc1_rc1)
    addElement(elementList,getLabelID(s2_rc1_rc1Label),i)
    i+=1
    """
    """
    row = 48, col = 50;
    j=0
    Matrix = [[0 for x in range(col)] for y in range(row)]

    for x in range (row):
        Matrix[x] = elementList[j]
        j+=1 
    """
    """
    return elementList
    """


def genTrainExamples(sents, trees):
    numTrans = parsing_system.numTransitions()

    features = []
    labels = []
    pbar = ProgressBar()
    for i in pbar(range(len(sents))):
    #for i in pbar(range(1000)):
        if trees[i].isProjective():
            c = parsing_system.initialConfiguration(sents[i])

            while not parsing_system.isTerminal(c):
                oracle = parsing_system.getOracle(c, trees[i])
                feat = getFeatures(c)
                label = []
                for j in range(numTrans):
                    t = parsing_system.transitions[j]
                    if t == oracle:
                        label.append(1.)
                    elif parsing_system.canApply(c, t):
                        label.append(0.)
                    else:
                        label.append(-1.)

                if 1.0 not in label:
                    print i, label
                features.append(feat)
                labels.append(label)
                c = parsing_system.apply(c, oracle)
    return features, labels


def load_embeddings(filename, wordDict, posDict, labelDict):
    dictionary, word_embeds = pickle.load(open(filename, 'rb'))

    embedding_array = np.zeros((len(wordDict) + len(posDict) + len(labelDict), Config.embedding_size))
    knownWords = wordDict.keys()
    foundEmbed = 0
    for i in range(len(embedding_array)):
        index = -1
        if i < len(knownWords):
            w = knownWords[i]
            if w in dictionary:
                index = dictionary[w]
            elif w.lower() in dictionary:
                index = dictionary[w.lower()]
        if index >= 0:
            foundEmbed += 1
            embedding_array[i] = word_embeds[index]
        else:
            embedding_array[i] = np.random.rand(Config.embedding_size) * 0.02 - 0.01
    print "Found embeddings: ", foundEmbed, "/", len(knownWords)

    return embedding_array


if __name__ == '__main__':

    wordDict = {}
    posDict = {}
    labelDict = {}
    parsing_system = None

    trainSents, trainTrees = Util.loadConll('train.conll')
    devSents, devTrees = Util.loadConll('dev.conll')
    testSents, _ = Util.loadConll('test.conll')
    genDictionaries(trainSents, trainTrees)

    embedding_filename = 'word2vec.model'

    embedding_array = load_embeddings(embedding_filename, wordDict, posDict, labelDict)

    labelInfo = []
    for idx in np.argsort(labelDict.values()):
        labelInfo.append(labelDict.keys()[idx])
    parsing_system = ParsingSystem(labelInfo[1:])
    print parsing_system.rootLabel

    print "Generating Traning Examples"
    trainFeats, trainLabels = genTrainExamples(trainSents, trainTrees)
    print "Done."

    # Build the graph model
    graph = tf.Graph()
    model = DependencyParserModel(graph, embedding_array, Config)

    num_steps = Config.max_iter
    with tf.Session(graph=graph) as sess:

        model.train(sess, num_steps)

        model.evaluate(sess, testSents)

