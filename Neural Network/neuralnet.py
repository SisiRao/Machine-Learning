

"""
Author: sisir
Date: 9March2018
"""
import numpy as np
import csv
import sys
import math

def readFile(filename):
    data = []
    labels = []
    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile, delimiter=",")
        for line in csvreader:
            if line!=[]:
                labels.append(line[0])
                data.append(line[1:])
    
    #build model   
    data = np.array(data,dtype=np.float)
    labels = np.array(labels,dtype=np.float)   
    data = np.concatenate((np.ones((len(data),1),dtype=np.float), data), axis=1)
    N = len(data)
    K = len(np.unique(labels))  
    new_labels = np.zeros((N,K))
    for i in range(N):
        new_labels[i][int(labels[i])] =1   
    return data, labels,new_labels


def LinearForward(x,Alphas):
    b = np.dot(x,Alphas.T)
    return b

def SigmoidForward(A): 
    b = 1 / (1 + np.exp(-A))
    return b

def SoftmaxForward(B):
    b = np.exp(B)/np.sum(np.exp(B))
    return b

def CrossEntropyForward(y,y_hat):
    b = np.dot(-y.T,np.log(y_hat))
    return b

def NNForward(Alphas,x,Betas,y):
    A = LinearForward(x,Alphas)
    Z = SigmoidForward(A)
    Z = np.insert(Z,0,1)
    B = LinearForward(Z,Betas)
    y_hat = SoftmaxForward(B)
    J = CrossEntropyForward(y,y_hat)
    return x,A,Z,B,y_hat,J

def CrossEntropyBackward(y,y_hat):
    ga = -np.divide(y,y_hat)
    return ga

def SoftmaxBackward(y_hat,gy_hat):
    vec = y_hat.reshape((-1,1))
    ga = np.dot(gy_hat.T, (np.diag(y_hat)-np.dot(vec,vec.T)))
    return ga

def SigmoidBackward(Z,gz):
    ga = np.multiply(np.multiply(gz,Z),(1-Z))
    return ga

def NNBackward(x, z, b, beta, y_hat, y):
    gy_hat = CrossEntropyBackward(y,y_hat)
    gb = SoftmaxBackward(y_hat,gy_hat)
    gbt = gb.reshape((-1,1))
    gbeta = z*gbt
    gz = np.sum(gb*beta.T,axis=1)
    ga = SigmoidBackward(z,gz)
    galpha = x*ga.reshape((-1,1))
    galpha = np.delete(galpha, 0, 0)
    return gbeta, galpha

def SGD(data,data_v, Alphas, Betas,labels,labels_v,num_epoch,hidden_units,learning_rate,metrics_out):
    N,M = data.shape
    D = hidden_units               
    fw = open(metrics_out, 'a')


    for epoch in range(num_epoch):
        J_train = 0;
        J_val = 0;
        for i in range(N):
            # compute neural network layers
            x,A,Z,B,y_hat,J = NNForward(Alphas,data[i], Betas,labels[i])
            # compute gradients via backprop
            gbeta,galpha = NNBackward(data[i],Z,B,Betas,y_hat,labels[i])            # update parameters
            Alphas = Alphas - learning_rate*galpha
            Betas = Betas - learning_rate*gbeta
        # evaluate mean corss-entropy
        for i in range(N):
            # compute neural network layers
            Jj = NNForward(Alphas,data[i], Betas,labels[i])[5]
            J_train += Jj
        trainloss =  "epoch="+str(epoch+1)+" crossentropy(train): "+str(J_train/N)
        # print trainloss
        for i in range(len(data_v)):
            # compute neural network layers
            Jj = NNForward(Alphas,data_v[i], Betas,labels_v[i])[5]
            J_val += Jj
        valiloss = "epoch="+str(epoch+1)+" crossentropy(validation): "+str(J_val/len(data_v))
        # print valiloss

        # print str(J_train/N),",",str(J_val/len(data_v))
        fw.write(trainloss + '\n')
        fw.write(valiloss + '\n')

    return Alphas,Betas

def predict(data,labels,Alphas,Betas):
    p = []
    for i in range(len(data)):
        y_hat = NNForward(Alphas,data[i], Betas,labels[i])[4]
        p.append(y_hat.argmax())
    return p

def writeLabel(outfile, labels):
    fw = open(outfile,'w')
    for i in range(len(labels)):
        fw.write(str(labels[i])+'\n')

def main(): 
    train           = sys.argv[1]
    validation      = sys.argv[2]
    out_train       = sys.argv[3]
    out_val         = sys.argv[4]
    metrics_out     = sys.argv[5]
    num_epoch       = int(sys.argv[6])
    hidden_units    = int(sys.argv[7])
    init_flag       = int(sys.argv[8])
    learning_rate   = float(sys.argv[9])


    data,labels_index,labels= readFile(train)
    data_v,labels_v_index,labels_v= readFile(validation)

    if init_flag==2:
        Alphas = np.zeros((hidden_units,data.shape[1])) 
        Betas = np.zeros((labels.shape[1],hidden_units+1))

    if init_flag ==1:
        Alphas = np.random.uniform(-0.1,0.1,(hidden_units,data.shape[1]))
        Betas = np.random.uniform(-0.1,0.1,(labels.shape[1],hidden_units+1))

    Alphas,Betas = SGD(data,data_v, Alphas, Betas,labels,labels_v,num_epoch,hidden_units,learning_rate,metrics_out)
    p = predict(data,labels,Alphas,Betas)
    p_v = predict(data_v,labels_v,Alphas,Betas)


    count=0.0
    for row in zip(p,labels_index):
        if row[0]== row[1]:
            count+=1
    err_train =  'error(train): {:.2f}'.format(1-(count/len(p)))
    print err_train

    count=0.0
    for row in zip(p_v,labels_v_index):
        if row[0]== row[1]:
            count+=1
    err_val =  'error(validation): {:.2f}'.format(1-(count/len(p_v)))
    print err_val

    writeLabel(out_train,p)
    writeLabel(out_val,p_v)
    fw = open(metrics_out,'a')
    fw.write(err_train + '\n' + err_val)



if __name__== "__main__":
    main()