#!/usr/bin/env python3.8
import argparse
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use(['fast'])
from sklearn.model_selection import train_test_split
import tensorflow as tf
import scipy 
print(f'using tf version {tf.__version__}')
from sklearn.preprocessing import StandardScaler
import datetime

parser = argparse.ArgumentParser()


# add args and just return them

parser.add_argument('-f','--file',dest='input',type=str,required=True,help="Path to cleaned input file")
parser.add_argument('-sc','--scale',dest='scale',action='store_true',help="Whether or not to scale the input data. defaults to False")
parser.add_argument('-n','--nodes',dest='nodes',type=int,default=50,help="Number of nodes to be used at hidden layer. defaults to ol reliable 50")
parser.add_argument('-ep','--epochs',dest='epochs',type=int,default=300,help="Number of iterations or epochs to be run. defaults to 300 for testing")
parser.add_argument('-afn','--activation-function',dest='activation',type=str,default='relu',help="activation function to use at hidden layer, which should be one available from the list of possible arguments. defaults to relu")
parser.add_argument('-name','--name',dest='name',type=str,help="number to add to end of model name (needed for repeat experiments)")

parser.set_defaults(scale=True)

args=parser.parse_args()



data=pd.read_csv(f"{args.input}",sep='\t',index_col=0)
regions = data.columns
print(f'data dims are {data.shape}')

sc = StandardScaler()
if args.scale:
	scale_data = sc.fit_transform(data)
else:
	scale_data=data

# the following is defining all functions - there may be more point to just writing modules to achieve this
def create_ae(bottleneck,scale_data):
    model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(bottleneck,input_shape=(scale_data.shape[1],),activation=args.activation),
    tf.keras.layers.Dense(scale_data.shape[1],activation='linear')
    ])
    model.compile(loss='mse',optimizer=tf.keras.optimizers.Adam(learning_rate=0.0007))
    return model



def training_loop(ae,epochs,noise_std,scale_data,verbose=True):
    for i in range(epochs):
        noise = np.random.normal(0.0,noise_std,(scale_data.shape[0],scale_data.shape[1]))

        noisy_data=scale_data+noise
        
        if verbose:
            print(f'epoch {i+1}')
            ae.fit(noisy_data, scale_data, epochs = 1, verbose = 1,batch_size=32)
        else:
           # print(f'epoch {i+1}')
            h=ae.fit(noisy_data, scale_data, epochs = 1, verbose = 0,batch_size=32)
    
    if verbose==False:
        fl=h.history['loss']
        print(f'final loss={fl}')
            
    
    return ae

def get_layer_name(model):
    return [layer.name for layer in model.layers if 'dense' in layer.name][0]

def get_activations(model,layer,data):
    repr_model = tf.keras.models.Model(
       [model.input],[model.get_layer(layer).output,model.output]
    )
    
    dims,reconstructions=repr_model(tf.Variable(data,np.float32))
    return dims


def get_weights_df_labs(model,regions):
    w_vec=model.get_weights()[0]
    labs=[]
    for i in range(w_vec.shape[1]):
        labs.append(f'node_{i+1}')
    weights_df=pd.DataFrame(w_vec)
    weights_df.index=regions
    weights_df.columns=labs
    return weights_df,labs



# bookmarked 


model=create_ae(args.nodes,scale_data)

mod=training_loop(model,args.epochs,0.0005,scale_data,False)
    
layer=get_layer_name(mod)

activations = get_activations(mod,layer,scale_data)

weights_df,labs = get_weights_df_labs(mod,regions)

node_activation_df=pd.DataFrame(activations.numpy(),columns=labs)

# save results to date 
timestamp=datetime.datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
if args.name is not None:
	outname = f'{timestamp}_{args.name}'
else:
	outname = f'{timestamp}'
model.save(f'autoencoder_{outname}.h5')
weights_df.to_csv(f'weights_df_{outname}.csv',sep='\t')
node_activation_df.to_csv(f'node_activation_df_{outname}.csv',sep='\t')

print('done :)')
