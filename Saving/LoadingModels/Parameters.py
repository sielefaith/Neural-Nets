import numpy as np
import nnfs
from nnfs.datasets import spiral_data, sine_data
import cv2
import os
import pickle

nnfs.init()

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):
		self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		
		#set regularization strength
		self.weight_regularizer_L1 = weight_regularizer_L1
		self.weight_regularizer_L2 = weight_regularizer_L2
		self.bias_regularizer_L1 = bias_regularizer_L1
		self.bias_regularizer_L2 = bias_regularizer_L2
		
	def forward(self, inputs,training):
		self.inputs = inputs
		self.output = np.dot(inputs, self.weights) + self.biases
	
	#Backward pass
	def backward(self, dvalues):
		 #Gradients on parameters
		 self.dweights = np.dot(self.inputs.T, dvalues)
		 self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
		 
		 #Gradients on regularization
		 #L1 on weights
		 if self.weight_regularizer_L1 > 0:
		 	dL1 = np.ones_like(self.weights)
		 	dL1[self.weights < 0] = -1
		 	self.dweights += self.weight_regularizer_L1 * dL1
		 #L2 on weights
		 if self.weight_regularizer_L2 > 0:
		 	self.dweights += 2*self.weight_regularizer_L2 * self.weights
		 
		 #L1 on biases
		 if self.bias_regularizer_L1 > 0:
		 	dL1 = np.ones_like(self.biases)
		 	dL[self.biases < 0] = -1
		 	self.dbiases += self.bias_regularizer_L1 * dL1
		 	
		 #L2 on biases
		 if self.bias_regularizer_L2 > 0:
		 	self.dbiases += 2 * self.bias_regularizer_L2 * self.biases
		 	
		 #Gradient on values
		 self.dinputs = np.dot(dvalues, self.weights.T)
	
	def get_parameters(self):
		 return self.weights, self.biases
		 
	#set weights and biases in a layer instance
	def set_parameters(self, weights, biases):
		self.weights = weights
		self.biases = biases
		
#Dropout
class Layer_Dropout:
	#init
	def __init__(self, rate):
		 #store rate, we invert it as for example for dropout of  0.1 we need success rate of 0.9
		self.rate = 1 - rate
	#forward pass
	def forward(self, inputs, training):
		#save input values
		self.inputs = inputs
		#if not in the training mode - return values
		if not training:
			self.output = inputs.copy()
			return
		#generate n save scaled mask
		self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
		#apply mask to output values
		self.output = inputs * self.binary_mask
	#Backward pass
	def backward(self, dvalues):
		#gradient on values
		self.dinputs = dvalues * self.binary_mask

#Input "layer"		 
class Layer_Input:
	#forward pass
	def forward(self, inputs, training):
		 self.output = inputs
		 
class Activation_ReLU:
	def forward(self, inputs, training):
		 self.inputs = inputs
		 self.output = np.maximum(0, inputs)
		 
	def backward(self, dvalues):
		 self.dinputs = dvalues.copy()
		 self.dinputs[self.inputs<=0]=0
		 
	#calculate predictions for outputs
	def predictions (self, outputs):
		return outputs
		
class Activation_Softmax:
	def forward(self, inputs, training):
	    self.inputs = inputs
	    exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
	    probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
	    self.output = probabilities
		 
	def backward(self, dvalues):
		 self.dinputs = np.empty_like(dvalues)
		 
		 for index, (single_output, single_dvalues) in enumerate(zip(self.output, dvalues)):
		 	single_output = single_output.reshape(-1, 1)
		 	jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
		 	self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
	
	def predictions (self, outputs):
		 return np.argmax(outputs, axis=1)	 
	#calc. predictions for outputs
	def predictions(self, outputs):
		return np.argmax(outputs, axis=1)
		
#Sigmoid Activation
class Activation_Sigmoid:
	#forward pass
	def forward(self, inputs, training):
		#save input n calc / save output
		#of the sigmoid function
		self.inputs = inputs
		self.output = 1 / (1 + np.exp(-inputs))
	#Backward pass
	def backward(self, dvalues):
		#derivatives - calc. from output of sigmoid function
		self.dinputs = dvalues * (1 - self.output) * self.output 
		#calc. predictions for outputs
	def predictions (self, outputs):
		return (outputs > 0.5)*1
		
#Linear Activation
class Activation_Linear:
	#Forward pass
	def forward(self, inputs, training):
		#Just remember values
		self.inputs = inputs
		self.output = inputs
	#Backward pass
	def backward(self, dvalues):
		#derivative=1, 1*dvalues - chain rule
		self.dinputs = dvalues.copy() 	
	#calc. predictions for outputs
	def predictions(self, outputs):
		return outputs
		######
		#OPTIMIZERS
			
class Optimizer_SGD:
	def __init__(self, learning_rate=1., decay=0.,momentum=0.):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.momentum = momentum
		
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1./(1. + self.decay * self.iterations))
	def update_params(self, layer):
		if self.momentum:
			if not hasattr(layer,'weight_momentums'):
				layer.weight_momentums = np.zeros_like(layer.weights)
				layer.bias_momentums = np.zeros_like(layer.biases)
				
			weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
			layer.weight_momentums = weight_updates
			
			bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
			layer.bias_momentums = bias_updates
		else:
			weight_updates = -self.current_learning_rate * layer.dweights
			bias_updates = -self.current_learning_rate * layer.dbiases
			
		layer.weights += weight_updates
		layer.biases += bias_updates
		
	def post_update_params(self):
		self.iterations += 1
		
#Adagrad Optimizer
class Optimizer_Adagrad:
	#Initialize Optimizer - Set Settings
	def __init__(self, learning_rate = 1.,decay = 0.,epsilon = 1e-7):
		  self.learning_rate = learning_rate
		  self.current_learning_rate = learning_rate
		  self.decay = decay
		  self.iterations = 0
		  self.epsilon = epsilon
		  
	#Call once before any parameter updates
	def pre_update_params(self):
		  if self.decay:
		  	self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
		  	
	#Update parameters	  	
	def update_params(self, layer):
		  #If layer doesn't contain cache arrays
		  #Create them filled with zeros
		  if not hasattr(layer, 'weight_cache'):
		  	layer.weight_cache = np.zeros_like(layer.weights)
		  	layer.bias_cache = np.zeros_like(layer.biases)
		  #Update cache with squared current gradients	
		  layer.weight_cache += layer.dweights**2
		  layer.bias_cache += layer.dbiases**2
		  
		  #Vanilla SGD parameter update + normalization
		  # with square rooted cache
		  layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		  layer.biases += -self.current_learning_rate * layer.dbiases /(np.sqrt(layer.bias_cache) + self.epsilon)
	
	#Call once after any parameter updates
	def post_update_params(self):
		self.iterations += 1
#RMSprop optimizer'		
class Optimizer_RMSprop:
	def __init__(self, learning_rate=0.001, decay=0., epsilon=1e-7, rho=0.9):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.rho = rho
		
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. / (1. + self.decay * self.iterations))
			
	def update_params(self, layer):
		if  not hasattr(layer, 'weight_cache'):
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_cache = np.zeros_like(layer.biases)
		#update cache with squared current gradients
		layer.weight_cache = self.rho * layer.weight_cache + (1 - self.rho) * layer.dweights**2
		layer.bias_cache = self.rho * layer.bias_cache + (1 - self.rho) * layer.dbiases **2
		
		layer.weights += -self.current_learning_rate * layer.dweights / (np.sqrt(layer.weight_cache) + self.epsilon)
		layer.biases += -self.current_learning_rate * layer.dbiases / (np.sqrt(layer.bias_cache) + self.epsilon)
	
	def post_update_params(self):
		self.iterations += 1
		
#Adam optimizer
class Optimizer_Adam:
	#Initialize optimizer - set settings
	def __init__(self, learning_rate=0.001, decay=0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.iterations = 0
		self.epsilon = epsilon
		self.beta_1 = beta_1
		self.beta_2 = beta_2
		
	#Call once before any parameter updates
	def pre_update_params(self):
		if self.decay:
			self.current_learning_rate = self.learning_rate * (1. /(1. + self.decay * self.iterations))
	#update parameters
	def update_params(self, layer):
		#if layer does not contain cache arrays 
		#create them filled with zeros
		if not hasattr(layer, 'weight_cache'):
			layer.weight_momentums = np.zeros_like(layer.weights)
			layer.weight_cache = np.zeros_like(layer.weights)
			layer.bias_momentums = np.zeros_like(layer.biases)
			layer.bias_cache = np.zeros_like(layer.biases)
		#update moment with current gradients
		layer.weight_momentums = self.beta_1 * layer.weight_momentums + (1 - self.beta_1) * layer.dweights
		layer.bias_momentums = self.beta_1 * layer.bias_momentums + (1 - self.beta_1) * layer.dbiases
		
		#Get corrected momentum
		#self.iteration is 0 at first pass
		#and we need to start with 1 here
		weight_momentums_corrected = layer.weight_momentums / (1 - self.beta_1**(self.iterations + 1))
		bias_momentums_corrected = layer.bias_momentums / (1 - self.beta_1 ** (self.iterations + 1))
		
		#update cache with squared current gradient
		layer.weight_cache = self.beta_2 * layer.weight_cache + (1 - self.beta_2) * layer.dweights**2
		layer.bias_cache = self.beta_2 * layer.bias_cache + (1 - self.beta_2) *layer.dbiases ** 2
		
		#Get corrected cache
		weight_cache_corrected = layer.weight_cache / (1 - self.beta_2 ** (self.iterations + 1))
		bias_cache_corrected = layer.bias_cache / (1 - self.beta_2 ** (self.iterations + 1))
		
		#Vanilla SGD + normalization
		#sqrt cache
		layer.weights += -self.current_learning_rate * weight_momentums_corrected /(np.sqrt(weight_cache_corrected) + self.epsilon)
		layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)
	
	#call once after any parameter updates
	def post_update_params(self):
		self.iterations += 1
		
		#######
class Loss:
	def regularization_loss(self):
		# 0 by default
		regularization_loss = 0
		#calculate regularization loss
		#iterate all trainable layers
		for layer in self.trainable_layers:
			
			#L1 regularization_weights
			#calculate only when factor greater than 0
			if layer.weight_regularizer_L1 > 0:
				regularization_loss += layer.weight_regularizer_L1 * np.sum(np.abs(layer.weights))
			
			#L2 regularization _weights
			if layer.weight_regularizer_L2 > 0:
				regularization_loss += layer.weight_regularizer_L2 * np.sum(layer.weights * layer.weights)
			
			#L1 regularization_biases
			#calculate only when factor greater than 0
			if layer.bias_regularizer_L1 > 0 :
				regularization_loss += layer.bias_regularizer_L1 * np.sum(np.abs(layer.biases))
			
			#L2 regularization_biases
			if layer.bias_regularizer_L2 > 0 :
				regularization_loss += layer.bias_regularizer_L2 * np.sum(layer.biases * layer.biases)
		return regularization_loss
			
	#Set / remember trainable layers
	def remember_trainable_layers(self, trainable_layers):
		self.trainable_layers = trainable_layers
		
	#Calculate data and regularization losses
	#given model output and ground truth values
	def calculate(self, output, y, *, include_regularization = False):
		 #calculate sample losses
		 sample_losses = self.forward(output, y)
		 #calculate mean loss
		 data_loss = np.mean(sample_losses)
		 #Add accumulated sum of losses and samples count
		 self.accumulated_sum += np.sum(sample_losses)
		 self.accumulated_count += len(sample_losses)
		 #if just data_loss - return it
		 if not include_regularization:
		 	#Return the data_loss and regularization losses
		 	return data_loss
		 	#Return the data and regularization loss
		 return data_loss, self.regularization_loss()
	#Calculate accumulated mean loss	 
	def calculate_accumulated(self,*,include_regularization=False):
		 #calculate mean loss
		 data_loss = self.accumulated_sum / self.accumulated_count
		 #if just data loss - return it
		 if not include_regularization:
		 	return data_loss
		 return data_loss, self.regularization_loss()
		 
	#Reset variables for accumulated loss
	def new_pass(self):
		 self.accumulated_sum = 0
		 self.accumulated_count = 0
		 
		 
class Loss_CategoricalCrossentropy(Loss):
	def forward(self, y_pred, y_true):
		 samples = len(y_pred)
		 y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		 
		 if len(y_true.shape) == 1:
		 	correctconfidence = y_pred_clipped[range(samples), y_true]
		 elif len(y_true.shape) == 2:
		 	correctconfidence = np.sum(y_pred_clipped * y_true, axis=1)
		 	
		 neg_log_likelihood = - np.log(correctconfidence)
		 return neg_log_likelihood
		 
	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		labels = len(dvalues[0])
		
		if len(y_true.shape) == 1:
			y_true = np.eye(labels)[y_true]
		self.dinputs = -y_true / dvalues
		self.dinputs = self.dinputs / samples

#Softmax classifier - combined softmax activation 
#and cross-entropy loss for faster backward step		
class Activation_Softmax_Loss_CategoricalCrossentropy():
	#Backward Pass
	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)
		self.dinputs = dvalues.copy()
		self.dinputs[range(samples), y_true] -=1
		self.dinputs = self.dinputs / samples
		
#Binary cross-entropy loss
class Loss_BinaryCrossentropy(Loss):
	#forward pass
	def forward(self, y_pred, y_true):
		#clip data to prevent division by 0
		#clip both sides to not drag mean towards any value
		y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
		#calculate sample-wise loss
		sample_losses = -(y_true * np.log(y_pred_clipped) + (1 - y_true) * np.log (1 - y_pred_clipped))
		sample_losses = np.mean(sample_losses, axis=-1)
		return sample_losses
		
	#Backward Pass
	def backward(self, dvalues, y_true):
		#Number of samples
		samples = len(dvalues)
		#number of outputs in every sample
		#use first sample to count
		outputs = len (dvalues[0])
		
		clipped_dvalues = np.clip(dvalues, 1e-7, 1 - 1e-7)
		#calculate gradient
		self.dinputs = -(y_true / clipped_dvalues - (1 - y_true) / (1 - clipped_dvalues)) / outputs
		#normalize gradient
		self.dinputs = self.dinputs / samples
		
#Mean Squared Error Loss
#L2 Loss
class Loss_MeanSquaredError(Loss):
	#Forward Pass
	def forward(self, y_pred, y_true):
		#calculate loss
		sample_losses = np.mean((y_true - y_pred)**2, axis=-1)
		#return losses
		return sample_losses
		
	#Backward Pass
	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		outputs = len(dvalues[0])
		self.dinputs = -2*(y_true - dvalues) / outputs
		self.dinputs = self.dinputs / samples
		
#Mean Absolute Error Loss
#L1 Loss
class Loss_MeanAbsoluteError(Loss):
	def forward(self, y_pred, y_true):
		#calculate loss
		sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)
		#Return loss
		return sample_losses
		
	#Backward pass
	def backward(self, dvalues, y_true):
		#number of samples
		samples = len(dvalues)
		#number of outputs in each sample
		#use the first sample to count
		outputs = len(dvalues[0])
		
		# Calculate gradient
		self.dinputs = np.sign(y_true - dvalues) / outputs
		#normalize
		self.dinputs = self.dinputs / samples
		
####

#Common accuracy class
class Accuracy:
	#calc. accuracy
	#given predictions & ground truth values
	def calculate(self, predictions, y):
		#Get comparison results
		comparisons = self.compare(predictions, y)
		#calc. accuracy
		accuracy = np.mean(comparisons)
		self.accumulated_sum += np.sum(comparisons)
		self.accumulated_count += len(comparisons)
		#Return accuracy
		return accuracy
		
	def calculate_accumulated(self):
		accuracy = self.accumulated_sum / self.accumulated_count
		return accuracy
		
	def new_pass(self):
		self.accumulated_sum = 0
		self.accumulated_count = 0
		
#Accuracy calculation for regression model
class Accuracy_Regression(Accuracy):
	def __init__(self):
		#create precision property
		self.precision = None
		
	#calc. presicion value
	#based on passed in ground truth
	def init(self, y, reinit=False):
		if self.precision is None or reinit:
			self.precision = np.std(y) /250
			
	#compare predictions to ground truth values
	def compare(self, predictions, y):
		return np.absolute(predictions - y) < self.precision
		
#Accuracy calc. for classification model
class Accuracy_Categorical(Accuracy):
	def __init__(self, *, binary = False):
	    #Binary model
	    self.binary = binary
	#no initialization is needed
	def init(self, y):
		pass
		
	#compress predictions to the ground truth values
	def compare(self, predictions, y):
		if not self.binary and len(y.shape) == 2:
			y = np.argmax(y, axis=1)
		return predictions == y
	
#Model class
class Model:
	def __init__(self):
		#create a list of network objects
		self.layers = []
		#softmax classifier's output object
		self.softmax_classifier_output = None
		
	#Add objects to the model
	def add(self, layer):
		self.layers.append(layer)
		
	def set(self, *, loss=None, optimizer=None, accuracy=None):
		if loss is not None:
			self.loss = loss
		if optimizer is not None:
			self.optimizer = optimizer
		if accuracy is not None:
			self.accuracy = accuracy
	
	#Finalize the model	
	def finalize(self):
		#create and set the input layer
		self.input_layer = Layer_Input()
		#count all the objects
		layer_count = len(self.layers)
		#Initialize a list containing trainable layers
		self.trainable_layers = []

		#Iterate the objects
		for i in range (layer_count):
			#If it's the 1st layer
			#The previous layer object is the input layer
			if i == 0:
				self.layers[i].prev = self.input_layer
				self.layers[i].next = self.layers[i + 1]
			#All layers except for the first and last
			elif i < layer_count - 1:
				self.layers [i].prev = self.layers [i - 1]
				self.layers[i].next = self.layers [i + 1]
			#The last layer - the next object is the loss
			#Also let's save aside the reference to the last object
			#Where output is the model's output
			else:
				self.layers[i].prev = self.layers[i - 1]
				self.layers[i].next = self.loss
				self.output_layer_activation = self.layers[i]
				
			#If layer contains an attribute called 'weights'
			#It's a trainable layer
			#Add it to the list of trainable layers
			#We don't need to check for biases
			#Checking for weights is enough
			if hasattr(self.layers[i],'weights'):
				self.trainable_layers.append(self.layers[i])
		#update loss object with trainable layers
		
		
		if self.loss is not None:
			self.loss.remember_trainable_layers(self.trainable_layers)
	
		if isinstance(self.layers[-1], Activation_Softmax) and isinstance (self.loss, Loss_CategoricalCrossentropy):
			#create an object of combined activation and loss functions
			self.softmax_classifier_output = Activation_Softmax_Loss_CategoricalCrossentropy()
	
	#Train the model			
	def train (self, X, y, *, epochs=1,batch_size=None, print_every = 1, validation_data=None):
		#Initialize accuracy object
		self.accuracy.init(y)
		#Default value if batch size is not set
		train_steps = 1
		#if there's validation data passed
		#set default number of steps for validation as well
		if validation_data is not None:
			validation_steps = 1
			#For better readability
			X_val, y_val = validation_data
		#Calculate number of steps
		if batch_size is not None:
			train_steps = len(X) // batch_size
			#Dividing rounds down.If there are some remaining
			#data, but not a full batch, this won't include it
			#Add 1 to include this not full batch
			if train_steps * batch_size < len(X):
				train_steps += 1
			if validation_data is not None:
				validation_steps = len(X_val) // batch_size
				#Dividing rounds down.If there are some remaining
				#data, but not a full batch, this won't include it
				#Add 1 to include this not full batch
				if validation_steps * batch_size < len (X_val):
					validation_steps += 1
					
		
		#Main training loop
		for epoch in range(1, epochs+1):
			#Print epoch number
			print(f'epoch:{epoch}')
			#reset accumulated values in loss and accuracy objects
			self.loss.new_pass()
			self.accuracy.new_pass()
			#Iterate over steps
			for step in range(train_steps):
				#if batch size is not set-
				#train using one step and full dataset
				if batch_size is None:
					batch_X = X
					batch_y = y
				#otherwise slice a batch
				else:
					batch_X = X[step*batch_size: (step + 1)*batch_size]
					batch_y = y[step * batch_size : (step + 1) * batch_size]
				#Perform the forward pass
				output = self.forward(batch_X, training=True)
				#Calculate loss
				data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization = True)
				loss = data_loss + regularization_loss
			
				#Get predictions and calculate accuracy
				predictions = self.output_layer_activation.predictions(output)
				accuracy = self.accuracy.calculate(predictions,batch_y)
				#perform backward pass
				self.backward(output, batch_y)
			
				#update weights n biases

				self.optimizer.pre_update_params()
				for layer in self.trainable_layers:
					self.optimizer.update_params(layer)
				self.optimizer.post_update_params()
			
			#print a summary
				if not step % print_every or step == train_steps - 1:
					print(f'step : {step}, ' + f'acc : {accuracy: .3f}, ' + f'loss : {loss: .3f}('+f'data_loss :{data_loss:.3f},'+f'reg_loss:{regularization_loss:.3f} ), ' + f'lr : {self.optimizer.current_learning_rate} ' )
				
		#Get and print epoch loss and accuracy
		epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization = True)
		epoch_loss = epoch_data_loss + epoch_regularization_loss
		epoch_accuracy = self.accuracy.calculate_accumulated()
		
		print(f'training,' + f'acc:{epoch_accuracy:.3f},' + f'loss:{epoch_loss:.3f}(' + f'data_loss:{epoch_data_loss:.3f},' + f'reg_loss:{epoch_regularization_loss:.3f}),' + f'lr:{self.optimizer.current_learning_rate}')
		
		#if there is the validation data	
		if validation_data is not None:
			#Evaluate the model
			self.evaluate(*validation_data,  batch_size = batch_size)
			
	#Evaluates the model using passed in dataset
	def evaluate(self, X_val, y_val, *, batch_size = None):
		#Default value if batch size is not being set
		validation_steps = 1
		#calculate number of steps
		if batch_size is not None:
			validation_steps = len(X_val) // batch_size
			#Dividing rounds down.If there are some remaining
			#data, but not a full batch, this won't include it
			#Add 1 to include this not full batch
			if validation_steps * batch_size < len(X_val):
				validation_steps += 1
	
		self.loss.new_pass()
		self.accuracy.new_pass()
			
		for step in range(validation_steps):
			if batch_size is None:
				batch_X = X_val
				batch_y = y_val
			else:
				batch_X = X_val[step * batch_size:(step + 1)*batch_size]
				batch_y = y_val[step*batch_size:(step + 1)*batch_size]
					
			output = self.forward(batch_X, training=False)
			self.loss.calculate(output, batch_y)
			predictions = self.output_layer_activation.predictions(output)
			self.accuracy.calculate(predictions, batch_y)
				
		validation_loss = self.loss.calculate_accumulated()
		validation_accuracy = self.accuracy.calculate_accumulated()
		print(f'validation,'+f'acc:{validation_accuracy:.3f},'+f'loss:{validation_loss:.3f}')
	
			######
	#Performs forward pass
	def forward(self, X, training):
		#Call forward method on input layer, this will set the output property that the first layer in "prev" object is expecting
		self.input_layer.forward (X, training)
		#Call forward method of every object in a chain, Pass output of previous object as parameter
		for layer in self.layers:
			layer.forward(layer.prev.output, training)
			#"layer" is now the last object from the list, return its output
		return layer.output
		
	#performs backward pass
	def backward(self, output, y):
		#if softmax classifier
		if self.softmax_classifier_output is not None:
			#First call backward method
			#on the combined activation/loss
			#this will set dinputs property
			self.softmax_classifier_output.backward(output, y)
			#since we'll  not call backward method of the last layer
			#which is Softmax activation
			#as we used combined activation/loss object
			# let's set dinputs in this object
			self.layers[-1].dinputs = self.softmax_classifier_output.dinputs
			#call backward method going through 
			#all the objects but first in reversed order passing dinputs as a parameter
			for layer in reversed(self.layers[:-1]):
				layer.backward(layer.next.dinputs)
			return
			#1st call backward method on the loss
			#This will set dinputs property that the last
			#layer will try to access 
		self.loss.backward(output, y)
		#call backward method going thru all objects
		#in reversed order passing dinputs as a parameter
		for layer in reversed(self.layers):
			layer.backward(layer.next.dinputs)
			
	#Retrieves and returns parameters of trainable layers
	def get_parameters(self):
		#create a list for parameters
		parameters = []
		#iterable trainable layers and get their parameters
		for layer in self.trainable_layers:
			parameters.append(layer.get_parameters())
		#Return a list
		return parameters
		
	#updates the model with new parameters
	def set_parameters(self, parameters):
		#iterate over the parameters and layers
		#and update each layer with each set of the parameters
		for parameter_set, layer in zip(parameters, self.trainable_layers):
			layer.set_parameters(*parameter_set)
	
	#saves the parameters to a file
	def save_parameters(self, path):
		#open a file in the binary write mode
		#and save parameters to it
		with open(path, 'wb') as f:
			pickle.dump(self.get_parameters(), f)
	#loads the weights and updates a model instance with them
	def load_parameters(self, path):
		#open file in the binary-read mode
		#load weights and update trainable layers
		with open(path, 'rb') as f:
			self.set_parameters(pickle.load(f))


#Loads a MNIST dataset
def load_mnist_dataset(dataset, path):
	#scan all directories and create list of labels
	labels = os.listdir(os.path.join(path, dataset))
	#create lists for samples and labels
	X = []
	y = []
	#for each label folder
	for label in labels:
		#And for each image in given folder
		for file in os.listdir(os.path.join(path, dataset, label)):
			#read the image
			image = cv2.imread(os.path.join(path, dataset, label, file), cv2.IMREAD_UNCHANGED)
			#And append it and a label to the lists
			X.append(image)
			y.append(label)
	#convert data to proper numpy arrays and return
	return np.array(X), np.array(y).astype('uint8')

#MNIST dataset (train + test)	
def create_data_mnist(path):
	#load both sets separately
	X, y = load_mnist_dataset('train', path)
	X_test, y_test = load_mnist_dataset('test', path)
	#And return all the data
	return X, y, X_test, y_test

#create dataset
X, y, X_test, y_test = create_data_mnist('fashion_mnist_images')		
#shuffle the Training dataset
keys = np.array(range(X.shape[0]))
np.random.shuffle(keys)
X = X[keys]
y = y[keys]
#scale and reshape samples
X = (X.reshape(X.shape[0], -1).astype(np.float32) - 127.5) / 127.5
X_test = (X_test.reshape(X_test.shape[0], -1).astype(np.float32) - 127.5) / 127.5

#Initiate the model
model = Model()

#Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128,128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
#set loss, optimizer and accuracy objects
model.set(loss = Loss_CategoricalCrossentropy(),
optimizer = Optimizer_Adam(decay = 1e-3 ),
accuracy = Accuracy_Categorical())
#finalize the model
model.finalize()
#train the model
model.train(X, y, validation_data = (X_test, y_test), epochs = 5, batch_size=128, print_every = 100)
#Retrieve model parameters
parameters = model.get_parameters()
#New model
#Instantiate the model
model = Model()

#Add layers
model.add(Layer_Dense(X.shape[1], 128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128,128))
model.add(Activation_ReLU())
model.add(Layer_Dense(128, 10))
model.add(Activation_Softmax())
#set loss and accuracy objects
#we do not set optimizer object this time
#as we won't train the model
model.set(loss = Loss_CategoricalCrossentropy(),
accuracy = Accuracy_Categorical())
#finalize the model
model.finalize()
#
model.load_parameters('fashion_mnist.parms')

#set model with parameters instead of training it
model.set_parameters(parameters)
#Evaluate the model
model.evaluate(X_test, y_test)
#to save parameters of a trained model
#model.save_parameters('fashion_mnist.parms')
