import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
	def __init__(self, n_inputs, n_neurons, weight_regularizer_L1=0, weight_regularizer_L2=0, bias_regularizer_L1=0, bias_regularizer_L2=0):
		self.weights = 0.01 * np.random.randn(n_inputs, n_neurons)
		self.biases = np.zeros((1, n_neurons))
		
		#set regularization strength
		self.weight_regularizer_L1 = weight_regularizer_L1
		self.weight_regularizer_L2 = weight_regularizer_L2
		self.bias_regularizer_L1 = bias_regularizer_L1
		self.bias_regularizer_L2 = bias_regularizer_L2
		
	def forward(self, inputs):
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
		 
#Dropout
class Layer_Dropout:
	#init
	def __init__(self, rate):
		 #store rate, we invert it as for example for dropout of  0.1 we need success rate of 0.9
		self.rate = 1 - rate
	#forward pass
	def forward(self, inputs):
		#save input values
		self.inputs = inputs
		#generate n save scaled mask
		self.binary_mask = np.random.binomial(1, self.rate, size = inputs.shape) / self.rate
		#apply mask to output values
		self.output = inputs * self.binary_mask
	#Backward pass
	def backward(self, dvalues):
		#gradient on values
		self.dinputs = dvalues * self.binary_mask
		 
		 
class Activation_ReLU:
	def forward(self, inputs):
		 self.inputs = inputs
		 self.output = np.maximum(0, inputs)
		 
	def backward(self, dvalues):
		 self.dinputs = dvalues.copy()
		 self.dinputs[self.inputs<=0]=0
		 
class Activation_Softmax:
	def forward(self, inputs):
		 exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
		 probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
		 self.output = probabilities
		 
	def backward(self, dvalues):
		 self.dinputs = np.empty_like(dvalues)
		 
		 for index, (single_output, single_dvalues) in enumeration(self.output, dvalues):
		 	single_output = single_output.reshape(-1, 1)
		 	jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
		 	self.dinputs[index] = np.dot(jacobian_matrix, single_dvalues)
		 	
class Loss:
	def regularization_loss(self, layer):
		# 0 by default
		regularization_loss = 0
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
			
	def calculate(self, output, y):
		 sample_losses = self.forward(output, y)
		 data_loss = np.mean(sample_losses)
		 return data_loss
		 
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
		
class Activation_Softmax_Loss_CategoricalCrossentropy():
	def __init__(self):
		self.activation = Activation_Softmax()
		self.loss = Loss_CategoricalCrossentropy()
		
	def forward(self, inputs, y_true):
		self.activation.forward(inputs)
		self.output = self.activation.output
		return self.loss.calculate(self.output, y_true)
		
	def backward(self, dvalues, y_true):
		samples = len(dvalues)
		
		if len(y_true.shape) == 2:
			y_true = np.argmax(y_true, axis=1)
		
		self.dinputs = dvalues.copy()
		self.dinputs[range(samples), y_true] -=1
		self.dinputs = self.dinputs / samples
		
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
		
		 
X,y = spiral_data(samples=1000, classes=3)

#regularization terms r added to hidden layer
dense1 = Layer_Dense(2, 512,weight_regularizer_L2=5e-4, bias_regularizer_L2=5e-4)
activation1 = Activation_ReLU()
#create dropout layer
dropout1 = Layer_Dropout(0.1)
#create 2nd Dense layer with 64 input features(as we take output of previous layer) and 3 output values.
dense2 = Layer_Dense(512, 3)
loss_activation = Activation_Softmax_Loss_CategoricalCrossentropy()
#optimizer = Optimizer_SGD(decay = 1e-3, momentum = 0.9)
#optimizer = Optimizer_Adagrad(decay=1e-5)
#optimizer = Optimizer_RMSprop(learning_rate=0.02,decay = 1e-5, rho=0.999)
optimizer = Optimizer_Adam(learning_rate = 0.05, decay=5e-5)

for epoch in range(10001):
	dense1.forward(X)
	activation1.forward(dense1.output)
	#perform 4ward pass thru Dropout layer
	dropout1.forward(activation1.output)
	
	dense2.forward(dropout1.output)
	#calculate loss from output of dense2
	data_loss = loss_activation.forward(dense2.output, y)
	#calculate regularization penalty
	regularization_loss = loss_activation.loss.regularization_loss(dense1) + loss_activation.loss.regularization_loss(dense2)
	#calculate overall loss
	Loss = data_loss + regularization_loss
	

#print(loss_activation.output[:5])
#print('Loss:',Loss)

	predictions = np.argmax(loss_activation.output, axis=1)
	if len(y.shape) == 2:
		y = np.argmax(y, axis=1)
 
	accuracy = np.mean(predictions == y)
	#print('Accuracy:', accuracy)
	
	if not epoch % 100:
		print(f'epoch : {epoch}, ' + f'acc : {accuracy: .3f}, ' + f'loss : {Loss: .3f}('+f'data_loss :{data_loss:.3f},'+f'reg_loss:{regularization_loss:.3f} ), ' + f'lr : {optimizer.current_learning_rate} ' )
 
#Backward
	loss_activation.backward(loss_activation.output, y)
	dense2.backward(loss_activation.dinputs)
	dropout1.backward(dense2.dinputs)
	activation1.backward(dropout1.dinputs)
	dense1.backward(activation1.dinputs)

#print(dense1.dweights)
#print(dense1.dbiases)
#print(dense2.dweights)
#print(dense2.dbiases)


#update weights n biases

	optimizer.pre_update_params()
	optimizer.update_params(dense1)
	optimizer.update_params(dense2)
	optimizer.post_update_params()

#Create test dataset
X_test, y_test = spiral_data(samples=100, classes=3)

#validate the model
#perform a forward pass
dense1.forward(X_test)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
loss = loss_activation.forward(dense2.output, y_test)

predictions = np.argmax(loss_activation.output, axis=1)
if len(y_test.shape) == 2:
	y_test = np.argmax(y_test, axis=1)
accuracy = np.mean(predictions==y_test)
print(f'Validation, acc:{accuracy:.3f},loss:{loss: .3f}')
