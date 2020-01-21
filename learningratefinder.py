from keras.callbacks import LambdaCallback
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import tempfile

class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        # Store the model, stop factor and beta value (for computing a smoothed, average loss)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        
        # Initialize our list of learning rates and losses respectively
        self.lrs = []
        self.losses = []
        
        # Initialize our learning rate multiplier, average loss, best loss found thus far, current batch number, and weights file
        self.lrMult = 1e9 # Learning rate multiplication factor
        self.avgLoss = 0 # Average loss over time
        self.bestLoss = 1e9 # The best loss we have found when training
        self.batchNum = 0 # The current batch number update
        self.weightsFile = None # Our path to the initial model weights
        
    def reset(self):
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None
            
    def is_data_iter(self, data):
        # Define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator", "DataFrameIterator", "Iterator", "Sequence"]
        # Return whether our data is an iterator
        return data.__class__.__name__ in iterClasses
        
    def on_batch_end(self, batch, logs):
        # Grab the current learning rate and log it to the list of learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        # Grab the loss at the end of this batch,
        # increment the total number of batches processed,
        # smooth it, and update the losses list with the smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)
        
        # Compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss
        
        # Check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # Stop returning and return from the method
            self.model.stop_training = True
            return
        
        # Check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth
            
        # Increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)
        
    def find(self, trainData, startLR, endLR, epochs=None, stepsPerEpoch=None, batchSize=32, sampleSize=2048, verbose=1):
        # Reset our class specific variables
        self.reset()
        
        # Determine if we are using a data generator or not
        useGen = self.is_data_iter(trainData)
        
        # If we're using a generator and steps per epoch is not supplied, raise an error
        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)
            
        # If we're not using a generator then our entire dataset must already be in the memory
        elif not useGen:
            # Grab the number of samples in the training data and then derive the number of steps per epoch
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))
            
        # If no number of training epochs are supplied,
        # compute the training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(batchSize)))
            
        # Compute the total number of batch updates that will take place
        # while we are attempting to find a good starting learning rate
        numBatchUpdates = epochs * stepsPerEpoch
        
        # Derive the learning rate multiplier based on ending learning rate, starting learning rate, and total number of batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
        
        # Create a temporary file path for the model weights and then save the weights
        # (so we can reset the weights when we are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)
        
        # Grab the *original* learning rate (so we can reset it later), and then set the *starting* learning rate
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)
        
        # Construct a callback that will be called at the end of each batch,
        # enabling us to increase our learning rate as training progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs: self.on_batch_end(batch, logs))
        
        # Check to see if we are using a data iterator
        if useGen:
            self.model.fit_generator(
                trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback])
        
        # Otherwise, our entire training data is already in memory
        else:
            # Train our model using Keras' fit method
            self.model.fit(
                trainData[0], trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                verbose=verbose,
                callbacks=[callback])
            
        # Restore the original model weights and learning rate
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, origLR)
            
    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # Grab the learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]
        
        # Plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        
        # If the title is not empty, add it to the plot
        if title != "":
            plt.title(title)