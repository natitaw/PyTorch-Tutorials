# 1. Data [1, 8, 9]

- The ETL Process
    - Extract: get the data from the source
    - Transform: transform the data into a tensor form
    - Load: put the data onto an object

### Custom DataLoader Class QData -- train_set [1]

- DataLoader: a PyTorch calss that can be used to make an object out of Qmicro data
- DataSet QData class that works with the DataLoader
    - Similar class as FasionMNIST
    - Represented as a train_loader with batch_size
    
    
# 2. Model [1, 10]

- The model needs to be defined either by a PyTorch class or by a sequential method
- The preference here is to use the PyTorch OOP method 
    - Easier to keep track of and clear to see
    
### Model Parapeters

- The model building process requires the following types of perameters:
    1. [Hyperparameters]: output_channels, kernel_size, out_channel, etc.
        - Arbitrarily chosen
    2. [Data Dependent Parameters]: in_channels, in_features
        - Learnt from data
        
### Implement Network Architecture and **forward()** Method

- The Network requires a certain type of architecture
- Forward method

# 3. Training Loop [1, 7]

- The training process
    1. Acquire batch from a training set
    2. Pass batch to the network
    3. Calculate the loss 
    4. Calculate the gradient of the loss function with respect to the weights of the network
    5. Update the weights using the gradients to reduce the loss
    6. Repeat steps 1-5 for the epoch
        - Epoch represents a time period in which an entire training set has been covered
    7. Repeat steps 1-6 for the given number of epochs to reach desired accuracy

# 4. Analysing a Trained Network [1, 6]

- Accuracy function
- Remember torch.no_grad()
- Confusion Matrix
