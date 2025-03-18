
import torch
from torch import nn

def trainloop(epoch, model, train_loader, criterion, optimizer):
  model.train()
  running_loss = 0.0
  correct = 0
  total = 0

  for batch_idx, (inputs, targets) in enumerate(train_loader):
    inputs, targets = inputs.cuda(), targets.cuda()
    print(batch_idx,end="\r")
    #compute prediction and loss
    pred = model(inputs)
    loss = criterion(pred,targets)

    #backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    running_loss += loss.item()
    value , predicted_class = pred.max(1)  #gets the predicted class for each example in the batch. It returns the value (the max) and the index (the predicted class)
    total += targets.size(0)
    correct += predicted_class.eq(targets).sum().item() #extract the value from tensor


  train_loss = running_loss / len(train_loader)
  train_accuracy = 100. * correct / total
  #print(f'Train Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
  return train_loss, train_accuracy

