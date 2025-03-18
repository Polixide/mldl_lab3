from tqdm.notebook import tqdm
from data.dataloader import TinyImageNetDataLoader
from eval import validate
from models.custom_model import CustomNet
import torch
from torch import nn

def train(epoch, model, train_loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader)):
        inputs, targets = inputs.cuda(), targets.cuda()
        
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
    print(f'Train Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')

def main():
  data_loader = TinyImageNetDataLoader(data_dir='dataset/tiny_imagenet/tiny-imagenet-200', batch_size=64)
  train_loader , val_loader = data_loader.get_dataloaders()
  model = CustomNet().cuda()
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  best_acc = 0

  # Run the training process for {num_epochs} epochs
  num_epochs = 10
  for epoch in range(1, num_epochs + 1):
      print("=================================")
      print("Epoch: ",epoch)


      train(epoch, model, train_loader, criterion, optimizer)
      # At the end of each training iteration, perform a validation step
      val_accuracy = validate(model, val_loader, criterion)

      # Best validation accuracy
      best_acc = max(best_acc, val_accuracy)
      

  print(f'Best validation accuracy: {best_acc:.2f}%')


if __name__ == "__main__":
  main()
