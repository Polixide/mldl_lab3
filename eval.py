import torch
# Validation loop
def validate(model, val_loader, criterion):

  model.eval()
  val_loss = 0
  correct, total = 0, 0

  with torch.no_grad():
    for batch_idx, (inputs, targets) in enumerate(val_loader):
      inputs, targets = inputs.cuda(), targets.cuda()

      pred = model(inputs)
      test_loss = criterion(pred,targets)

      val_loss += test_loss.item()
      value , predicted_class = pred.max(1)
      total += targets.size(0)
      correct += predicted_class.eq(targets).sum().item()

  val_loss = val_loss / len(val_loader)
  val_accuracy = 100. * correct / total

  #print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
  return val_loss, val_accuracy