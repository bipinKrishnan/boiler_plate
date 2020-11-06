from tqdm.notebook import tqdm

class Trainer:
  def __init__(self, model, train_loader, optim, loss, val_loader=None, device='cpu', hide_pbar: bool=False):
    self.model = model.to(device)
    self.trainloader = train_loader
    self.valloader = val_loader
    self.optim = optim
    self.optim.param_groups[0]['params'] = self.model.parameters()
    self.criterion = loss
    self.device = device 
    self.hide_pbar = hide_pbar

  def training_loop(self):
    self.model.train()
    for data, label in tqdm(self.trainloader, total=len(self.trainloader), leave=False, disable=self.hide_pbar):
      data, label = data.to(self.device), label.to(self.device)
      self.optim.zero_grad()
      out = self.model(data)

      loss = self.criterion(out, label)
      loss.backward()
      self.optim.step()

    return loss

  def validation_loop(self):
    self.model.eval()
    for data, label in tqdm(self.valloader, total=len(self.valloader), leave=False, disable=self.hide_pbar):
      data, label = data.to(self.device), label.to(self.device)
      out = self.model(data)
      loss = self.criterion(out, label)

    return loss

  def check_dataloader(self, dataloader, num_batch):
    for i, (data, label) in enumerate(dataloader):
      print(f'Batch: {i+1}   data_size: {data.shape}   label_size: {label.shape}')
      if (i+1)==num_batch:
        break

  def check_loader(self, num_batches=3):
    print("Training set")
    self.check_dataloader(self.trainloader, num_batches)
    
    if self.valloader:
      print("\nValidation set")
      self.check_dataloader(self.valloader, num_batches)

  def fit(self, epochs: int):
    for epoch in tqdm(range(epochs), total=epochs, leave=False, disable=self.hide_pbar):
      train_loss = self.training_loop()

      if self.valloader:
        val_loss = self.validation_loop()
        print(f'epoch: {epoch+1}\ttrain_loss: {train_loss}\tval_loss: {val_loss}')
      else:
        print(f'epoch: {epoch+1}\ttrain_loss: {train_loss}')
