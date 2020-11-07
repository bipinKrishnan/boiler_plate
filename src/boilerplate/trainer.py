import torch
from tqdm.notebook import tqdm

class Trainer:
  def __init__(self, model, optim, loss, train_loader=None, val_loader=None, device='cpu', hide_pbar: bool=False):
    if not train_loader and not val_loader:
      print("Warning: No training and validation dataloader is provided to the Trainer()")

    self.model = model.to(device)
    self.trainloader = train_loader
    self.valloader = val_loader
    self.optim = optim
    self.optim.param_groups[0]['params'] = self.model.parameters()
    self.criterion = loss
    self.device = device 
    self.hide_pbar = hide_pbar

  def train_batch(self, data, label):
    data, label = data.to(self.device), label.to(self.device)
    self.optim.zero_grad()
    out = self.model(data)

    loss = self.criterion(out, label)
    loss.backward()
    self.optim.step()

    return out, loss

  def val_batch(self, data, label):
    data, label = data.to(self.device), label.to(self.device)
    out = self.model(data)
    loss = self.criterion(out, label)

    return out, loss

  def train_loop(self, dataloader=None):
    if not dataloader and not self.trainloader:
      raise TypeError("No training dataloader is provided")

    self.model.train()
    if dataloader:
      for i, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=self.hide_pbar):
        out, loss = self.train_batch(data, label)
    else: 
      for i, (data, label) in tqdm(enumerate(self.trainloader), total=len(self.trainloader), leave=False, disable=self.hide_pbar):
        out, loss = self.train_batch(data, label)

    return out, loss

  def val_loop(self, dataloader=None):
    if not dataloader and not self.valloader:
      raise TypeError("No validation dataloader is provided")

    if self.valloader:
      for i, (data, label) in tqdm(enumerate(self.valloader), total=len(self.valloader), leave=False, disable=self.hide_pbar):
        out, loss = self.val_batch(data, label)
    else:
      for i, (data, label) in tqdm(enumerate(dataloader), total=len(dataloader), leave=False, disable=self.hide_pbar):
        out, loss = self.val_batch(data, label)

    return out, loss

  def check_loaders(self, dataloader, num_batch):
    for i, (data, label) in enumerate(dataloader):
      print(f'Batch: {i+1}  data_size: {data.shape}  label_size: {label.shape}')
      if (i+1)==num_batch:
        break 

  def check_data_loader(self, n_batches=3):
    if not self.trainloader and not self.valloader:
      raise TypeError("No training and validation dataloader is provided to the Trainer()")

    print("Training set")
    self.check_loaders(self.trainloader, n_batches)

    if self.valloader:
      print("\nValidation set")
      self.check_loaders(self.valloader, n_batches)

  def fit(self, epochs: int):
    for epoch in tqdm(range(epochs), total=epochs, leave=False, disable=self.hide_pbar):
      train_loss = self.train_loop(self.trainloader)[1]

      if self.valloader:
        val_loss = self.val_loop(self.valloader)[1]
        print(f'epoch: {epoch+1}\ttrain_loss: {train_loss}\tval_loss: {val_loss}')
      else:
        print(f'epoch: {epoch+1}\ttrain_loss: {train_loss}')

  def predict(self, input_tensor):
    with torch.no_grad():
      pred = self.model(input_tensor)
      return pred

  def sanity_check(self, n_batch):
    if not self.trainloader and not self.valloader:
      raise TypeError("No training and validation dataloader is provided to the Trainer()")
      
    print("Training")
    for i, (data_t, label_t) in enumerate(self.trainloader):
      _, loss_t = self.train_batch(data_t, label_t)
      print(f"Batch: {i+1}  loss: {loss_t}")
      if (i+1)==n_batch:
        break

    if self.valloader:
      print("\nValidating")
      for j ,(data_v, label_v) in enumerate(self.valloader): 
        _, loss_v = self.val_batch(data_v, label_v)
        print(f"Batch: {j+1}  loss: {loss_v}")
        if (j+1)==n_batch:
          break
