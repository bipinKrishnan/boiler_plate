## boiler-plate

This library aims at replacing boilerplate code that occur while training neural networks using [PyTorch](https://pytorch.org/)

### Installing the library
      pip install boiler-plate-pytorch
      
### Usage
```python

     from boilerplate import Trainer
     
     ###Create the dataloader
     mnist = MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
     mnist_ = MNIST(root='./data', train=False, transform=transforms.ToTensor(), download=True)
     
     mnist_train = DataLoader(mnist, 12, shuffle=True)
     mnist_val = DataLoader(mnist_, 12)
     
     ###Build the model
     model = nn.Sequential(
        nn.Conv2d(1, 6, 3),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(4056, 10)
)

    ###Initialize the optimizer and loss function
    opt = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    ###Initialize the trainer by passing the model, optimizer, loss function, dataloaders and device(cpu or cuda)
    trainer = Trainer(model, 
                      opt, 
                      criterion,
                      mnist_train, 
                      mnist_val,  
                      device='cuda')
                      
    ###Check the dataloader
    trainer.check_data_loader()
    
    ###Run sanity check on N number(here 5) of batches
    trainer.sanity_check(5)

    ###Train the model
    trainer1.fit(2)
     
 ```
