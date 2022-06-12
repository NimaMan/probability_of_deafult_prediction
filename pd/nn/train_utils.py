
import torch 
from pd.metric import amex_metric


def train(model, loader, num_epochs=15):
    optimizer = torch.optim.Adam(model.parameters(),)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs): 
        for feat, clabel in loader:
            pred = model(feat)
            #weight = clabel.clone()
            #weight[weight==0] = 4
            #criterion.weight = weight
            loss = criterion(pred, clabel)
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f"{epoch}, BCE loss: {loss.item():.3f}, amex: {amex_metric(clabel.detach().numpy(), pred.detach().numpy()):.3f}")

    return model