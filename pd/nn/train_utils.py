
import torch 
from pd.metric import amex_metric
from pd.params import *


def train(model, loader, num_epochs=15, output_model_name=""):
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
            model_metric = amex_metric(clabel.detach().numpy(), pred.detach().numpy())
            print(f"{epoch}, BCE loss: {loss.item():.3f}, amex: {model_metric:.3f}")
            if model_metric > 0.79:
                output_model_name = output_model_name + f"{epoch}"
                torch.save(model.state_dict(), OUTDIR+output_model_name)




    return model