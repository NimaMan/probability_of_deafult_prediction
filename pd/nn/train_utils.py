
import torch 
from pd.metric import amex_metric
from pd.params import *
from pd.pred import pred_test_npy as predict


def train(model, loader, num_epochs=15, output_model_name=""):
    optimizer = torch.optim.Adam(model.parameters(),)
    criterion = torch.nn.BCELoss()

    for epoch in range(num_epochs): 
        for idx, (feat, clabel) in enumerate(loader):
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
            if model_metric > PerfThreshold:
                output = output_model_name + f"_{int(1000*model_metric)}_{epoch}_{idx}"
                torch.save(model.state_dict(), OUTDIR+output)
                predict(model=model, model_name=output_model_name)





    return model