
import os
import tempfile

import torch 
from pd.metric import amex_metric, sigmoid_focal_loss
from pd.params import *
from pd.pred import pred_test_npy as predict
from pd.utils import write_log


def train_torch_model(model, train_loader,  validation_data=None, output_model_name="", tempdir=None, config={"weight_decay":0.01},):
    optimizer = torch.optim.Adam(model.parameters(), weight_decay=config["weight_decay"])
    criterion = torch.nn.BCELoss()
    #criterion = sigmoid_focal_loss

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

    if tempdir is None:
        tempdir = tempfile.mkdtemp(prefix=f"train_torch_{output_model_name}_", dir=OUTDIR)

    for epoch in range(config["num_epochs"]): 
        for idx, (feat, clabel) in enumerate(train_loader):
            if len(feat.shape) == 4:  ## Reduce shape if its coming from a ratio version of the loader
                feat = feat.squeeze(dim=0)
                clabel = clabel.squeeze(dim=0)

            pred = model(feat.to(device))
            #weight = clabel.clone()
            #weight[weight==0] = 4
            #criterion.weight = weight
            loss = criterion(pred, clabel.to(device))
            
            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            model_metric, gini, recall = amex_metric(clabel.detach().numpy(), pred.detach().numpy(), return_components=True)
            val_metrix = 0
            val_gini, val_recall = 0, 0
            if model_metric > valThreshold:
                X_test, y_test = validation_data
                val_features = torch.as_tensor(X_test, dtype=torch.float32)
                val_pred = model(val_features)
                val_metrix, val_gini, val_recall = amex_metric(y_test, val_pred.detach().numpy(), return_components=True)

            log_message = f"{epoch}, BCE loss: {loss.item():.3f},train -> amex {model_metric:.3f}, gini {gini:.3f}, recall {recall:.3f}, val -> amex {val_metrix:.3f} gini {val_gini:.3f}, recall {val_recall:.3f}"
            print(log_message)
            write_log(log=log_message, log_desc=output_model_name+"_log", out_dir=tempdir)

            if val_metrix > PerfThreshold:
                output = output_model_name + f"_{int(1000*val_metrix)}_{epoch}_{idx}"
                output_file = os.path.join(tempdir, output)
                torch.save(model.state_dict(), output_file)
                predict(model=model, model_name=output)





    return model