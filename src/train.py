import torch
import argparse
import torchaudio
import os
import numpy as np

from tensorboardX import SummaryWriter

from models import *
from Datasets import DatasetSSL,Audio_Collate

from utils.hparams import HParam
from utils.writer import MyWriter

from common import run
#from common import run, get_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str, required=True,
                        help="yaml for configuration")
    parser.add_argument('--default', type=str, default=None,
                        help="default configuration")
    parser.add_argument('--version_name', '-v', type=str, required=True,
                        help="version of current training")
    parser.add_argument('--chkpt',type=str,required=False,default=None)
    parser.add_argument('--step','-s',type=int,required=False,default=0)
    parser.add_argument('--device','-d',type=str,required=False,default="cuda:0")
    args = parser.parse_args()

    hp = HParam(args.config,args.default)
    print("NOTE::Loading configuration : "+args.config)

    device = args.device
    version = args.version_name
    torch.cuda.set_device(device)

    batch_size = hp.train.batch_size
    num_epochs = hp.train.epoch
    num_workers = hp.train.num_workers

    best_loss = 1e7
    best_acc = 0

    ## load
    modelsave_path = hp.log.root +'/'+'chkpt' + '/' + version
    log_dir = hp.log.root+'/'+'log'+'/'+version

    os.makedirs(modelsave_path,exist_ok=True)
    os.makedirs(log_dir,exist_ok=True)

    writer = MyWriter(hp, log_dir)

    train_dataset = DatasetSSL(hp,is_train=True)
    test_dataset= DatasetSSL(hp,is_train=False)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers, collate_fn = lambda x : Audio_Collate(x))
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False,num_workers=num_workers,collate_fn = lambda x : Audio_Collate(x))


    ## Criterion
    if hp.loss.type == "CrossEntropyLoss":
        criterion = torch.nn.CrossEntropyLoss()
        last_activation = "Softmax"
    elif hp.loss.type == "BCELoss":
        criterion = torch.nn.BCELoss()
        last_activation = "Sigmoid"
    else :
        raise Exception("ERROR::Unsupported criterion : {}".format(hp.loss.type))

    ## Model
    if hp.model.type == "m1" : 
        model = CRNN(4,
        pool_type = hp.model.m1.pool_type,
        last_activation = last_activation
        ).to(device)
    elif hp.model.type =="m2":
        model = TCRN(4,
        last_activation = last_activation
        ).to(device)
    elif hp.model.type =="TCRNv2":
        model = TCRNv2(4,
        last_activation = last_activation
        ).to(device)
    elif hp.model.type == "CRNNv2":
        model = CRNNv2(4,
        pool_type = hp.model.CRNNv2.pool_type,
        last_activation = last_activation
        ).to(device)
    else :
        raise Exception("ERROR::Unknown model type : {}".format(hp.model.type))
    # or model = get_model(hp).to(device)

    if not args.chkpt == None : 
        print('NOTE::Loading pre-trained model : '+ args.chkpt)
        model.load_state_dict(torch.load(args.chkpt, map_location=device))

    optimizer = torch.optim.Adam(model.parameters(), lr=hp.train.adam)

    if hp.scheduler.type == 'Plateau': 
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
            mode=hp.scheduler.Plateau.mode,
            factor=hp.scheduler.Plateau.factor,
            patience=hp.scheduler.Plateau.patience,
            min_lr=hp.scheduler.Plateau.min_lr)
    elif hp.scheduler.type == 'oneCycle':
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                max_lr = hp.scheduler.oneCycle.max_lr,
                epochs=hp.train.epoch,
                steps_per_epoch = len(train_loader)
                )
    elif hp.scheduler.type == "CosineAnnealingLR" : 
       scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=hp.scheduler.CosineAnnealingLR.T_max, eta_min=hp.scheduler.CosineAnnealingLR.eta_min)
    else :
        raise Exception("ERROR::Unsupported sceduler type : {}".format(hp.scheduler.type))

    step = args.step

    for epoch in range(num_epochs):
        ### TRAIN ####
        model.train()
        train_loss=0
        for i, (data,label) in enumerate(train_loader):
            data = data.to(device)
            label = label.to(device)
            step +=1

            loss = run(data,label,model,criterion)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
           

            if step %  hp.train.summary_interval == 0:
                print('TRAIN::{} : Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(version,epoch+1, num_epochs, i+1, len(train_loader), loss.item()))
                writer.log_value(loss,step,'train loss : '+hp.loss.type)

        train_loss = train_loss/len(train_loader)
        torch.save(model.state_dict(), str(modelsave_path)+'/lastmodel.pt')
            
        #### EVAL ####
        model.eval()
        with torch.no_grad():
            test_loss =0.

            n_test = 0
            n_correct = 0
            for j, (data,label) in enumerate(test_loader):
                data = data.to(device)
                label = label.to(device)
                output,loss = run(data,label,model,criterion,ret_output=True)
                test_loss += loss.item()

                n_test += data.shape[0]

                _label  = label.nonzero(as_tuple=True)[1]
                _output = output.max(axis=1)[1]
                n_correct += torch.sum(_label==_output).detach().cpu().numpy()


                test_loss +=loss.item()

            test_loss = test_loss/len(test_loader)
            acc = n_correct/n_test
            print('TEST::{} :  Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Acc : {:.4f}'.format(version, epoch+1, num_epochs, j+1, len(test_loader), test_loss,acc))


            scheduler.step(test_loss)
            
            writer.log_value(test_loss,step,'test lost : ' + hp.loss.type)
            writer.log_value(acc,step,'acc')


            if best_loss > test_loss:
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel.pt')
                best_loss = test_loss

            if best_acc < acc and acc > 0.96:
                print("SAVE::model_{}.pt prev acc {}, new acc {}".format(acc,best_acc,acc))
                torch.save(model.state_dict(), str(modelsave_path)+'/bestmodel_{}.pt'.format(acc))
                best_acc = acc


