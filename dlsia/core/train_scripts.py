import numpy as np
import torch
import torch.nn.utils
from dlsia.core import corcoef
from torchmetrics import F1Score
import pandas as pd
import logging

def segmentation_metrics(preds, target, missing_label=-1, are_probs=True, num_classes=None):
    """
    Computes a variety of F1 scores.
    See : https://towardsdatascience.com/micro-macro-weighted-averages-of-f1-score-clearly-explained-b603420b292f

    Parameters
    ----------
    preds : Predicted labels
    target : Target labels
    missing_label : missing label
    are_probs: preds are probabilities (pre or post softmax)
    num_classes: if preds are not probabilities, we need to know the number of classes.

    Returns
    -------
    micro and macro F1 scores
    """

    if are_probs:
        num_classes = preds.shape[1]
        tmp = torch.argmax(preds, dim=1)
    else:
        tmp = preds

    assert num_classes is not None

    F1_eval_macro = F1Score(task='multiclass',
                            num_classes=num_classes,
                            average='macro')
    F1_eval_micro = F1Score(task='multiclass',
                            num_classes=num_classes,
                            average='micro')
    a = tmp.cpu()
    b = target.cpu()

    sel = b == missing_label
    a = a[~sel]
    b = b[~sel]
    tmp_macro = torch.Tensor([0])
    tmp_micro = torch.Tensor([0])
    if len(a.flatten()) > 0:
        tmp_macro = F1_eval_macro(a, b)
        tmp_micro = F1_eval_micro(a, b)

    return tmp_micro, tmp_macro


def train_segmentation(net, trainloader, validationloader, NUM_EPOCHS,
                       criterion, optimizer, device,
                       savepath=None, saveevery=None,
                       scheduler=None, show=0,
                       use_amp=False, clip_value=None):
    """
    Loop through epochs passing images to be segmented on a pixel-by-pixel
    basis.

    :param net: input network
    :param trainloader: data loader with training data
    :param validationloader: data loader with validation data
    :param NUM_EPOCHS: number of epochs
    :param criterion: target function
    :param optimizer: optimization engine
    :param device: the device where we calculate things
    :param savepath: filepath in which we save networks intermittently
    :param saveevery: integer n for saving network every n epochs
    :param scheduler: an optional schedular. can be None
    :param show: print stats every n-th epoch
    :param use_amp: use pytorch automatic mixed precision
    :param clip_value: value for gradient clipping. Can be None.
    :return: A network and run summary stats
    """

    train_loss = []
    F1_train_trace_micro = []
    F1_train_trace_macro = []

    # Skip validation steps if False or None loaded
    if validationloader is False:
        validationloader = None
    if validationloader is not None:
        validation_loss = []
        F1_validation_trace_micro = []
        F1_validation_trace_macro = []

    best_score = 1e10
    best_index = 0
    best_state_dict = None

    if savepath is not None:
        if saveevery is None:
            saveevery = 1

    for epoch in range(NUM_EPOCHS):
        running_train_loss = 0.0
        running_F1_train_micro = 0.0
        running_F1_train_macro = 0.0
        tot_train = 0.0

        if validationloader is not None:
            running_validation_loss = 0.0
            running_F1_validation_micro = 0.0
            running_F1_validation_macro = 0.0
            tot_val = 0.0
        count = 0

        for data in trainloader:
            count += 1
            noisy, target = data  # load noisy and target images
            N_train = noisy.shape[0]
            tot_train += N_train

            noisy = noisy.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)
            noisy = noisy.to(device)
            target = target.to(device)

            if criterion.__class__.__name__ == 'CrossEntropyLoss':
                target = target.type(torch.LongTensor)
                target = target.to(device).squeeze(1)

            if use_amp is False:
                # forward pass, compute loss and accuracy
                output = net(noisy)
                loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
            else:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    # forward pass, compute loss and accuracy
                    output = net(noisy)
                    loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # update the parameters
                scaler.step(optimizer)
                scaler.update()

            # update the parameters
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            optimizer.step()


            tmp_micro, tmp_macro = segmentation_metrics(output, target)

            running_F1_train_micro += tmp_micro.item()
            running_F1_train_macro += tmp_macro.item()
            running_train_loss += loss.item()
        if scheduler is not None:
            scheduler.step()

        # compute validation step
        if validationloader is not None:
            with torch.no_grad():
                for x, y in validationloader:
                    x = x.to(device)
                    y = y.to(device)
                    N_val = y.shape[0]
                    tot_val += N_val
                    if criterion.__class__.__name__ == 'CrossEntropyLoss':
                        y = y.type(torch.LongTensor)
                        y = y.to(device).squeeze(1)

                    # forward pass, compute validation loss and accuracy
                    if use_amp is False:
                        yhat = net(x)
                        val_loss = criterion(yhat, y)
                    else:
                        with torch.cuda.amp.autocast():
                            yhat = net(x)
                            val_loss = criterion(yhat, y)

                    tmp_micro, tmp_macro = segmentation_metrics(yhat, y)
                    running_F1_validation_micro += tmp_micro.item()
                    running_F1_validation_macro += tmp_macro.item()

                    # update running validation loss and accuracy
                    running_validation_loss += val_loss.item()

        loss = running_train_loss / len(trainloader)
        F1_micro = running_F1_train_micro / len(trainloader)
        F1_macro = running_F1_train_macro / len(trainloader)
        train_loss.append(loss)
        F1_train_trace_micro.append(F1_micro)
        F1_train_trace_macro.append(F1_macro)

        if validationloader is not None:
            val_loss = running_validation_loss / len(validationloader)
            F1_val_micro = running_F1_validation_micro / len(validationloader)
            F1_val_macro = running_F1_validation_macro / len(validationloader)
            validation_loss.append(val_loss)
            F1_validation_trace_micro.append(F1_val_micro)
            F1_validation_trace_macro.append(F1_val_macro)

        if show != 0:
            learning_rates = []
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
            mean_learning_rate = np.mean(np.array(learning_rates))
            if np.mod(epoch + 1, show) == 0:
                if validationloader is not None:
                    print(
                        f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    print(
                        f'   Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}')
                    print(
                        f'   Micro Training F1: {F1_micro:.4f} | Micro Validation F1: {F1_val_micro:.4f}')
                    print(
                        f'   Macro Training F1: {F1_macro:.4f} | Macro Validation F1: {F1_val_macro:.4f}')
                else:
                    print(
                        f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')
                    print(
                        f'   Training Loss: {loss:.4e} | Micro Training F1: {F1_micro:.4f} | Macro Training F1: {F1_macro:.4f}')

        if validationloader is not None:
            if val_loss < best_score:
                best_state_dict = net.state_dict()
                best_index = epoch
                best_score = val_loss
        else:
            if loss < best_score:
                best_state_dict = net.state_dict()
                best_index = epoch
                best_score = loss

            if savepath is not None:
                torch.save(best_state_dict, savepath + '/net_best')
                print('   Best network found and saved')
                print('')

        if savepath is not None:
            if np.mod(epoch + 1, saveevery) == 0:
                torch.save(net.state_dict(), savepath + '/net_checkpoint')
                print('   Network intermittently saved')
                print('')

    if validationloader is None:
        validation_loss = None
        F1_validation_trace_micro = None
        F1_validation_trace_macro = None

    results = {"Training loss": train_loss,
               "Validation loss": validation_loss,
               "F1 training micro": F1_train_trace_micro,
               "F1 training macro": F1_train_trace_macro,
               "F1 validation micro": F1_validation_trace_micro,
               "F1 validation macro": F1_validation_trace_macro,
               "Best model index": best_index}

    net.load_state_dict(best_state_dict)
    return net, results

## 20240320, Trainer() added by xchong ##
class Trainer():
    def __init__(self, net, trainloader, validationloader, NUM_EPOCHS,
                       criterion, optimizer, device, dvclive=None,
                       savepath=None, saveevery=None,
                       scheduler=None, show=0,
                       use_amp=False, clip_value=None):


        """
        Loop through epochs passing images to be segmented on a pixel-by-pixel
        basis.

        :param net: input network
        :param trainloader: data loader with training data
        :param validationloader: data loader with validation data
        :param NUM_EPOCHS: number of epochs
        :param criterion: target function
        :param optimizer: optimization engine
        :param device: the device where we calculate things
        :param dvclive: use dvclive object to save metrics
        :param savepath: filepath in which we save networks intermittently
        :param saveevery: integer n for saving network every n epochs
        :param scheduler: an optional schedular. can be None
        :param show: print stats every n-th epoch
        :param use_amp: use pytorch automatic mixed precision
        :param clip_value: value for gradient clipping. Can be None.
        :return: A network and run summary stats
        """

        self.net = net
        self.trainloader = trainloader
        self.validationloader = validationloader
        self.NUM_EPOCHS = NUM_EPOCHS
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.dvclive = dvclive
        self.savepath = savepath
        self.saveevery = saveevery
        self.scheduler = scheduler
        self.show = show
        self.use_amp = use_amp
        self.clip_value = clip_value

        self.train_loss = []
        self.F1_train_trace_micro = []
        self.F1_train_trace_macro = []

          

        # Skip validation steps if False or None loaded
        if self.validationloader is False:
            self.validationloader = None
        if self.validationloader is not None:
            self.validation_loss = []
            self.F1_validation_trace_micro = []
            self.F1_validation_trace_macro = []

        self.best_score = 1e10
        self.best_index = 0
        self.best_state_dict = None

        if self.savepath is not None:
            if self.saveevery is None:
                self.saveevery = 1

        self.losses = pd.DataFrame()

    # Save Loss
    def save_loss(self,
            epoch,
            loss,
            F1_micro,
            F1_macro,
            val_loss=None,
            F1_val_micro=None,
            F1_val_macro=None,
            ):
        if self.validationloader is not None:
            table = pd.DataFrame(
                {
                    'epoch': [epoch],
                    'loss': [loss], 
                    'val_loss': [val_loss], 
                    'F1_micro': [F1_micro], 
                    'F1_macro': [F1_macro],
                    'F1_val_micro': [F1_val_micro],
                    'F1_val_macro': [F1_val_macro]
                }
            )
        
        else:
            table = pd.DataFrame(
                {
                    'epoch': [epoch],
                    'loss': [loss], 
                    'F1_micro': [F1_micro], 
                    'F1_macro': [F1_macro]
                }
            )

        return table
    
    # Save metrics by dvclive
    def save_dvc(self):
        if self.dvclive is not None:
            self.dvclive.log_metric("train/loss", self.train_loss[-1])
            self.dvclive.log_metric("train/F1_micro", self.F1_train_trace_micro[-1])
            self.dvclive.log_metric("train/F1_macro", self.F1_train_trace_macro[-1])
            if self.validationloader is not None:
                self.dvclive.log_metric("val/loss", self.validation_loss[-1])
                self.dvclive.log_metric("val/F1_micro", self.F1_validation_trace_micro[-1])
                self.dvclive.log_metric("val/F1_macro", self.F1_validation_trace_macro[-1])
            self.dvclive.next_step()           
        return True
    
    def train_one_epoch(self, epoch):
        print(
                f"*****  memory allocated at epoch {epoch} is {torch.cuda.memory_allocated(0)}"
        )
        running_train_loss = 0.0
        running_F1_train_micro = 0.0
        running_F1_train_macro = 0.0
        tot_train = 0.0

        if self.validationloader is not None:
            running_validation_loss = 0.0
            running_F1_validation_micro = 0.0
            running_F1_validation_macro = 0.0
            tot_val = 0.0
        count = 0

        for data in self.trainloader:
            count += 1
            noisy, target = data  # load noisy and target images
            N_train = noisy.shape[0]
            tot_train += N_train

            noisy = noisy.type(torch.FloatTensor)
            target = target.type(torch.LongTensor)
            noisy = noisy.to(self.device)
            target = target.to(self.device)

            if self.criterion.__class__.__name__ == 'CrossEntropyLoss':
                target = target.type(torch.LongTensor)
                target = target.to(self.device).squeeze(1)

            if self.use_amp is False:
                # forward pass, compute loss and accuracy
                output = self.net(noisy)
                loss = self.criterion(output, target)

                # backpropagation
                self.optimizer.zero_grad()
                loss.backward()
            else:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    # forward pass, compute loss and accuracy
                    output = self.net(noisy)
                    loss = self.criterion(output, target)

                # backpropagation
                self.optimizer.zero_grad()
                scaler.scale(loss).backward()

                # update the parameters
                scaler.step(self.optimizer)
                scaler.update()

            # update the parameters
            if self.clip_value is not None:
                torch.nn.utils.clip_grad_value_(self.net.parameters(), self.clip_value)
            self.optimizer.step()


            tmp_micro, tmp_macro = segmentation_metrics(output, target)

            running_F1_train_micro += tmp_micro.item()
            running_F1_train_macro += tmp_macro.item()
            running_train_loss += loss.item()
        if self.scheduler is not None:
            self.scheduler.step()

        # compute validation step
        if self.validationloader is not None:
            with torch.no_grad():
                for x, y in self.validationloader:
                    x = x.type(torch.FloatTensor)
                    y = y.type(torch.LongTensor)
                    x = x.to(self.device)
                    y = y.to(self.device)
                    N_val = y.shape[0]
                    tot_val += N_val
                    if self.criterion.__class__.__name__ == 'CrossEntropyLoss':
                        y = y.type(torch.LongTensor)
                        y = y.to(self.device).squeeze(1)

                    # forward pass, compute validation loss and accuracy
                    if self.use_amp is False:
                        yhat = self.net(x)
                        val_loss = self.criterion(yhat, y)
                    else:
                        with torch.cuda.amp.autocast():
                            yhat = self.net(x)
                            val_loss = self.criterion(yhat, y)

                    tmp_micro, tmp_macro = segmentation_metrics(yhat, y)
                    running_F1_validation_micro += tmp_micro.item()
                    running_F1_validation_macro += tmp_macro.item()

                    # update running validation loss and accuracy
                    running_validation_loss += val_loss.item()

        loss = running_train_loss / len(self.trainloader)
        F1_micro = running_F1_train_micro / len(self.trainloader)
        F1_macro = running_F1_train_macro / len(self.trainloader)
        self.train_loss.append(loss)
        self.F1_train_trace_micro.append(F1_micro)
        self.F1_train_trace_macro.append(F1_macro)

        val_loss = None
        if self.validationloader is not None:
            val_loss = running_validation_loss / len(self.validationloader)
            F1_val_micro = running_F1_validation_micro / len(self.validationloader)
            F1_val_macro = running_F1_validation_macro / len(self.validationloader)
            self.validation_loss.append(val_loss)
            self.F1_validation_trace_micro.append(F1_val_micro)
            self.F1_validation_trace_macro.append(F1_val_macro)

        print(f'Epoch: {epoch}')

        # Note: This is a very temporary solution to address the single frame mask case.
        if self.validationloader is None:
            F1_val_micro = None
            F1_val_macro = None

        table = self.save_loss(
                epoch,
                loss,
                F1_micro,
                F1_macro,
                val_loss=val_loss,
                F1_val_micro=F1_val_micro,
                F1_val_macro=F1_val_macro,
                )
        
        self.losses = pd.concat([self.losses, table])
       

        if self.show != 0:
            learning_rates = []
            for param_group in self.optimizer.param_groups:
                learning_rates.append(param_group["lr"])
            mean_learning_rate = np.mean(np.array(learning_rates))
            if np.mod(epoch + 1, self.show) == 0:
                if self.validationloader is not None:
                    logging.info(
                        f"Epoch {epoch + 1} of {self.NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}"
                    )
                    logging.info(
                        f"   Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}"
                    )
                    logging.info(
                        f"   Micro Training F1: {F1_micro:.4f} | Micro Validation F1: {F1_val_micro:.4f}"
                    )
                    logging.info(
                        f"   Macro Training F1: {F1_macro:.4f} | Macro Validation F1: {F1_val_macro:.4f}"
                    )
                else:
                    logging.info(
                        f"Epoch {epoch + 1} of {self.NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}"
                    )
                    logging.info(
                        f"   Training Loss: {loss:.4e} | Micro Training F1: {F1_micro:.4f} "
                        + "| Macro Training F1: {F1_macro:.4f}"
                    )

        if epoch == 0:
            self.best_state_dict = self.net.state_dict()
            if self.validationloader is not None:
                self.best_score = val_loss
            else:
                self.best_score = loss

        if self.validationloader is not None:
            if val_loss < self.best_score:
                self.best_state_dict = self.net.state_dict()
                self.best_index = epoch
                self.best_score = val_loss
        else:
            if loss < self.best_score:
                self.best_state_dict = self.net.state_dict()
                self.best_index = epoch
                self.best_score = loss

            if self.savepath is not None:
                torch.save(self.best_state_dict, self.savepath + "/net_best")
                logging.info("Best network found and saved")
                logging.info("")

        if self.savepath is not None:
            if np.mod(epoch + 1, self.saveevery) == 0:
                torch.save(self.net.state_dict(), self.savepath + "/net_checkpoint")
                logging.info("Network intermittently saved")
                logging.info("")

        return True

    def train_segmentation(self):

        for epoch in range(self.NUM_EPOCHS):
            self.train_one_epoch(epoch)
            self.save_dvc()
            
        if self.validationloader is None:
            self.validation_loss = []
            self.F1_validation_trace_micro = []
            self.F1_validation_trace_macro = []

        results = {
            "Training loss": self.train_loss,
            "Validation loss": self.validation_loss,
            "F1 training micro": self.F1_train_trace_micro,
            "F1 training macro": self.F1_train_trace_macro,
            "F1 validation micro": self.F1_validation_trace_micro,
            "F1 validation macro": self.F1_validation_trace_macro,
            "Best model index": self.best_index,
        }

        self.net.load_state_dict(self.best_state_dict)
        self.losses.to_parquet(self.savepath + "/losses.parquet", engine="pyarrow")
        return self.net, results
## 20240320, Trainer() class added by xchong ##

train_labeling = train_segmentation


def regression_metrics(preds, target):
    """
    Compute a pearson correlation coefficient.
    Useful for validating / understanding regression performance

    Parameters
    ----------
    preds : Predicted values
    target : Target values

    Returns
    -------
    A correlation coefficient
    """

    tmp = corcoef.cc(preds.cpu().flatten(), target.cpu().flatten())
    return tmp


def train_regression(net, trainloader, validationloader, NUM_EPOCHS,
                     criterion, optimizer, device,
                     savepath=None, saveevery=None,
                     scheduler=None, use_amp=False,
                     show=0, clip_value=None):
    """
    Loop through epochs passing dirty images to net.

    :param net: input network
    :param trainloader: data loader with training data
    :param validationloader: data loader with validation data
    :param NUM_EPOCHS: number of epochs
    :param criterion: target function
    :param optimizer: optimization engine
    :param device: the device where we calculate things
    :param savepath: filepath in which we save networks intermittently
    :param saveevery: integer n for saving network every n epochs
    :param scheduler: an optional schedular. can be None
    :param show: print stats every n-th epoch
    :param use_amp: use pytorch automatic mixed precision
    :param clip_value: value for gradient clipping. Can be None.
    :return: A network and run summary stats.
    """

    train_loss = []
    validation_loss = []
    CC_train_trace = []
    CC_validation_trace = []

    best_score = 1e10
    best_index = 0
    best_state_dict = None

    if savepath is not None:
        if saveevery is None:
            saveevery = 1

    for epoch in range(NUM_EPOCHS):
        running_train_loss = 0.0
        running_validation_loss = 0.0
        running_CC_train_val = 0.0
        running_CC_validation_val = 0.0
        tot_train = 0.0
        tot_val = 0.0
        count = 0
        for data in trainloader:
            count += 1

            noisy, target = data  # load noisy and target images

            N_train = noisy.shape[0]
            tot_train += N_train

            noisy = noisy.type(torch.FloatTensor)
            target = target.type(torch.FloatTensor)
            noisy = noisy.to(device)
            target = target.to(device)

            if use_amp is False:
                # forward pass, compute loss and accuracy
                output = net(noisy)
                loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
            else:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    # forward pass, compute loss and accuracy
                    output = net(noisy)
                    loss = criterion(output, target)

                # backpropagation
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # update the parameters
                scaler.step(optimizer)
                scaler.update()

            # update the parameters
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            optimizer.step()

            tmp = regression_metrics(output, target)
            running_CC_train_val += tmp.item()  # *N_train
            running_train_loss += loss.item()

        if scheduler is not None:
            scheduler.step()

        # compute validation step
        with torch.no_grad():
            for dataVal in validationloader:

                x, y = dataVal  # load noisy and target images
                x = x.to(device)
                y = y.to(device)

                N_val = y.shape[0]
                tot_val += N_val
                if criterion.__class__.__name__ == 'CrossEntropyLoss':
                    y = y.type(torch.LongTensor)
                    y = y.to(device).squeeze(1)

                # forward pass, compute validation loss and accuracy
                if use_amp is False:
                    yhat = net(x)
                    val_loss = criterion(yhat, y)
                else:
                    with torch.cuda.amp.autocast():
                        yhat = net(x)
                        val_loss = criterion(yhat, y)

                tmp = regression_metrics(yhat, y)
                running_CC_validation_val += tmp.item()  # *N_val

                # update running validation loss and accuracy
                running_validation_loss += val_loss.item()

        loss = running_train_loss / len(trainloader)
        val_loss = running_validation_loss / len(validationloader)
        CC = running_CC_train_val / len(trainloader)
        CC_val = running_CC_validation_val / len(validationloader)

        train_loss.append(loss)
        validation_loss.append(val_loss)
        CC_train_trace.append(CC)
        CC_validation_trace.append(CC_val)

        if show != 0:

            learning_rates = []
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
            mean_learning_rate = np.mean(np.array(learning_rates))
            if np.mod(epoch + 1, show) == 0:
                print(
                    f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')

            if np.mod(epoch + 1, show) == 0:
                print(f'Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}')
                print(f'Training CC: {CC:.4f}   Validation CC  : {CC_val:.4f} ')

        if val_loss < best_score:
            best_state_dict = net.state_dict()
            best_index = epoch
            best_score = val_loss

            if savepath is not None:
                torch.save(best_state_dict, savepath + '/net_best')
                print('   Best network found and saved')
                print('')

        if savepath is not None:
            if np.mod(epoch + 1, saveevery) == 0:
                torch.save(net.state_dict(), savepath + '/net_checkpoint')
                print('   Network intermittently saved')
                print('')

    results = {"Training loss": train_loss,
               "Validation loss": validation_loss,
               "CC training": CC_train_trace,
               "CC validation": CC_validation_trace,
               "Best model index": best_index}
    net.load_state_dict(best_state_dict)
    return net, results


def train_autoencoder(net, trainloader, validationloader, NUM_EPOCHS,
                      criterion, optimizer, device,
                      savepath=None, saveevery=None,
                      scheduler=None, use_amp=False,
                      show=0, clip_value=None, mask=None):
    """
    Loop through epochs passing dirty images to net.

    :param net: input network
    :param trainloader: data loader with training data
    :param validationloader: data loader with validation data
    :param NUM_EPOCHS: number of epochs
    :param criterion: target function
    :param optimizer: optimization engine
    :param device: the device where we calculate things
    :param savepath: filepath in which we save networks intermittently
    :param saveevery: integer n for saving network every n epochs
    :param scheduler: an optional schedular. can be None
    :param show: print stats every n-th epoch
    :param use_amp: use pytorch automatic mixed precision
    :param clip_value: value for gradient clipping. Can be None.
    :return: A network and run summary stats.
    """

    if mask is None:
        mask = 1.0
    train_loss = []
    validation_loss = []
    CC_train_trace = []
    CC_validation_trace = []

    best_score = 1e10
    best_index = 0
    best_state_dict = None

    if savepath is not None:
        if saveevery is None:
            saveevery = 1

    for epoch in range(NUM_EPOCHS):
        running_train_loss = 0.0
        running_validation_loss = 0.0
        running_CC_train_val = 0.0
        running_CC_validation_val = 0.0
        tot_train = 0.0
        tot_val = 0.0
        count = 0
        for data in trainloader:
            count += 1

            noisy = data[0]

            N_train = noisy.shape[0]
            tot_train += N_train

            noisy = noisy.type(torch.FloatTensor)
            noisy = noisy.to(device)

            if use_amp is False:
                # forward pass, compute loss and accuracy
                output = net(noisy)
                loss = criterion(output*mask, noisy*mask)

                # backpropagation
                optimizer.zero_grad()
                loss.backward()
            else:
                scaler = torch.cuda.amp.GradScaler()
                with torch.cuda.amp.autocast():
                    # forward pass, compute loss and accuracy
                    output = net(noisy)
                    loss = criterion(output*mask, noisy*mask)

                # backpropagation
                optimizer.zero_grad()
                scaler.scale(loss).backward()

                # update the parameters
                scaler.step(optimizer)
                scaler.update()

            # update the parameters
            if clip_value is not None:
                torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
            optimizer.step()


            tmp = regression_metrics(output, noisy)
            running_CC_train_val += tmp.item()  # *N_train
            running_train_loss += loss.item()

        # compute validation step
        if scheduler is not None:
            scheduler.step()

        with torch.no_grad():
            for dataVal in validationloader:

                x = dataVal[0]  # load noisy and target images
                x = x.to(device)

                N_val = x.shape[0]
                tot_val += N_val

                # forward pass, compute validation loss and accuracy
                if use_amp is False:
                    yhat = net(x)
                    val_loss = criterion(yhat*mask, x*mask)
                else:
                    with torch.cuda.amp.autocast():
                        yhat = net(x)
                        val_loss = criterion(yhat*mask, x*mask)

                tmp = regression_metrics(yhat, x)
                running_CC_validation_val += tmp.item()  # *N_val

                # update running validation loss and accuracy
                running_validation_loss += val_loss.item()

        loss = running_train_loss / len(trainloader)
        val_loss = running_validation_loss / len(validationloader)
        CC = running_CC_train_val / len(trainloader)
        CC_val = running_CC_validation_val / len(validationloader)

        train_loss.append(loss)
        validation_loss.append(val_loss)
        CC_train_trace.append(CC)
        CC_validation_trace.append(CC_val)

        if show != 0:

            learning_rates = []
            for param_group in optimizer.param_groups:
                learning_rates.append(param_group['lr'])
            mean_learning_rate = np.mean(np.array(learning_rates))
            if np.mod(epoch + 1, show) == 0:
                print(
                    f'Epoch {epoch + 1} of {NUM_EPOCHS} | Learning rate {mean_learning_rate:4.3e}')

            if np.mod(epoch + 1, show) == 0:
                print(f'Training Loss: {loss:.4e} | Validation Loss: {val_loss:.4e}')
                print(f'Training CC: {CC:.4f}   Validation CC  : {CC_val:.4f} ')

        if val_loss < best_score:
            best_state_dict = net.state_dict()
            best_index = epoch
            best_score = val_loss

            if savepath is not None:
                torch.save(best_state_dict(), savepath + '/net_best')
                print('   Best network found and saved')
                print('')

        if savepath is not None:
            if np.mod(epoch + 1, saveevery) == 0:
                torch.save(net.state_dict, savepath + '/net_checkpoint')
                print('   Network intermittently saved')
                print('')

    results = {"Training loss": train_loss,
               "Validation loss": validation_loss,
               "CC training": CC_train_trace,
               "CC validation": CC_validation_trace,
               "Best model index": best_index}
    net.load_state_dict(best_state_dict)
    return net, results


def autoencode_and_classify_training(net,
                                     trainloader,
                                     validationloader,
                                     macro_epochs,
                                     mini_epochs,
                                     criteria_autoencode,
                                     minimizer_autoencode,
                                     criteria_classify,
                                     minimizer_classify,
                                     device,
                                     scheduler=None,
                                     clip_value=None,
                                     show=0, ):
    """

    Parameters
    ----------
    net : input network
    trainloader : training data loader
    validationloader : validation data loader
    macro_epochs : the total number of passes of optimization
    mini_epochs : the number of passes of the data for autoencoding or classification
    criteria_autoencode : loss function for autoencoding
    minimizer_autoencode : minimizer for autyoencoding
    criteria_classify : loss function for classification
    minimizer_classify : minimizer for classification
    device : where do we train?
    scheduler : a scheduler, optional.
    clip_value : a clip value to avoid large shifts. optional
    show : when 0 show all, if set a number, every n-th pass of the data is shown.

    Returns
    -------
    optimzed network, performance trace for autoencoding, performance trace for classification
    """

    train_losses_AE = []
    validation_losses_AE = []
    train_losses_C = []
    validation_losses_C = []

    CC_train = []
    CC_validation = []
    F1_micro_train = []
    F1_micro_validation = []
    F1_macro_train = []
    F1_macro_validation = []

    overall_count = 0

    for epoch in range(macro_epochs):

        mode = "classify"
        if epoch % 2 == 0:
            mode = "autoencode"

        for mini_epoch in range(mini_epochs):
            overall_count += 1
            running_train_loss_AE = 0
            running_train_loss_C = 0
            running_train_CC = 0
            running_F1_micro_train = 0
            running_F1_macro_train = 0

            running_validation_loss_AE = 0
            running_validation_loss_C = 0
            running_validation_CC = 0
            running_F1_micro_validation = 0
            running_F1_macro_validation = 0

            for batch in trainloader:
                imgs, lbls = batch
                imgs = imgs.type(torch.FloatTensor).to(device)
                lbls = lbls.squeeze(1).type(torch.LongTensor).to(device)

                imgs_AE, p_class = net(imgs)
                is_only_missing = torch.sum(lbls == -1).type(torch.int).item() == lbls.shape[0]

                AE_loss = criteria_autoencode(imgs_AE, imgs)
                C_loss = 0.0
                if not is_only_missing:
                    C_loss = criteria_classify(p_class, lbls)

                # backpropagation
                if mode == "autoencode":
                    minimizer_autoencode.zero_grad()
                    AE_loss.backward()

                    # update the parameters
                    if clip_value is not None:
                        torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
                    minimizer_autoencode.step()
                    if scheduler is not None:
                        scheduler.step()

                if mode == "classify":
                    if not is_only_missing:
                        minimizer_classify.zero_grad()
                        C_loss.backward()

                        # update the parameters
                        if clip_value is not None:
                            torch.nn.utils.clip_grad_value_(net.parameters(), clip_value)
                        minimizer_classify.step()


                tmp = regression_metrics(imgs_AE, imgs)
                running_train_CC += tmp.item()
                running_train_loss_AE += AE_loss.item()

                if not is_only_missing:
                    running_train_loss_C += C_loss.item()
                    tmp_micro, tmp_macro = segmentation_metrics(p_class, lbls)
                    running_F1_micro_train += tmp_micro.item()
                    running_F1_macro_train += tmp_macro.item()

            running_train_loss_AE /= len(trainloader)
            running_train_loss_C /= len(trainloader)
            running_train_CC /= len(trainloader)
            running_F1_micro_train /= len(trainloader)
            running_F1_macro_train /= len(trainloader)

            train_losses_AE.append(running_train_loss_AE)
            train_losses_C.append(running_train_loss_C)
            CC_train.append(running_train_CC)
            F1_micro_train.append(running_F1_micro_train)
            F1_macro_train.append(running_F1_macro_train)

            if scheduler is not None:
                scheduler.step()

            for batch in validationloader:
                imgs, lbls = batch
                imgs = imgs.type(torch.FloatTensor).to(device)
                lbls = lbls.squeeze(1).type(torch.LongTensor).to(device)

                with torch.no_grad():
                    imgs_AE, p_class = net(imgs)

                    AE_loss = criteria_autoencode(imgs_AE, imgs)
                    C_loss = criteria_classify(p_class, lbls)

                    tmp = regression_metrics(imgs_AE, imgs)
                    running_validation_CC += tmp.item()
                    running_validation_loss_AE += AE_loss.item()
                    running_validation_loss_C += C_loss.item()

                    tmp_micro, tmp_macro = segmentation_metrics(p_class, lbls)
                    running_F1_micro_validation += tmp_micro.item()
                    running_F1_macro_validation += tmp_macro.item()

            running_validation_loss_AE /= len(validationloader)
            running_validation_loss_C /= len(validationloader)
            running_validation_CC /= len(validationloader)
            running_F1_micro_validation /= len(validationloader)
            running_F1_macro_validation /= len(validationloader)

            validation_losses_AE.append(running_validation_loss_AE)
            validation_losses_C.append(running_validation_loss_C)
            CC_validation.append(running_validation_CC)
            F1_micro_validation.append(running_F1_micro_validation)
            F1_macro_validation.append(running_F1_macro_validation)

            if show != 0:
                learning_rates = []
                for param_group in minimizer_autoencode.param_groups:
                    learning_rates.append(param_group['lr'])
                mean_learning_rate = np.mean(np.array(learning_rates))
                if np.mod(epoch + 1, show) == 0:
                    print(
                        f'Epoch {epoch + 1:4d},  of {macro_epochs} >-*-< Mini Epoch  {mini_epoch + 1:4d} of {mini_epochs} >-*-< Learning rate {mean_learning_rate:4.3e}')

                if np.mod(epoch + 1, show) == 0:
                    if mode == "autoencode":
                        print(f'** Autoencoding Losses **      <---- Now Optimizing')
                    else:
                        print(f'** Autoencoding Losses **')

                    print(
                        f'Training Loss    : {running_train_loss_AE:.4e} | Validation Loss      : {running_validation_loss_AE:.4e}')
                    print(
                        f'Training CC      : {running_train_CC:.4f}     | Validation CC        : {running_validation_CC:.4f} ')

                    if mode == "classify":
                        print(f'** Classification Losses **    <---- Now Optimizing')
                    else:
                        print(f'** Classification Losses **')
                    print(
                        f'Training Loss    : {running_train_loss_C:.4e} | Validation Loss      : {running_validation_loss_C:.4e}')
                    print(
                        f'Training F1 Macro: {running_F1_macro_train:.4f}     | Validation F1 Macro  : {running_F1_macro_validation:.4f} ')
                    print(
                        f'Training F1 Micro: {running_F1_micro_train:.4f}     | Validation F1 Micro  : {running_F1_micro_validation:.4f} ')
                    print()
    results_AE = {"Training loss": train_losses_AE,
                  "Validation loss": validation_losses_AE,
                  "CC training": CC_train,
                  "CC validation": CC_validation}

    results_C = {"Training loss": train_losses_C,
                 "Validation loss": validation_losses_C,
                 "F1 training micro": F1_micro_train,
                 "F1 training macro": F1_macro_train,
                 "F1 validation micro": F1_micro_validation,
                 "F1 validation macro": F1_macro_validation,
                 }
    return net, results_AE, results_C


def tst():
    print('Add a test')
    print(torch.__version__)
    print(torch.version.cuda)

    print(torch._C._cuda_getCompiledVersion(), 'cuda compiled version')


if __name__ == "__main__":
    tst()
