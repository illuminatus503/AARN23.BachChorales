import os
import time
import math

import matplotlib.pyplot as plt

from torch import save, set_grad_enabled, sum, max
from torch import optim, cuda, load, device
from torch.nn.utils import clip_grad_norm_

from preprocessing.nn_dataset import (
    get_data_set,
    TOTAL_BATCHES,
    TRAIN_BATCHES,
    N_TOKENS,
)

from train.models import CrossEntropyTimeDistributedLoss
from train.models import TonicNet

from train.external import Lookahead

"""
File containing functions which train various neural networks defined in train.models
"""


CV_PHASES = ["train", "val"]
TRAIN_ONLY_PHASES = ["train"]


def train_TonicNet(
    epochs=30,
    save_model=True,
    load_path="",
    shuffle_batches=False,
    num_batches=TOTAL_BATCHES,
    val=True,
    train_emb_freq=1,
    lr_range_test=False,
    sanity_test=False,
    factorize=False,
):
    model = TonicNet(
        nb_tags=N_TOKENS, z_dim=32, nb_layers=3, nb_rnn_units=256, dropout=0.3
    )

    if load_path != "":
        try:
            if cuda.is_available():
                model.load_state_dict(load(load_path)["model_state_dict"])
            else:
                model.load_state_dict(
                    load(load_path, map_location=device("cpu"))["model_state_dict"]
                )
            print("loded params from", load_path)
        except:
            raise ImportError(
                f"No file located at {load_path}, could not load parameters"
            )

    # Factorize the model after loading it
    num_params_before = 0
    for p in model.parameters():
        if p.requires_grad:
            num_params_before += p.numel()
    print(f"Number of parameters: {num_params_before}")

    if factorize:
        model.factorize_model()

        # Print the status of the model after factorization
        # num_params_after = 0
        # for p in model.parameters():
        #     if p.requires_grad:
        #         num_params_after += p.numel()
        # print(f"Number of parameters (after factorization): {num_params_after}")
        # print(f"Reduction of params.? {num_params_before / num_params_after - 1:^2.2%}")

    print()
    print(model)
    print()

    if cuda.is_available():
        model.cuda()

    base_lr = 0.2
    max_lr = 0.2

    if lr_range_test:
        base_lr = 0.000003
        max_lr = 0.5

    step_size = 3 * min(TRAIN_BATCHES, num_batches)

    if sanity_test:
        base_optim = optim.RAdam(model.parameters(), lr=base_lr)
        optimiser = Lookahead(base_optim, k=5, alpha=0.5)
    else:
        optimiser = optim.SGD(model.parameters(), base_lr)
    criterion = CrossEntropyTimeDistributedLoss()

    print(criterion)
    print(f"min lr: {base_lr}, max_lr: {max_lr}, stepsize: {step_size}")
    print()

    if not sanity_test and not lr_range_test:
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimiser,
            max_lr,
            epochs=60,
            steps_per_epoch=TRAIN_BATCHES,
            pct_start=0.3,
            anneal_strategy="cos",
            cycle_momentum=True,
            base_momentum=0.8,
            max_momentum=0.95,
            div_factor=25.0,
            final_div_factor=1000.0,
            last_epoch=-1,
        )

    elif lr_range_test:
        lr_lambda = lambda x: math.exp(
            x * math.log(max_lr / base_lr) / (epochs * num_batches)
        )
        scheduler = optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)

    best_val_loss = 100.0

    if lr_range_test:
        lr_find_loss = []
        lr_find_lr = []

        itr = 0
        smoothing = 0.05

    if val:
        phases = CV_PHASES
    else:
        phases = TRAIN_ONLY_PHASES

    for epoch in range(epochs):
        start = time.time()
        pr_interval = 50

        print(f"Beginning EPOCH {epoch + 1}")

        for phase in phases:
            count = 0
            batch_count = 0
            loss_epoch = 0
            running_accuray = 0.0
            running_batch_count = 0
            print_loss_batch = 0  # Reset on print
            print_acc_batch = 0  # Reset on print

            print(f"\n\tPHASE: {phase}")

            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            for x, y, psx, i, c in get_data_set(
                phase, shuffle_batches=shuffle_batches, return_I=1
            ):
                model.zero_grad()

                if phase == "train" and (epoch > -1 or load_path != ""):
                    if train_emb_freq < 1000:
                        train_emb = (count % train_emb_freq) == 0
                    else:
                        train_emb = False

                else:
                    train_emb = False

                with set_grad_enabled(phase == "train"):
                    y_hat = model(x, z=i, train_embedding=train_emb)
                    _, preds = max(y_hat, 2)
                    loss = criterion(
                        y_hat,
                        y,
                    )

                if phase == "train":
                    loss.backward()
                    clip_grad_norm_(model.parameters(), 5)
                    optimiser.step()
                    if not sanity_test:
                        scheduler.step()

                loss_epoch += loss.item()
                print_loss_batch += loss.item()
                running_accuray += sum(preds == y)
                print_acc_batch += sum(preds == y)

                count += 1
                batch_count += x.shape[1]
                running_batch_count += x.shape[1]

                if lr_range_test:
                    lr_step = optimiser.state_dict()["param_groups"][0]["lr"]
                    lr_find_lr.append(lr_step)

                    # smooth the loss
                    if itr == 0:
                        lr_find_loss.append(min(loss, 7))
                    else:
                        loss = (
                            smoothing * min(loss, 7)
                            + (1 - smoothing) * lr_find_loss[-1]
                        )
                        lr_find_loss.append(loss)

                    itr += 1

                # print loss for recent set of batches
                if count % pr_interval == 0:
                    ave_loss = print_loss_batch / pr_interval
                    ave_acc = 100 * print_acc_batch.float() / running_batch_count
                    print_acc_batch = 0
                    running_batch_count = 0
                    print("\t\t[%d] loss: %.3f, acc: %.3f" % (count, ave_loss, ave_acc))
                    print_loss_batch = 0

                if count == num_batches:
                    break

            # calculate loss and accuracy for phase
            ave_loss_epoch = loss_epoch / count
            epoch_acc = 100 * running_accuray.float() / batch_count
            print(
                "\tfinished %s phase [%d] loss: %.3f, acc: %.3f"
                % (phase, epoch + 1, ave_loss_epoch, epoch_acc)
            )

        print("\n\ttime:", __time_since(start), "\n")

        # save model when validation loss improves
        if ave_loss_epoch < best_val_loss:
            best_val_loss = ave_loss_epoch
            print("\tNEW BEST LOSS: %.3f" % ave_loss_epoch, "\n")

            if save_model:
                if factorize:
                    model_name = "TonicNet_Factorized"
                else:
                    model_name = "TonicNet"

                __save_model(epoch, ave_loss_epoch, model, model_name, epoch_acc)
        else:
            print("\tLOSS DID NOT IMPROVE FROM %.3f" % best_val_loss, "\n")

    print("DONE")

    if lr_range_test:
        plt.plot(lr_find_lr, lr_find_loss)
        plt.xscale("log")
        plt.grid("true")
        plt.savefig("lr_finder.png")
        plt.show()


def __save_model(epoch, ave_loss_epoch, model, model_name, acc):
    test = os.listdir("eval")

    for item in test:
        if item.endswith(".pt"):
            os.remove(os.path.join("eval", item))

    path_loss = round(ave_loss_epoch, 3)
    path_acc = "%.3f" % acc
    path = f"eval/{model_name}_epoch-{epoch}_loss-{path_loss}_acc-{path_acc}.pt"
    save(
        {"epoch": epoch, "model_state_dict": model.state_dict(), "loss": path_loss},
        path,
    )
    print("\tSAVED MODEL TO:", path)


def __time_since(t):
    now = time.time()
    s = now - t
    return "%s" % (__as_minutes(s))


def __as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)
