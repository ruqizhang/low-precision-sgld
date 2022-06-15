import sys
sys.path.append("..")
import argparse
import time
import torch
import torch.nn.functional as F
import utils
import tabulate
import models
from qtorch.quant import *
from qtorch.optim import OptimLP
from torch.optim import SGD
from qtorch import BlockFloatingPoint, FixedPoint, FloatingPoint
from optim import OptimLP
import os
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
num_types = ["weight", "activate", "grad", "error"]

parser = argparse.ArgumentParser(description="SGLD training")
parser.add_argument(
    "--dataset", type=str, default="CIFAR10", help="dataset name: CIFAR10 or CIFAR100"
)
parser.add_argument(
    "--data_path",
    type=str,
    default="../data",
    metavar="PATH",
    help='path to datasets location (default: "../data")',
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=128,
    metavar="N",
    help="input batch size (default: 128)",
)
parser.add_argument(
    "--model",
    type=str,
    default=None,
    required=True,
    metavar="MODEL",
    help="model name (default: None)",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=200,
    help="number of epochs to train (default: 200)",
)
parser.add_argument(
    "--eval_freq",
    type=int,
    default=5,
    help="evaluation frequency (default: 5)",
)
parser.add_argument(
    "--lr_init",
    type=float,
    default=0.01,
    help="initial learning rate (default: 0.01)",
)
parser.add_argument(
    "--wd", type=float, default=1e-4, help="weight decay (default: 1e-4)"
)
parser.add_argument(
    "--seed", type=int, default=1, help="random seed (default: 1)"
)
for num in num_types:
    parser.add_argument(
        "--wl-{}".format(num),
        type=int,
        default=-1,
        help="word length in bits for {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--fl-{}".format(num),
        type=int,
        default=-1,
        help="number of fractional bits for {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--{}-man".format(num),
        type=int,
        default=-1,
        help="number of bits to use for mantissa of {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--{}-exp".format(num),
        type=int,
        default=-1,
        help="number of bits to use for exponent of {}; -1 if full precision.".format(num),
    )
    parser.add_argument(
        "--{}-type".format(num),
        type=str,
        default="full",
        choices=["fixed", "block", "float", "full","hfloat"],
        help="quantization type for {}; fixed or block.".format(num),
    )
    parser.add_argument(
        "--{}-rounding".format(num),
        type=str,
        default="stochastic",
        choices=["stochastic", "nearest"],
        help="rounding method for {}, stochastic or nearest".format(num),
    )
parser.add_argument(
    "--noise", type=bool, default=False, help="whether to add Gaussian noise in the update. True: SGLD, False: SGD"
)
parser.add_argument(
    "--temperature", type=float, default=0.001, help="temperature"
)
parser.add_argument(
    "--quant_acc", type=int, default=-1, help="accumulator precision. -1: low-precision, -2: full-precision"
)
parser.add_argument(
    "--quant_type", type=str, default="naive", help="quant type"
)
parser.add_argument(
    "--num_savemodel", type=int, default=50, help="number of saved models"
)
parser.add_argument(
    "--lr_type", type=str, default='decay', help="LR schedule"
)
parser.add_argument(
    "--M", type=int, default=7, help="number of cycles in cyclical LR schedule"
)

args = parser.parse_args()
args.cuda = torch.cuda.is_available()
utils.set_seed(args.seed, args.cuda)

loaders = utils.get_data(args.dataset, args.data_path, args.batch_size, num_workers=0)
print('quant_type:', args.quant_type, 'quant_acc', args.quant_acc)

def make_number(dataset,number, wl=-1, fl=-1, exp=-1, man=-1):
    if number == "fixed":
        return FixedPoint(wl, fl)
    elif number == "block":
        return BlockFloatingPoint(wl,dim=0)
    elif number == "float":
        return FloatingPoint(exp, man)
    elif number == "full":
        return FloatingPoint(8, 23)
    else:
        raise ValueError("invalid number type")

if args.wl_weight >0: # low-precision model
    number_dict = {}
    for num in num_types:
        num_wl = getattr(args, "wl_{}".format(num))
        num_fl = getattr(args, "fl_{}".format(num))
        num_type = getattr(args, "{}_type".format(num))
        num_rounding = getattr(args, "{}_rounding".format(num))
        num_man = getattr(args, "{}_man".format(num))
        num_exp = getattr(args, "{}_exp".format(num))
        number_dict[num] = make_number(args.dataset,
            num_type, wl=num_wl, fl=num_fl, exp=num_exp, man=num_man
        )
        print("{:10}: {}".format(num, number_dict[num]))

    weight_quantizer = quantizer(
        forward_number=number_dict["weight"], forward_rounding=args.weight_rounding
    )
    grad_quantizer = quantizer(
        forward_number=number_dict["grad"], forward_rounding=args.grad_rounding
    )
    acc_err_quant = lambda: Quantizer(
        number_dict["activate"],
        number_dict["error"],
        args.activate_rounding,
        args.error_rounding,
    )

    # Build model
    print("Model: {}".format(args.model))
    model_cfg = getattr(models, args.model)
    model_cfg.kwargs.update({"quant": acc_err_quant})

    if args.dataset == "CIFAR10":
        num_classes = 10
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "CIFAR100":
        num_classes = 100
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "IMAGENET12":
        num_classes = 1000
    model.cuda()
    if args.quant_acc == -2:
        quant_acc = "full"
    else:
        quant_acc = None
    optimizer = SGD(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
    optimizer = OptimLP(
        optimizer,
        weight_quant=weight_quantizer,
        grad_quant=grad_quantizer,
        acc_quant=quant_acc,
        noise=args.noise,
        temperature=args.temperature,
        datasize=ds,
        WL=args.wl_weight,
        FL=args.fl_weight,
        EXP=args.weight_exp,
        MAN=args.weight_man,
        quant_type=args.quant_type,
        number_type=args.weight_type
    )
else: # full-precision model
    print("Model: {}".format(args.model))
    model_cfg = getattr(models, args.model)
    if args.dataset == "CIFAR10":
        num_classes = 10
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "CIFAR100":
        num_classes = 100
        ds = 50000
        model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
        criterion = F.cross_entropy
    elif args.dataset == "IMAGENET12":
        num_classes = 1000

    model.cuda()
    optimizer = SGD(model.parameters(), lr=args.lr_init, weight_decay=args.wd)
    optimizer = OptimLP(
        optimizer,
        noise=args.noise,
        temperature=args.temperature,
        datasize=ds
    )

# Prepare logging
columns = ["ep", "lr", "tr_loss", "tr_acc", "tr_time", "te_loss", "te_acc", "te_time"]
# cyclic lr params
num_batch = int(ds/args.batch_size)+1
T = args.epochs*num_batch
num_savemodel_percycle = args.num_savemodel/args.M

mt = 0
for epoch in range(args.epochs):
    time_ep = time.time()

    train_res = utils.run_epoch(args,
        loaders["train"], model, criterion, epoch, num_batch=num_batch, T=T, optimizer=optimizer, phase="train"
    )
    time_pass = time.time() - time_ep
    train_res["time_pass"] = time_pass

    if (
        epoch == 0
        or epoch % args.eval_freq == args.eval_freq - 1
        or epoch == args.epochs - 1
    ):
        time_ep = time.time()
        test_res = utils.run_epoch(args,loaders["test"], model, criterion, epoch, num_batch=num_batch, T=T, phase="eval")
        time_pass = time.time() - time_ep
        test_res["time_pass"] = time_pass
    else:
        test_res = {"loss": None, "accuracy": None, "time_pass": None}

    values = [
        epoch + 1,
        optimizer.param_groups[0]["lr"],
        train_res["loss"],
        train_res["accuracy"],
        train_res["time_pass"],
        test_res["loss"],
        test_res["accuracy"],
        test_res["time_pass"],
    ]

    table = tabulate.tabulate([values], columns, tablefmt="simple", floatfmt="8.4f")
    if epoch % 40 == 0:
        table = table.split("\n")
        table = "\n".join([table[1]] + table)
    else:
        table = table.split("\n")[2]
    print(table)
    if args.lr_type == 'cyclic':
        if (epoch%(args.epochs/args.M))+1>(args.epochs/args.M-num_savemodel_percycle): 
            print('save!')
            torch.save(model.state_dict(),'checkpoints/%s_%s_%s_%d_lrtype%s_noise%s_acc%d_type%s_%i.pt'%(args.weight_type,args.dataset,args.model,args.wl_weight,args.lr_type,args.noise,args.quant_acc,args.quant_type,mt))
            mt += 1
    elif args.noise and epoch>args.epochs-args.num_savemodel-1:
        print('save!')
        torch.save(model.state_dict(),'checkpoints/%s_%s_%s_%d_lrtype%s_noise%s_acc%d_type%s_%i.pt'%(args.weight_type,args.dataset,args.model,args.wl_weight,args.lr_type,args.noise,args.quant_acc,args.quant_type,mt))
        mt += 1  
if not args.noise and args.lr_type == 'decay':
    print('save!')
    torch.save(model.state_dict(),'checkpoints/%s_%s_%s_%d_noise%s_acc%d_%i.pt'%(args.weight_type,args.dataset,args.model,args.wl_weight,args.noise,args.quant_acc,mt))
        