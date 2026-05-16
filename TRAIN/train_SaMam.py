import argparse
from argparse import ArgumentParser
from pathlib import Path
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

project_root = os.path.abspath('..')
import sys

sys.path.append(project_root)

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger

from TRAIN.lightning_module import dataset
from TRAIN.lightning_module.datamodule import DataModule
from TRAIN.lightning_module.lightningmodel import LightningModel

import time
import json
from datetime import datetime
from pytorch_lightning.callbacks import Callback
import pytorch_lightning as pl
# torch.backends.cudnn.enabled = False

class TensorBoardImageLogger(TensorBoardLogger):
    """
    Wrapper for TensorBoardLogger which logs images to disk,
        instead of the TensorBoard log file.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        exp = self.experiment

        # if not hasattr(exp, 'add_image'):
        exp.add_image = self.add_image

    def add_image(self, tag, img_tensor, global_step):
        dir = Path(self.log_dir, 'images')
        dir.mkdir(parents=True, exist_ok=True)

        file = dir.joinpath(f'{tag}_{global_step:09}.jpg')
        dataset.save(img_tensor, file)
        
class TrainingTimeLogger(Callback):
    def __init__(self, output_file):
        super().__init__()
        self.output_file = output_file
        self.start_time = None
        self.start_global_step = 0

    def on_train_start(self, trainer, pl_module):
        # Synchronize so GPU work before timing is completed.
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        self.start_time = time.perf_counter()
        self.start_global_step = int(trainer.global_step)

    def on_train_end(self, trainer, pl_module):
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        end_time = time.perf_counter()
        elapsed_seconds = end_time - self.start_time

        end_global_step = int(trainer.global_step)
        iterations_this_run = end_global_step - self.start_global_step

        iterations_per_second = (
            iterations_this_run / elapsed_seconds
            if elapsed_seconds > 0
            else 0.0
        )

        result = {
            "finished_at": datetime.now().isoformat(timespec="seconds"),
            "elapsed_seconds": elapsed_seconds,
            "elapsed_minutes": elapsed_seconds / 60.0,
            "elapsed_hours": elapsed_seconds / 3600.0,
            "start_global_step": self.start_global_step,
            "end_global_step": end_global_step,
            "iterations_this_run": iterations_this_run,
            "max_steps": trainer.max_steps,
            "iterations_per_second": iterations_per_second,
            "seconds_per_iteration": (
                elapsed_seconds / iterations_this_run
                if iterations_this_run > 0
                else None
            ),
        }

        output_dir = os.path.dirname(self.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        with open(self.output_file, "w") as f:
            json.dump(result, f, indent=4)

        print("\n========== Training Time Summary ==========")
        print(f"Elapsed time: {elapsed_seconds:.2f} seconds")
        print(f"Elapsed time: {elapsed_seconds / 60.0:.2f} minutes")
        print(f"Iterations this run: {iterations_this_run}")
        print(f"End global step: {end_global_step}")
        print(f"Iterations/sec: {iterations_per_second:.4f}")
        if iterations_this_run > 0:
            print(f"Seconds/iteration: {elapsed_seconds / iterations_this_run:.4f}")
        print(f"Saved to: {self.output_file}")
        print("===========================================\n")


def parse_args():
    # Init parser
    parser = ArgumentParser()

    parser.add_argument('--output', type=str, default="./checkpoint.ckpt",help="")

    # train_model setting
    parser.add_argument('--gpus', nargs='+', default='0',
                        help='GPU for training. Command usage sample: --gpus 0 1')
    parser.add_argument('--iterations', type=int, default=200_000,
                        help='The number of training iterations.')
    parser.add_argument('--log-dir', type=str, default='./',
                        help='The directory where the logs are saved to.')
    parser.add_argument('--checkpoint', type=str,
                        default=None,
                        help='Resume training from a checkpoint file.')
    parser.add_argument('--val-interval', type=int, default=1000,
                        help='How often a validation step is performed. '
                             'Applies the model to several fixed images and calculate the losses.')

    # dataset setting
    # you need to set your own training content and style datasets manually
    parser.add_argument('--content', type=str, default='/home/liuhd/dataset_ST/train2017',
                        help='Directory with content images.')
    parser.add_argument('--style', type=str, default='/home/liuhd/dataset_ST/train',
                        help='Directory with style images.')
    # test data during training
    parser.add_argument('--test-content', type=str, default='./test_images/content',
                        help='Directory with test content images. If not set, takes 5 random train_model content images.')
    parser.add_argument('--test-style', type=str, default='./test_images/style',
                        help='Directory with test style images. If not set, takes 5 random train_model style images.')
    parser.add_argument('--batch-size', type=int, default=8,
                        help='Training batch size.')
    parser.add_argument('--accumulate-grad-batches', type=int, default=1,
                        help='Accumulate Grad Batches.')

    # model setting
    parser.add_argument('--nVSSMs', type=int, default=2)
    parser.add_argument('--nSAVSSMs', type=int, default=2)
    parser.add_argument('--nSAVSSGs', type=int, default=2)
    parser.add_argument('--embed-dim', type=int, default=256)
    parser.add_argument('--patch-size', type=int, default=8)
    parser.add_argument('--representation-dim', type=int, default=64)
    parser.add_argument('--d-state', type=int, default=16)
    parser.add_argument('--expand', type=float, default=2.0)
    parser.add_argument('--compress-ratio', type=int, default=8)
    parser.add_argument('--squeeze-factor', type=int, default=8)
    parser.add_argument('--mamba-from-trion', type=int, default=1) #########

    # training setting
    # Losses
    parser.add_argument('--style-weight', type=float, default=7.0)
    parser.add_argument('--content-weight', type=float, default=7.0)
    parser.add_argument('--if-identity-loss', type=int, default=1)
    parser.add_argument('--lambda1', type=float, default=70.0)
    parser.add_argument('--lambda2', type=float, default=1.0)
    parser.add_argument('--ssim-weight', type=float, default=5.0)
    
    # Optimizer
    parser.add_argument('--lr', type=float, default=0.0001)
    
    ## APPLY IMPROVEMENTS apply_huber_loss
    parser.add_argument('--low-vram', action='store_true', help="Enable activation checkpointing")
    parser.add_argument('--apply-huber-loss', action='store_true', help="apply_huber_loss")
    parser.add_argument('--apply-SSIM-loss', action='store_true', help="apply_SSIM_loss")
    parser.add_argument('--apply-identity-loss', action='store_true', help="apply_identity_loss")
    
    parser.formatter_class = argparse.ArgumentDefaultsHelpFormatter

    # Add early stopping
    parser.add_argument('--early-stopping',action='store_true',help='Enable early stopping during training')
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--delta', type=float, default=4.0)

    #loss log file path
    parser.add_argument('--loss-log', type=str, default='./loss_logs/loss.txt',
                        help='loss log path/filename.txt')

    #quiet
    parser.add_argument('--quiet', action='store_true', help="quiet")

    parser.add_argument(
        '--time-log',
        type=str,
        default='./loss_logs/training_time.json',
        help='Path to save training time, iterations, and iterations/sec.'
    )

    parser.add_argument(
        "--activation",
        type=str,
        default="silu",
        choices=["silu", "tanh", "relu", "softsign", "tanhshrink"],
        help="Activation function for SS2D encoder"
    )

    parser.add_argument(
        '--huber-deltas',
        type=float,
        nargs=3,
        metavar=('C', 'S', 'I'),
        default=[0.5, 0.1, 0.1],
        help='huber deltas'
    )

    parser.add_argument('--seed', type=int, default=1234)

    parser.add_argument('--apply-batching', action='store_true', help="enable true batching")
        
    return vars(parser.parse_args())

if __name__ == '__main__':
    args = parse_args()

    if args['quiet']:
        print("quiet missing")

    pl.seed_everything(args['seed'], workers=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    
    gpus = []
    for i in args['gpus']:
        gpus.append(int(i))

    if args['checkpoint'] is None:
        max_epochs = 1
        model = LightningModel(**args)
    else:
        # We need to increment the max_epoch variable, because PyTorch Lightning will
        # resume training from the beginning of the next epoch if resuming from a mid-epoch checkpoint.
        max_epochs = torch.load(args['checkpoint'])['epoch'] + 1
        model = LightningModel.load_from_checkpoint(checkpoint_path=args['checkpoint'])

    datamodule = DataModule(**args)        
    logger = TensorBoardImageLogger(args['log_dir'], name='logs')
    lr_monitor = LearningRateMonitor(logging_interval='step')

    #######
    #checkpoint_cb = ModelCheckpoint(
    #    dirpath=os.path.dirname(args["output"]),
    #    filename=Path(args["output"]).stem,   # "checkpoint"
    #    monitor="val_loss",
    #    mode="min",
    #    save_top_k=1,
    #    save_last=True,
    #    auto_insert_metric_name=False,
    #)
    ######
    
    early_stop = None

    if args['early_stopping']:
        early_stop = EarlyStopping(
            monitor="val_loss",
            min_delta=args['delta'],
            patience=args['patience'],
            mode="min",
            verbose=True
        )
        
    time_logger = TrainingTimeLogger(args['time_log'])

    callbacks = [lr_monitor, time_logger]
    
    if early_stop is not None:
        callbacks.append(early_stop)
    
    trainer = Trainer(
        accelerator="gpu",
        devices=gpus,
        max_epochs=max_epochs,
        max_steps=args['iterations'],
        val_check_interval=args['val_interval'],
        logger=logger,
        callbacks=callbacks,
        accumulate_grad_batches=args['accumulate_grad_batches'],
        #deterministic=True, [does not work]
        benchmark=False,
        num_sanity_val_steps=0,
        #precision="16-mixed", [does not work]
    )

    trainer.fit(model, datamodule=datamodule)
    output = args["output"]
    trainer.save_checkpoint(output)
