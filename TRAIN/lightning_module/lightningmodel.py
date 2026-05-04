from argparse import ArgumentParser
from math import sqrt
from statistics import mean
import os
import pytorch_lightning as pl
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torchvision.utils import make_grid

from MODEL.SaMam_model import SaMam
from LOSS.losses import Integration_loss

class LightningModel(pl.LightningModule):

    def __init__(self,
                 # model setting
                 nVSSMs, nSAVSSMs, nSAVSSGs,
                 embed_dim, patch_size, representation_dim, d_state, expand, compress_ratio, squeeze_factor,
                 mamba_from_trion,

                 # loss setting
                 style_weight=7.0,
                 content_weight=7.0,
                 lambda1=70.0, lambda2=1.0,
                 ssim_weight = 5,

                 lr=0.0001,

                 low_vram=False,
                 apply_huber_loss=False,
                 apply_SSIM_loss=False,
                 apply_identity_loss=False,
                 loss_log="./loss_logs/loss.txt",
                 **_):
        super().__init__()
        self.validation_step_outputs = []
        self.save_hyperparameters()
        self.loss_log_file = None
        self.lr = lr
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.ssim_weight = ssim_weight
        ##################
        self._printed_val = False
        ##################
        
        # Style loss
        self.loss_func = Integration_loss(apply_huber_loss = apply_huber_loss, apply_SSIM_loss = apply_SSIM_loss, apply_identity_loss=apply_identity_loss)

        self.val_loss_buffer = []
        self.pending_val_write = False

        self.loss_log = loss_log

        # Model
        self.model = SaMam(
            nVSSMs=nVSSMs,
            nSAVSSMs=nSAVSSMs,
            nSAVSSGs=nSAVSSGs,

            embed_dim=embed_dim,
            patch_size=patch_size,
            representation_dim=representation_dim,
            d_state=d_state,
            expand=expand,
            compress_ratio=compress_ratio,
            squeeze_factor=squeeze_factor,
            mamba_from_trion=mamba_from_trion,
            use_checkpoint=low_vram
        )

    def on_fit_start(self):
        if not self.trainer.is_global_zero:
            return
    
        log_dir = os.path.dirname(self.loss_log)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
        self.loss_log_file = self.loss_log
    
        with open(self.loss_log_file, "w") as f:
            f.write("step,val_loss,content,style,id1,id2,ssim\n")

    def log_to_file(self, line: str):
        if self.loss_log_file is None:
            return
        with open(self.loss_log_file, "a") as f:
            f.write(line)
            
    def forward(self, content, style):
        return self.model(content, style)

    def training_step(self, batch, batch_idx):
        self.flush_validation_loss_log()
        return self.shared_step(batch, 'train_model')

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, 'val')

    def shared_step(self, batch, step):
        content, style = batch['content'], batch['style']
    
        # Batched model calls.
        # Requires SS2D_Decoder.forward_core_S7 to support batched style-dependent A/D.
        Ics = self.model(content, style)
        output_Icc = self.model(content, content)
        output_Iss = self.model(style, style)
    
        content_loss, style_loss, identity_loss1, identity_loss2, ssim_loss = self.loss(
            Ics,
            output_Icc,
            output_Iss,
            content,
            style,
        )
    
        self.log(rf'id_loss1', identity_loss1.item(), prog_bar=step == 'train_model')
        self.log(rf'id_loss2', identity_loss2.item(), prog_bar=step == 'train_model')
    
        self.log(rf'loss_style', style_loss.item(), prog_bar=step == 'train_model')
        self.log(rf'loss_content', content_loss.item(), prog_bar=step == 'train_model')
        self.log(rf'ssim_loss', ssim_loss.item(), prog_bar=step == 'train_model')
    
        total_loss = (
            content_loss +
            style_loss +
            identity_loss1 +
            identity_loss2 +
            ssim_loss
        )
    
        if step == 'val':
            if self.trainer.is_global_zero and self.loss_log_file is not None:
                self.val_loss_buffer.append({
                    "val_loss": total_loss.detach().cpu(),
                    "content": content_loss.detach().cpu(),
                    "style": style_loss.detach().cpu(),
                    "id1": identity_loss1.detach().cpu(),
                    "id2": identity_loss2.detach().cpu(),
                    "ssim": ssim_loss.detach().cpu(),
                })
    
                self.pending_val_write = True
    
            # IMPORTANT: this is what EarlyStopping monitors
            self.log(
                "val_loss",
                total_loss,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
                sync_dist=False,
            )
    
            if not self._printed_val:
                print(
                    f"[VAL] total={total_loss.item():.4f} | "
                    f"content={content_loss.item():.4f} | "
                    f"style={style_loss.item():.4f} | "
                    f"id1={identity_loss1.item():.4f} | "
                    f"id2={identity_loss2.item():.4f}"
                )
                self._printed_val = True
    
            self.validation_step_outputs.append({
                'loss': total_loss,
                'output': Ics,
                'content': content,
                'style': style,
            })
    
            return {
                'loss': total_loss,
                'output': Ics,
            }
    
        return total_loss
    
    def on_validation_epoch_end(self):
        outputs = self.validation_step_outputs
        if self.global_step == 0:
            return

        ##################
        self._printed_val = False
        ##################

        with torch.no_grad():
            imgs = [x['output'] for x in outputs]
            imgs = [img for triple in imgs for img in triple]
            nrow = int(sqrt(len(imgs)))
            grid = make_grid(imgs, nrow=nrow, padding=0)
            logger = self.logger.experiment
            logger.add_image(rf'val_img', grid, global_step=self.global_step + 1)
            self.validation_step_outputs = []

    def loss(self, Ics, output_Icc, output_Iss, content, style):
        content_loss, style_loss, id1_loss, id2_loss, loss_ssim = self.loss_func(Ics, output_Icc, output_Iss, content, style)

        return self.content_weight * content_loss, self.style_weight * style_loss, self.lambda1 * id1_loss, self.lambda2 * id2_loss, loss_ssim * self.ssim_weight

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        def lr_lambda(iter):
            return 1 / (1 + 0.00002 * iter)

        lr_scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
    
    def flush_validation_loss_log(self):
        if not self.pending_val_write:
            return
    
        if self.trainer.sanity_checking:
            self.val_loss_buffer.clear()
            self.pending_val_write = False
            return
    
        if not self.trainer.is_global_zero:
            self.val_loss_buffer.clear()
            self.pending_val_write = False
            return
    
        if self.loss_log_file is None or not self.val_loss_buffer:
            self.val_loss_buffer.clear()
            self.pending_val_write = False
            return
    
        avg = {
            k: mean([x[k].item() for x in self.val_loss_buffer])
            for k in self.val_loss_buffer[0]
        }
    
        with open(self.loss_log_file, "a") as f:
            f.write(
                f"{self.global_step},"
                f"{avg['val_loss']:.6f},"
                f"{avg['content']:.6f},"
                f"{avg['style']:.6f},"
                f"{avg['id1']:.6f},"
                f"{avg['id2']:.6f},"
                f"{avg['ssim']:.6f}\n"
            )
    
        self.val_loss_buffer.clear()
        self.pending_val_write = False
