from diff_scripts.data import get_dataloader_ddp
import torch
from accelerate import Accelerator
from torch.optim import Adam
from ema_pytorch import EMA
from pathlib import Path
from tqdm import tqdm
import os
import json
def cycle(dl):
    while True:
        for data in dl:
            yield data

def exists(x):
    return x is not None

def divisible_by(numer, denom):
    return (numer % denom) == 0
class Trainer:
    def __init__(
        self,
        diffusion_model,
        *,
        base_dir,
        batch_size_per_gpu = 4,
        gradient_accumulate_every = 1,
        train_lr = 1e-4,
        train_num_steps = 10000,
        ema_update_every = 10,
        ema_decay = 0.995,
        adam_betas = (0.9, 0.99),
        save_and_sample_every = 100,
        num_samples = 9,
        results_folder = './results',
        amp = False,
        mixed_precision_type = 'fp16',
        split_batches = True,
        max_grad_norm = 1.,
        seed = 0, 
    ):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches = split_batches,
            mixed_precision = mixed_precision_type if amp else 'no'
        )

        # model

        self.model = diffusion_model

        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size_per_gpu = batch_size_per_gpu
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps

        self.max_grad_norm = max_grad_norm

        dl = get_dataloader_ddp(base_dir=base_dir, batch_size_per_gpu=batch_size_per_gpu, num_workers=4, seed=seed)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer
        if isinstance(train_lr, str):
            train_lr = float(train_lr)
        self.opt = Adam(diffusion_model.parameters(), lr = train_lr, betas = adam_betas)

        # for logging results in a folder periodically
        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta = ema_decay, update_every = ema_update_every)
            self.ema.to(self.device)
        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok = True)

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)
        
    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return

        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None,
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'), map_location=device, weights_only=True)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial = self.step, total = self.train_num_steps, disable = not accelerator.is_main_process) as pbar:
            report_loss = 0.
            while self.step < self.train_num_steps:
                self.model.train()

                total_loss = 0.
                
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl)
                    data = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in data.items()}
                    with self.accelerator.autocast():
                        loss_multi, loss_gauss = self.model.module.mixed_loss(data)
                        loss = loss_multi + loss_gauss
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()
                        report_loss += loss.item()
                    self.accelerator.backward(loss)

                if self.step % self.save_and_sample_every == 0 and self.step != 0:
                    pbar.set_description(f'loss: {report_loss / self.save_and_sample_every:.4f}')
                    # 在results_folder 下保存report_loss
                    results_file = os.path.join(self.results_folder, 'training_log.json')
                    if accelerator.is_main_process:
                        if os.path.exists(results_file):
                            with open(results_file, 'r') as f:
                                log_data = json.load(f)
                        else:
                            log_data = []
                        log_data.append({
                            'step': self.step,
                            'loss': report_loss / self.save_and_sample_every
                        })
                        with open(results_file, 'w') as f:
                            json.dump(log_data, f, indent=4)
                    
                    report_loss = 0.

                accelerator.wait_for_everyone()
                accelerator.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and divisible_by(self.step, self.save_and_sample_every):
                        self.ema.ema_model.eval()
                        
                        milestone = f'{self.step}'
                        self.save(milestone)

                pbar.update(1)

        accelerator.print('training complete')