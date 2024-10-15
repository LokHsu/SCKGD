from datetime import datetime

import torch
import wandb

from params import args


class WandbLogger():
    def __init__(self):
        self.best_perf = 0
        self.runs = f'run-{datetime.now():%Y-%m-%d_%H-%M-%S}'
        wandb.init(
            project='SCKGD',
            name=self.runs,
            config=args,
        )

    def log_metrics(self, epoch, loss, res, gcn):
        wandb.log({'epoch': epoch, 'loss': loss, 'hr': res['hr'], 'ndcg': res['ndcg']})
        if  self.best_perf < sum(res.values()):
            self.best_perf = sum(res.values())
            wandb.run.summary['epoch(Best)'] = epoch
            wandb.run.summary['hr(Best)'] = res['hr']
            wandb.run.summary['ndcg(Best)'] = res['ndcg']
            
            if args.save_model:
                torch.save(gcn.state_dict(), f'{args.param_path}/{self.runs}.pth')
        
        if epoch == args.epochs:
            self.finish()

    def finish(self):
        wandb.finish()