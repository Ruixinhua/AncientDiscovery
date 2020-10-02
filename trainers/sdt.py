# -*- coding: utf-8 -*-
# @Organization  : BDIC
# @Author        : Liu Dairui
# @Time          : 2020/4/28 17:45
# @Function      : This is the class of single decoder trainer

from trainers.base_trainer import BaseTrainer
from util import loss_helper, model_helper


class SingleDecoderTrainer(BaseTrainer):

    def _train_epoch(self, **kwargs):
        target_recon_loss, source_recon_loss, dis_loss_all, count = 0.0, 0.0, 0.0, 0.0
        target_data, source_data, labels = self.dataset.random_data(self.target_data, self.source_data, self.labels)
        for target_batch, source_batch, label_batch in zip(target_data, source_data, labels):
            target_batch, source_batch = target_batch.to(self.config.device), source_batch.to(self.config.device)
            if not self.add_cons:
                # run a batch of data
                target_code, target_loss = model_helper.run_batch(self.model, target_batch, self.core)
                self._backward(self.optimizer, target_loss)
                source_code, source_loss = model_helper.run_batch(self.model, source_batch, self.core)
                self._backward(self.source_optimizer, source_loss)
                dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)
            else:
                # Here target models is equal to source models
                target_code, target_loss = model_helper.run_batch(self.model, target_batch, self.core)
                source_code, source_loss = model_helper.run_batch(self.model, source_batch, self.core)
                dis_loss = loss_helper.cal_dis_err(target_code, source_code, label_batch, criterion=self.criterion)

                # add weight here and balance distance weight
                target_weight, source_weight, = 1, 1,
                dis_loss = (target_loss / dis_loss + source_loss / dis_loss) / 2 * dis_loss
                combined_loss = target_weight * target_loss + source_weight * source_loss + dis_loss
                self._backward(self.optimizer, combined_loss)

            # calculate loss here
            count += len(label_batch)
            target_recon_loss += target_loss.item()
            source_recon_loss += source_loss.item()
            dis_loss_all += dis_loss.item()
        return {"Target reconstruct loss": target_recon_loss / count,
                "Source reconstruct loss": source_recon_loss / count, "Distance loss": dis_loss_all / count}
