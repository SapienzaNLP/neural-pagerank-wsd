from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from wsd.layers.word_encoder import WordEncoder
from wsd.layers.graph_encoder import GraphEncoder


class SimpleModel(pl.LightningModule):
    def __init__(self, hparams, synset_embeddings=None, padding_token_id=0, padding_target_id=-1):
        super(SimpleModel, self).__init__()
        self.hparams = hparams
        self.padding_token_id = padding_token_id
        self.padding_target_id = padding_target_id
        self.num_synsets = self.hparams.num_synsets

        self.word_encoder = WordEncoder(self.hparams, padding_target_id)
        word_embedding_size = self.word_encoder.word_embedding_size

        if self.hparams.use_graph_convolution:
            self.graph_encoder = GraphEncoder(self.hparams)

        if self.hparams.use_syntag_related_graph:
            self.synder_graph_encoder = GraphEncoder(self.hparams, graph_path='data/synder_graph.json')

        # For predictions on diff models with argument alpha (un)optimized
        try:
            if self.hparams.optimize_alpha:
                self.alpha = torch.nn.Parameter(torch.tensor(0.15), requires_grad=True)
            else:
                self.alpha = self.hparams.alpha
        except (AttributeError, KeyError):
            self.alpha = self.hparams.alpha
        except RuntimeError:
            self.alpha = torch.nn.Parameter(torch.tensor(0.15), requires_grad=True)

        self.synset_scorer = nn.Linear(word_embedding_size, self.num_synsets, bias=False)
        if synset_embeddings is not None:
            with torch.no_grad():
                self.synset_scorer.weight.copy_(synset_embeddings)

    def forward(self, x):
        word_ids = x['word_ids']
        subword_indices = x['subword_indices']
        tokenized_sequence_lengths = x['tokenized_sequence_lengths']
        synset_indices = x['synset_indices']

        word_embeddings = self.word_encoder(
            word_ids, subword_indices=subword_indices, sequence_lengths=tokenized_sequence_lengths)
        word_embeddings = word_embeddings[synset_indices]

        synset_scores = self.synset_scorer(word_embeddings)
        synset_scores /= self.hparams.temperature

        if self.hparams.thaw_embeddings_after and self.current_epoch > self.hparams.thaw_embeddings_after:
            for param in self.synset_scorer.parameters():
                param.requires_grad = True

        if self.hparams.power_iterations > 0:
            k_iter = self.hparams.power_iterations
            curr_logits = synset_scores
            for iteration_ in range(k_iter):
                if self.hparams.use_syntag_related_graph and iteration_ < self.hparams.train_synder_for:
                    curr_logits = (1 - self.alpha) * (self.synder_graph_encoder(curr_logits) + self.graph_encoder(curr_logits)) + (self.alpha * synset_scores)
                else:
                    curr_logits = (
                        1 - self.alpha) * self.graph_encoder(curr_logits) + (self.alpha * synset_scores)
            synset_scores = curr_logits

        return {'synsets': synset_scores}

    def configure_optimizers(self):
        base_parameters = []
        base_parameters.extend(list(self.synset_scorer.parameters()))

        language_model_parameters = []
        for parameter_name, parameter in self.word_encoder.named_parameters():
            if 'word_embedding' not in parameter_name:
                base_parameters.append(parameter)
            # elif 'layer.23' in parameter_name or 'layer.22' in parameter_name or 'layer.21' in parameter_name or 'layer.20' in parameter_name:
            else:
                language_model_parameters.append(parameter)

        optimizer = torch.optim.Adam(
            [
                {
                    'params': base_parameters
                },
                {
                    'params': language_model_parameters,
                    'lr': self.hparams.language_model_learning_rate,
                    'weight_decay': self.hparams.language_model_weight_decay,
                    'correct_bias': False
                },
            ],
            lr=self.hparams.learning_rate,
            # weight_decay=self.hparams.weight_decay
        )
        return optimizer

    def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure=None, using_native_amp=None):
        step = self.trainer.global_step
        warmup_steps = self.hparams.warmup_epochs * self.hparams.steps_per_epoch
        cooldown_steps = warmup_steps + \
            self.hparams.cooldown_epochs * self.hparams.steps_per_epoch
        training_steps = self.hparams.max_epochs * self.hparams.steps_per_epoch

        if step < warmup_steps:
            lr_scale = min(1., float(step + 1) / warmup_steps)
            optimizer.param_groups[0]['lr'] = lr_scale * \
                self.hparams.learning_rate
            optimizer.param_groups[1]['lr'] = lr_scale * \
                self.hparams.language_model_learning_rate

        elif step < cooldown_steps:
            progress = float(step - warmup_steps) / \
                float(max(1, cooldown_steps - warmup_steps))
            lr_scale = (1. - progress)
            optimizer.param_groups[0]['lr'] = self.hparams.min_learning_rate + lr_scale * (
                self.hparams.learning_rate - self.hparams.min_learning_rate)
            optimizer.param_groups[1]['lr'] = self.hparams.language_model_min_learning_rate + lr_scale * (
                self.hparams.language_model_learning_rate - self.hparams.language_model_min_learning_rate)

        else:
            progress = float(step - cooldown_steps) / \
                float(max(1, training_steps - cooldown_steps))
            lr_scale = (1. - progress)
            optimizer.param_groups[0]['lr'] = lr_scale * \
                self.hparams.min_learning_rate
            optimizer.param_groups[1]['lr'] = lr_scale * \
                self.hparams.language_model_min_learning_rate

        # Update params.
        optimizer.step()
        optimizer.zero_grad()

    def training_step(self, batch, batch_index):
        sample, labels = batch
        results = self.shared_step(sample, labels)
        loss = results['loss']

        tensorboard_logs = {
            'train_loss': loss,
        }

        return {
            'loss': loss,
            'log': tensorboard_logs,
        }

    def validation_step(self, batch, batch_index):
        sample, labels = batch
        results = self.shared_step(sample, labels, compute_metrics=True)
        return {
            'loss': results['loss'],
            'metrics': results['metrics'],
        }

    def test_step(self, batch, batch_index):
        sample, labels = batch
        results = self.shared_step(sample, labels, compute_metrics=True)
        return {
            'loss': results['loss'],
            'metrics': results['metrics'],
        }

    def shared_step(self, sample, labels, compute_metrics=False):
        scores = self(sample)

        if self.hparams.loss_type == 'cross_entropy':
            labels['synsets'] = labels['synsets'][1]
            loss = SimpleModel._compute_classification_loss(
                scores['synsets'],
                labels['synsets'],
                self.num_synsets,
                candidates=sample['synset_candidates'] if self.hparams.loss_masking else None,
                negative_samples=labels['negative_samples'] if self.hparams.loss_masking and self.hparams.num_negative_samples > 0 else None,
                ignore_index=self.padding_target_id)

        elif self.hparams.loss_type == 'binary_cross_entropy':
            positive_samples = labels['synsets'].tolist()
            labels['synsets'] = torch.sparse.FloatTensor(
                labels['synsets'], labels['synset_values'], scores['synsets'].size()).to_dense()
            loss = SimpleModel._compute_binary_classification_loss(
                scores['synsets'],
                labels['synsets'],
                candidates=sample['synset_candidates'] if self.hparams.loss_masking else None,
                positive_samples=positive_samples if self.hparams.loss_masking else None,
                negative_samples=labels['negative_samples'] if self.hparams.loss_masking and self.hparams.num_negative_samples > 0 else None)
        else:
            raise ValueError('Unsupported loss type "{}".'.format(
                self.hparams.loss_type))

        if torch.isnan(loss) or not torch.isfinite(loss):
            print('Loss:', loss)
            raise ValueError('NaN loss!')

        metrics = {}
        if compute_metrics:
            metrics = SimpleModel._compute_step_metrics(
                scores,
                labels,
                loss_type=self.hparams.loss_type,
                candidates=sample['synset_candidates'])

        return {
            'loss': loss,
            'metrics': metrics,
        }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = SimpleModel._compute_epoch_metrics(outputs)

        tensorboard_logs = {
            'val_loss': avg_loss,
            'val_wsd_f1': metrics['wsd']['f1'],
            'val_overall_f1': metrics['overall']['f1'],
        }

        return {
            'val_loss': avg_loss,
            'val_f1': metrics['overall']['f1'],
            'val_wsd_f1': metrics['wsd']['f1'],
            'log': tensorboard_logs,
        }

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        metrics = SimpleModel._compute_epoch_metrics(outputs)

        return {
            'test_loss': avg_loss,
            'test_f1': metrics['overall']['f1'],
            'test_wsd_metrics': metrics['wsd'],
            'test_overall_metrics': metrics['overall'],
        }

    @staticmethod
    def _compute_classification_loss(scores, labels, num_classes, candidates=None, negative_samples=None, ignore_index=-1, weight=None):
        if candidates is not None:
            mask = torch.zeros_like(scores) - 50.0
            mask[candidates] = 0.0
            if negative_samples is not None:
                mask[negative_samples] = 0.0
            scores += mask

        classification_loss = F.cross_entropy(
            scores.view(-1, num_classes),
            labels.view(-1),
            weight=weight,
            ignore_index=ignore_index)

        return classification_loss

    @staticmethod
    def _compute_binary_classification_loss(scores, labels, candidates=None, positive_samples=None, negative_samples=None, weight=None):
        if candidates is not None:
            mask = torch.zeros_like(scores) - 50.0
            mask[candidates] = 0.0
            if positive_samples is not None:
                mask[positive_samples] = 0.0
            if negative_samples is not None:
                mask[negative_samples] = 0.0
            scores += mask

        if weight is None:
            classification_loss = F.binary_cross_entropy_with_logits(
                scores,
                labels,
                reduction='mean')
        else:
            classification_loss = F.binary_cross_entropy_with_logits(
                scores,
                labels,
                reduction='none')
            weight_mask = torch.zeros_like(classification_loss)
            weight_mask[candidates] = 1.0
            if positive_samples is not None:
                weight_mask[positive_samples] = weight
            if negative_samples is not None:
                weight_mask[negative_samples] = 1.0
            classification_loss = (classification_loss * weight_mask).mean()

        return classification_loss

    @staticmethod
    def _compute_mse_loss(scores, labels, candidates=None, positive_samples=None, negative_samples=None, weight=None):
        if candidates is not None:
            _scores = torch.zeros_like(scores)
            _scores[candidates] = scores[candidates]
            if positive_samples is not None:
                _scores[positive_samples] = scores[positive_samples]
            if negative_samples is not None:
                _scores[negative_samples] = scores[negative_samples]
            scores = _scores

        if weight is None:
            classification_loss = F.mse_loss(
                scores,
                labels,
                reduction='mean')
        else:
            classification_loss = F.mse_loss(
                scores,
                labels,
                reduction='none')
            weight_mask = torch.zeros_like(classification_loss)
            weight_mask[candidates] = 1.0
            if positive_samples is not None:
                weight_mask[positive_samples] = weight
            if negative_samples is not None:
                weight_mask[negative_samples] = 1. / len(negative_samples[1])
            classification_loss = (classification_loss * weight_mask).mean()

        return classification_loss

    @staticmethod
    def _compute_step_metrics(scores, labels, loss_type, candidates=None):
        if candidates is not None:
            mask = torch.zeros_like(scores['synsets']) - 50.0
            mask[candidates] = 0.0
            scores['synsets'] += mask

        synsets_g = labels['synsets']
        synsets_p = torch.argmax(scores['synsets'], dim=-1)
        if loss_type == 'cross_entropy':
            num_correct_synsets = (
                synsets_p[synsets_g >= 0] == synsets_g[synsets_g >= 0]).sum()
            num_synsets = (synsets_g >= 0).sum()
        else:
            num_correct_synsets = 0
            for g, p in zip(synsets_g, synsets_p):
                if g[p] == 1.0:
                    num_correct_synsets += 1
            num_correct_synsets = torch.as_tensor([num_correct_synsets])
            num_synsets = torch.as_tensor([synsets_g.shape[0]])

        return {
            'num_correct_synsets': num_correct_synsets,
            'num_synsets': num_synsets,
        }

    @staticmethod
    def _compute_epoch_metrics(outputs):
        num_correct_synsets = torch.stack(
            [o['metrics']['num_correct_synsets'] for o in outputs]).sum()
        num_synsets = torch.stack(
            [o['metrics']['num_synsets'] for o in outputs]).sum()
        wsd_f1 = torch.true_divide(num_correct_synsets, num_synsets)
        overall_f1 = wsd_f1

        return {
            'wsd': {
                'f1': wsd_f1,
            },
            'overall': {
                'f1': overall_f1,
            },
        }

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--loss_type', type=str, default='cross_entropy')

        parser.add_argument('--synset_embeddings_path', type=str,
                            default='data/embeddings/synset_embeddings.txt')
        parser.add_argument('--use_synset_embeddings',
                            default=True, action='store_true')
        parser.add_argument('--thaw_embeddings_after', type=int)

        parser.add_argument('--graph_path', type=str,
                            default='data/wn_graph.json')
        parser.add_argument('--use_graph_convolution',
                            default=True, action='store_true')
        parser.add_argument('--use_trainable_graph',
                            default=True, action='store_true')

        parser.add_argument('--loss_masking', default=True,
                            action='store_true')
        parser.add_argument('--temperature', type=float, default=1.0)
        parser.add_argument('--num_negative_samples', type=int, default=64)

        parser.add_argument('--word_projection_size', type=int, default=512)
        parser.add_argument('--word_dropout', type=float, default=0.3)
        parser.add_argument('--language_model', type=str,
                            default='bert-large-cased')
        parser.add_argument('--language_model_fine_tuning',
                            action='store_true')

        parser.add_argument('--warmup_epochs', type=float, default=0.0)
        parser.add_argument('--cooldown_epochs', type=int, default=10)

        parser.add_argument('--learning_rate', type=float, default=5e-4)
        parser.add_argument('--min_learning_rate', type=float, default=1e-7)
        parser.add_argument('--weight_decay', type=float, default=1e-4)

        parser.add_argument('--language_model_learning_rate',
                            type=float, default=1e-5)
        parser.add_argument(
            '--language_model_min_learning_rate', type=float, default=1e-7)
        parser.add_argument('--language_model_weight_decay',
                            type=float, default=1e-4)

        parser.add_argument('--power_iterations', type=int, default=10)
        parser.add_argument('--alpha', type=float, default=0.15)
        parser.add_argument('--optimize_alpha',
                            default=False, action='store_true')

        parser.add_argument('--train_synder_for', type=int, default=1000)

        return parser

    def count_parameters(self):
        from prettytable import PrettyTable
        table = PrettyTable(["Modules", "Parameters"])
        total_params = 0
        for name, parameter in self.named_parameters():
            if not parameter.requires_grad:
                continue
            param = parameter.numel()
            table.add_row([name, param])
            total_params += param
        print(table)
        print(f"Total Trainable Params: {total_params}")
        return total_params
