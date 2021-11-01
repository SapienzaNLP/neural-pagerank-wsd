import os
from argparse import ArgumentParser

from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import DataLoader

from wsd.data.dataset import WordSenseDisambiguationDataset
from wsd.data.processor import Processor
from wsd.models.model import SimpleModel

if __name__ == '__main__':
    parser = ArgumentParser()

    # Add trial name.
    parser.add_argument('--name', type=str, required=True)

    # Add seed arg.
    parser.add_argument('--seed', type=int, default=2021)

    # Add data args.
    parser.add_argument('--train_path', type=str,
                        default='data/preprocessed/glosses/semcor.glosses.untagged.json')
    parser.add_argument('--dev_path', type=str,
                        default='data/preprocessed/semeval2007/semeval2007.json')

    # Data processing
    parser.add_argument('--include_hypernyms',
                        default=True, action='store_true')
    parser.add_argument('--include_hyponyms',
                        default=True, action='store_true')
    parser.add_argument('--include_similar', default=True, action='store_true')
    parser.add_argument('--include_related', default=True, action='store_true')
    parser.add_argument('--include_also_see',
                        default=True, action='store_true')
    parser.add_argument('--include_verb_groups',
                        default=True, action='store_true')
    parser.add_argument('--include_instance_hypernyms', action='store_true')
    parser.add_argument('--include_instance_hyponyms', action='store_true')
    parser.add_argument('--include_pertainyms', action='store_true')
    parser.add_argument('--include_syntag', action='store_true')

    parser.add_argument('--include_pagerank',
                        default=True, action='store_true')
    parser.add_argument('--pagerank_k', type=int, default=10)
    parser.add_argument('--offline_pagerank_path', type=str, default=None)

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--shuffle', action='store_true', default=True)
    parser.add_argument('--num_workers', type=int, default=4)

    # Add syntag & related edges in a graph of its own
    parser.add_argument('--use_syntag_related_graph',
                        default=False, action='store_true')

    # Add checkpoint args.
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    # Add resume from checkpoint path
    parser.add_argument('--resume_from', type=str, default=None)

    # Add model-specific args.
    parser = SimpleModel.add_model_specific_args(parser)

    # Add all the available trainer options to argparse.
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(
        min_epochs=1,
        max_epochs=30,
        gpus=1,
        precision=16,
        gradient_clip_val=5.0,
        row_log_interval=128,
        deterministic=True
    )

    # Store the arguments in hparams.
    hparams = parser.parse_args()
    print(hparams)

    seed_everything(hparams.seed)

    train_dataset = WordSenseDisambiguationDataset(hparams.train_path)
    dev_dataset = WordSenseDisambiguationDataset(hparams.dev_path)

    processor = Processor(
        language_model=hparams.language_model,
        loss_type=hparams.loss_type,
        num_negative_samples=hparams.num_negative_samples,
        include_similar_synsets=hparams.include_similar,
        include_related_synsets=hparams.include_related,
        include_verb_group_synsets=hparams.include_verb_groups,
        include_hypernym_synsets=hparams.include_hypernyms,
        include_hyponym_synsets=hparams.include_hyponyms,
        include_syntags=hparams.include_syntag,
        include_instance_hypernyms_synsets=hparams.include_instance_hypernyms,
        include_instance_hyponyms_synsets=hparams.include_instance_hyponyms,
        include_also_see_synsets=hparams.include_also_see,
        include_pertainyms_synsets=hparams.include_pertainyms,
        include_pagerank_synsets=hparams.include_pagerank,
        pagerank_k=hparams.pagerank_k,
        use_synder=hparams.use_syntag_related_graph,
        offline_pagerank_path=hparams.offline_pagerank_path)

    synset_embeddings = None if not hparams.use_synset_embeddings else processor.load_synset_embeddings(
        hparams.synset_embeddings_path)

    train_dataloader = DataLoader(train_dataset, batch_size=hparams.batch_size,
                                  shuffle=hparams.shuffle,
                                  num_workers=hparams.num_workers,
                                  collate_fn=processor.collate_sentences)

    dev_dataloader = DataLoader(dev_dataset,
                                batch_size=hparams.batch_size,
                                num_workers=hparams.num_workers,
                                collate_fn=processor.collate_sentences)

    # Additional hparams.
    hparams.steps_per_epoch = int(
        len(train_dataset) / (hparams.batch_size * hparams.accumulate_grad_batches)) + 1
    hparams.num_synsets = processor.num_synsets

    model = SimpleModel(hparams,
                        synset_embeddings=synset_embeddings,
                        padding_token_id=processor.padding_token_id)

    model_dir = os.path.join(hparams.checkpoint_dir, hparams.name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    processor_config_path = os.path.join(model_dir, 'processor_config.json')
    model_checkpoint_path = os.path.join(model_dir,
                                         'checkpoint_{val_f1:0.4f}_{epoch:03d}')

    processor.save_config(processor_config_path)
    checkpoint_callback = ModelCheckpoint(filepath=model_checkpoint_path,
                                          monitor='val_f1', mode='max',
                                          save_top_k=2, verbose=True)

    early_stopping_callback = EarlyStopping(monitor='val_f1', patience=5,
                                            verbose=True, mode='max') if hparams.thaw_embeddings_after is None else None

    trainer = Trainer.from_argparse_args(hparams,
                                         checkpoint_callback=checkpoint_callback,
                                         early_stop_callback=early_stopping_callback)

    trainer.fit(model, train_dataloader=train_dataloader,
                val_dataloaders=dev_dataloader)
