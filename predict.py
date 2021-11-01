from argparse import ArgumentParser

import torch
from torch.utils.data import DataLoader

from wsd.data.dataset import WordSenseDisambiguationDataset
from wsd.data.processor import Processor
from wsd.models.model import SimpleModel


if __name__ == '__main__':
    parser = ArgumentParser()

    # Add data args.
    parser.add_argument('--processor', type=str, required=True)
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--model_input', type=str, required=True)
    parser.add_argument('--model_output', type=str, required=True)

    # Add dataloader args.
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)

    # Other
    parser.add_argument('--device', type=str, default='cuda')

    # Store the arguments in hparams.
    args = parser.parse_args()

    processor = Processor.from_config(args.processor)

    test_dataset = WordSenseDisambiguationDataset(args.model_input)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = SimpleModel.load_from_checkpoint(args.model)
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model.to(device)
    model.eval()

    predictions = {}

    with torch.no_grad():
        for x, _ in test_dataloader:
            x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
            y = model(x)
            batch_predictions = processor.decode(x, y)
            predictions.update(batch_predictions)

    predictions = sorted(list(predictions.items()), key=lambda kv: kv[0])

    with open(args.model_output, 'w') as f:
        for instance_id, synset_id in predictions:
            f.write('{} {}\n'.format(instance_id, synset_id))
