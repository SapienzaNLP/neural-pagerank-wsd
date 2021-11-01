def fetch_best_model(ckpt_dir):
    """
    Utility function to help make prediction script even easier
    just enter the experiment checkpoints directory and it returns
    the best model
    """
    import os
    ckpt_files = os.listdir(ckpt_dir)
    val_scores = [float(filename[18:-15]) for filename in ckpt_files]
    best_score = max(val_scores)
    best_epoch = [filename[31:-5]
                  for filename in ckpt_files if float(filename[18:-15]) == best_score]
    best_model_name = f'checkpoint_val_f1={best_score}_epoch={best_epoch[0]}.ckpt'
    return os.path.join(ckpt_dir, best_model_name)


def write_predictions(best_model, args):
    processor = Processor.from_config(args.processor)

    test_dataset = WordSenseDisambiguationDataset(args.model_input)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=processor.collate_sentences)

    model = SimpleModel.load_from_checkpoint(best_model)
    device = 'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    model.to(device)
    model.eval()

    predictions = {}

    with torch.no_grad():
        for x, _ in test_dataloader:
            x = {k: v.to(device) if not isinstance(
                v, list) else v for k, v in x.items()}
            y = model(x)
            batch_predictions = processor.decode(x, y)
            predictions.update(batch_predictions)

    predictions = sorted(list(predictions.items()), key=lambda kv: kv[0])

    with open(args.model_output, 'w') as f:
        for instance_id, synset_id in predictions:
            f.write('{} {}\n'.format(instance_id, synset_id))



def evaluate_model(best_model, processor, model_input,
                   model_output, evaluation_input,
                   batch_size=32, num_workers=4, device='cuda'):
    import torch
    processor = Processor.from_config(processor)
    test_dataset = WordSenseDisambiguationDataset(model_input)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=processor.collate_sentences)

    model = SimpleModel.load_from_checkpoint(best_model)
    model.to(device)
    model.eval()

    predictions = {}
    with torch.no_grad():
        for x, _ in test_dataloader:
            x = {k: v.to(device) if not isinstance(
                v, list) else v for k, v in x.items()}
            y = model(x)
            batch_predictions = processor.decode(x, y)
            predictions.update(batch_predictions)

    predictions = sorted(list(predictions.items()), key=lambda kv: kv[0])

    with open(model_output, 'w') as f:
        for instance_id, synset_id in predictions:
            f.write('{} {}\n'.format(instance_id, synset_id))

    correct, total = 0, 0
    gold = {}
    pred = {}
    with open(evaluation_input) as f_gold:
        for line in f_gold:
            instance_id, *gold_senses = line.strip().split()
            gold_synsets = [wn.lemma_from_key(
                s).synset().name() for s in gold_senses]
            gold[instance_id] = gold_synsets
    with open(model_output) as f_pred:
        for line in f_pred:
            instance_id, pred_synset = line.strip().split()
            pred[instance_id] = pred_synset

    pos_correct = {
        'NOUNs': 0,
        'VERBs': 0,
        'ADJs': 0,
        'ADVs': 0,
    }

    pos_total = {
        'NOUNs': 0,
        'VERBs': 0,
        'ADJs': 0,
        'ADVs': 0,
    }

    for instance_id in gold:
        if instance_id not in pred:
            print('Warning: {} not in predictions.'.format(instance_id))
            continue
        total += 1
        predicted_synset = pred[instance_id]
        pos = predicted_synset.split('.')[1]
        if pos == 'n':
            pos_total['NOUNs'] += 1
        elif pos == 'v':
            pos_total['VERBs'] += 1
        elif pos == 'r':
            pos_total['ADVs'] += 1
        elif pos == 'a' or pos == 's':
            pos_total['ADJs'] += 1
        if predicted_synset in gold[instance_id]:
            correct += 1
            if pos == 'n':
                pos_correct['NOUNs'] += 1
            elif pos == 'v':
                pos_correct['VERBs'] += 1
            elif pos == 'r':
                pos_correct['ADVs'] += 1
            elif pos == 'a' or pos == 's':
                pos_correct['ADJs'] += 1

    print()
    print('Accuracy    = {:0.3f}% ({}/{})'.format(100. *
          correct / total, correct, total))
    if pos_total['NOUNs'] > 0:
        print('NOUNs       = {:0.3f}% ({}/{})'.format(
            100. * pos_correct['NOUNs'] / pos_total['NOUNs'], pos_correct['NOUNs'], pos_total['NOUNs']))
    if pos_total['VERBs'] > 0:
        print('VERBs       = {:0.3f}% ({}/{})'.format(
            100. * pos_correct['VERBs'] / pos_total['VERBs'], pos_correct['VERBs'], pos_total['VERBs']))
    if pos_total['ADJs'] > 0:
        print('ADJs        = {:0.3f}% ({}/{})'.format(
            100. * pos_correct['ADJs'] / pos_total['ADJs'], pos_correct['ADJs'], pos_total['ADJs']))
    if pos_total['ADVs'] > 0:
        print('ADVs        = {:0.3f}% ({}/{})'.format(
            100. * pos_correct['ADVs'] / pos_total['ADVs'], pos_correct['ADVs'], pos_total['ADVs']))
    print()


def test_trained_mdl_performance(ckpt_dir, processor, model_input,
                                 model_output, evaluation_input):
    best_model = fetch_best_model(ckpt_dir)
    test_set = evaluation_input.split()[-1]
    print(f"****** Evaluating {best_model} model on {test_set} test set ******")
    write_predictions()
    evaluate_model(best_model, processor, model_input,
                   model_output, evaluation_input)


