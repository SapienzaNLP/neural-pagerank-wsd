from nltk.corpus import wordnet as wn
from prettytable import PrettyTable


if __name__ == '__main__':
    model1_output = '/home/elsheikh/wsd_thesis/output.model_ewier_k10_syntag.gold.txt'
    model2_output = '/home/elsheikh/wsd_thesis/output.k1_no_syntag_big.gold.txt'
    evaluation_input = 'data/original/all/all.gold.key.txt'

    correct, total = 0, 0
    gold = {}
    pred, pred2 = {}, {}
    with open(evaluation_input) as f_gold:
        for line in f_gold:
            instance_id, *gold_senses = line.strip().split()
            if 'semeval2007' in instance_id:
                continue
            gold_synsets = [wn.lemma_from_key(
                s).synset().name() for s in gold_senses]
            gold[instance_id] = gold_synsets
    with open(model1_output) as f_pred:
        for line in f_pred:
            instance_id, pred_synset = line.strip().split()
            if 'semeval2007' in instance_id:
                continue
            pred[instance_id] = pred_synset
    with open(model2_output) as f_pred2:
        for line in f_pred2:
            instance_id, pred_synset = line.strip().split()
            if 'semeval2007' in instance_id:
                continue
            pred2[instance_id] = pred_synset

    model1_pos, model1_neg = {}, {}
    model2_pos, model2_neg = {}, {}
    for instance_id in gold:
        if instance_id not in pred:
            print('Warning: {} not in predictions.'.format(instance_id))
            continue
        predicted_synset = pred[instance_id]
        if predicted_synset in gold[instance_id]:
            model1_pos[instance_id] = predicted_synset
        else:
            model1_neg[instance_id] = predicted_synset

        if instance_id not in pred2:
            print('Warning: {} not in predictions.'.format(instance_id))
            continue
        predicted_synset = pred2[instance_id]
        if predicted_synset in gold[instance_id]:
            model2_pos[instance_id] = predicted_synset
        else:
            model2_neg[instance_id] = predicted_synset

    model1_pos = set(list(model1_pos.keys()))
    model1_neg = set(list(model1_neg.keys()))

    model2_pos = set(list(model2_pos.keys()))
    model2_neg = set(list(model2_neg.keys()))

    mdl1pos_mdl2_pos = len(model1_pos & model2_pos)
    mdl1pos_mdl2_neg = len(model1_pos & model2_neg)

    mdl1neg_mdl2_pos = len(model1_neg & model2_pos)
    mdl1neg_mdl2_neg = len(model1_neg & model2_neg)

    myTable = PrettyTable(
        ["Test 1 \ Test 2", "Test 2 +ve", "Test 2 -ve", "Total"])
    myTable.add_row(["Test 1 +ve", f"{mdl1pos_mdl2_pos}",
                    f"{mdl1pos_mdl2_neg}", f'{mdl1pos_mdl2_pos + mdl1pos_mdl2_neg}'])
    myTable.add_row(["Test 1 -ve", f"{mdl1neg_mdl2_pos}",
                    f"{mdl1neg_mdl2_neg}", f'{mdl1neg_mdl2_pos + mdl1neg_mdl2_neg}'])
    myTable.add_row(["Totals", f"{mdl1neg_mdl2_pos + mdl1pos_mdl2_pos}", f"{mdl1pos_mdl2_neg + mdl1neg_mdl2_neg}",
                    f'{mdl1pos_mdl2_pos + mdl1pos_mdl2_neg + mdl1neg_mdl2_pos + mdl1neg_mdl2_neg}'])
    print(myTable)
