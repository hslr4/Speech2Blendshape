import argparse
from data_loader import get_dataloaders
import torch
import metrics
from predict import load_model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/beat")
    parser.add_argument("--test_subjects", type=str, default="5 10 20 30")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model_path", type=str, default="ANONYMIZED-FOR-REVIEW/speech2blendshape")
    parser.add_argument("--epoch", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")

    args = parser.parse_args()
    return args

# Mean values for each blendshape to predict baseline (all values)
label_means = np.array([[[0.06682316,0.06681694,0.25371084,0.12956057,0.12956651,0.02150766
 ,0.17319861,0.18742893,0.07452572,0.07430915,0.08153474,0.08158347
 ,0.1344843,0.09138267,0.03961422,0.07395752,0.10274948,0.10300316
 ,0.25183038,0.2521907,0.21227364,0.2121901,0.12837101,0.02263843
 ,0.19499805,0.01192453,0.12104702,0.12321976,0.11982337,0.00546072
 ,0.00777778,0.23678,0.01196589,0.33497216,0.33919418,0.04842478
 ,0.05234776,0.06621926,0.00688575,0.05919219,0.06104041,0.04430697
 ,0.22396134,0.22447881,0.20307746,0.26737597,0.26826902,0.11037954
 ,0.10609268,0.15481554,0.16048624]]])

# train data with train_subjects: "1 2 3 4 6 7 8 9 11 12 13 14 15 16 17 18 19 21 22 23 24 25 26 27 28 29"
label_means = np.array([[[0.05939229, 0.05945967, 0.22304964, 0.10572807, 0.10567163, 0.02447005,
   0.16437491, 0.17466197, 0.09918842, 0.09888752, 0.07283838, 0.07278535,
   0.04451876, 0.14229834, 0.06494,    0.01367886, 0.10990832, 0.1095022,
   0.23656385, 0.23673449, 0.18043641, 0.18017701, 0.14415655, 0.01278421,
   0.18945965, 0.01465326, 0.1157396,  0.13236836, 0.13155622, 0.00418604,
   0.00928174, 0.268034,   0.01380929, 0.2800457,  0.29206638, 0.05789116,
   0.06163813, 0.07747735, 0.00708962, 0.04835804, 0.05687925, 0.05853164,
   0.2524513,  0.22409583, 0.18362966, 0.28361953, 0.30309623, 0.1685067,
   0.15495914, 0.19970551, 0.19586541]]])

blink_left_idx = 8
blink_right_idx = 9
jaw_idx = 24

def test_loop(dataloader, model):
    model.eval()
    num_batches = len(dataloader)
    mae = 0
    mouth_mae = 0
    mse = 0
    mouth_mse = 0
    rae = 0
    blink_preds_left = []
    blink_preds_right = []
    jaw_preds = []

    with torch.no_grad():
        for X, y, subj, emo in dataloader:
            # Baseline predicting mean for each blendshape: (comment out 3 lines after)
            # pred = label_means.repeat([y.shape[1]], axis=1)
            # pred = pred.repeat([y.shape[0]], axis=0)
            pred = model(X, subj, emo)
            pred, y = pred.cpu(), y.cpu()
            y = y.cpu().numpy()
            mae += metrics.mae(pred, y)
            mouth_mae += metrics.masked_mae(pred, y, metrics.MOUTH_MASK)
            mse += metrics.mse(pred, y)
            mouth_mse += metrics.masked_mse(pred, y, metrics.MOUTH_MASK)
            rae += metrics.rae(pred,y)
            pred = pred.numpy()
            jaw_preds.extend(pred[:,:,jaw_idx].flatten())
            blink_preds_left.extend(pred[:,:,blink_left_idx].flatten())
            blink_preds_right.extend(pred[:,:,blink_right_idx].flatten())
    mae /= num_batches
    mouth_mae /= num_batches
    mse /= num_batches
    mouth_mse /= num_batches
    rae /= num_batches
    jaw_var = np.var(np.array(jaw_preds))
    blink_left_var = np.var(np.array(blink_preds_left))
    blink_right_var = np.var(np.array(blink_preds_right))
    print(f"Test Metrics: \n MAE: {mae:>8f} \n Mouth MAE: {mouth_mae:>8f} \n MSE: {mse:>8f} \n Mouth MSE: {mouth_mse:>8f} \n RAE: {rae} \n Jaw variance: {jaw_var} \n Blink left variance: {blink_left_var} \n Blink right variance: {blink_right_var}")
    return [mae.item(), mouth_mae.item(), mse, mouth_mse, rae.item(), jaw_preds, blink_preds_left, blink_preds_right]


if __name__ == "__main__":
    args = parse_args()

    model = load_model(args.model_path, args.device)
    
    metric_rows = [['epoch', 'train_loss', 'test_loss', 'mae', 'mouth_mae', 'mse', 'mouth_mse']]
    preds_df = None
    strategies = ['random', 'zeros', 'ones', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', ['6','7','8','9'], ['2','4','6','8']]
    strategies_str = [''.join(emb) if isinstance(emb, list) else emb for emb in strategies]
    mets_df = pd.DataFrame(index=strategies_str, columns=['mae', 'mouth_mae', 'rae'])
    for i, subj_emb_strategy in enumerate(strategies): 
        print(f"Subject embedding strategy: {subj_emb_strategy}")
        data_loaders, n_emos = get_dataloaders(
            args.data_path, 
            model.train_subjects, 
            model.train_subjects, # not optimal
            args.test_subjects,
            args.batch_size, 
            subj_emb_strategy,
            args.device
        )
        test_metrics = test_loop(data_loaders['test'], model)
        df = pd.DataFrame({'jaw': test_metrics[5], 'blink_left': test_metrics[6], 'blink_right': test_metrics[7]})
        df['emb'] = strategies_str[i]

        if preds_df is None:
            preds_df = df
        else:
            preds_df = pd.concat([preds_df, df])

        mets_df.loc[strategies_str[i]] = [test_metrics[0], test_metrics[1], test_metrics[4]]

    boxplot = preds_df.boxplot(by='emb', return_type=None, figsize=(10,10))
    plt.savefig(f'./test_metrics/prediction_boxplots_{args.model_path.split("/")[-1]}.png')

    barchart = mets_df.plot.bar(subplots=True)
    plt.savefig(f'./test_metrics/metrics_barchart_{args.model_path.split("/")[-1]}.png')

    print('Metric means across different subject embedding strategies:')
    print(mets_df.mean())