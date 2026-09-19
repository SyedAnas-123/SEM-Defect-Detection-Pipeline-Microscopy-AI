import pandas as pd

df = pd.read_csv('scratch_exp12b_results.csv')
df.columns = df.columns.str.strip()

p_col = 'metrics/precision(B)'
r_col = 'metrics/recall(B)'
m50_col = 'metrics/mAP50(B)'
m95_col = 'metrics/mAP50-95(B)'

df['fitness'] = 0.1 * df[m50_col] + 0.9 * df[m95_col]
df['f1'] = 2 * (df[p_col] * df[r_col]) / (df[p_col] + df[r_col])

best_fit_idx = df['fitness'].idxmax()
best_fit_row = df.loc[best_fit_idx]

best_map50_idx = df[m50_col].idxmax()
best_map50_row = df.loc[best_map50_idx]

best_recall_idx = df[r_col].idxmax()
best_recall_row = df.loc[best_recall_idx]

best_f1_idx = df['f1'].idxmax()
best_f1_row = df.loc[best_f1_idx]

last_row = df.iloc[-1]

print('=== BEST FITNESS (ULTRALYTICS best.pt checkpoint) ===')
print(f'Epoch: {int(best_fit_row["epoch"])}')
print(f'Precision: {best_fit_row[p_col]:.4f} ({best_fit_row[p_col]*100:.2f}%)')
print(f'Recall: {best_fit_row[r_col]:.4f} ({best_fit_row[r_col]*100:.2f}%)')
print(f'mAP50: {best_fit_row[m50_col]:.4f} ({best_fit_row[m50_col]*100:.2f}%)')
print(f'mAP50-95: {best_fit_row[m95_col]:.4f} ({best_fit_row[m95_col]*100:.2f}%)')
print(f'F1: {best_fit_row["f1"]:.4f} ({best_fit_row["f1"]*100:.2f}%)')
print(f'Fitness: {best_fit_row["fitness"]:.4f}')

print('\n=== BEST mAP50 EPOCH ===')
print(f'Epoch: {int(best_map50_row["epoch"])}, mAP50: {best_map50_row[m50_col]:.4f}, Recall: {best_map50_row[r_col]:.4f}, Precision: {best_map50_row[p_col]:.4f}, F1: {best_map50_row["f1"]:.4f}')

print('\n=== BEST RECALL EPOCH ===')
print(f'Epoch: {int(best_recall_row["epoch"])}, Recall: {best_recall_row[r_col]:.4f}, Precision: {best_recall_row[p_col]:.4f}, mAP50: {best_recall_row[m50_col]:.4f}, F1: {best_recall_row["f1"]:.4f}')

print('\n=== BEST F1 EPOCH ===')
print(f'Epoch: {int(best_f1_row["epoch"])}, F1: {best_f1_row["f1"]:.4f}, Precision: {best_f1_row[p_col]:.4f}, Recall: {best_f1_row[r_col]:.4f}, mAP50: {best_f1_row[m50_col]:.4f}')

print('\n=== LAST RECORDED EPOCH (EPOCH 182) ===')
print(f'Precision: {last_row[p_col]:.4f}, Recall: {last_row[r_col]:.4f}, mAP50: {last_row[m50_col]:.4f}, mAP50-95: {last_row[m95_col]:.4f}, F1: {last_row["f1"]:.4f}')
