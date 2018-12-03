import pandas as pd
model_dict = {'kNN':        7,
                  'LR':     6,
                  'SVM':    5,
                  'DT':     4,
                  'GNB':    3,
                  'RF':     2,
                  'GBDT':   1
              }

results_df = pd.DataFrame(columns=['Not Scaled (%)', 'Min Max Scaled (%)', 'Std Scaled (%)'],
                          index=list(model_dict.keys()))
results_df.index.name = 'Model'
print(results_df)
