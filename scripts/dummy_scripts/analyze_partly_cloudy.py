import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 

# Load the DataFrame
df = pd.read_csv('results.csv')
model = 'MLP'
# Filter the DataFrame for only rows where the model is 'MLP'
model_df = df[df['model'] == model]

# Compute the average score for 'blis' scattering type
blis_avg_score = model_df[model_df['scattering_type'] == 'blis']['score'].mean()
print(f"Average score for 'blis' scattering type with {model} model: {blis_avg_score:.4f}")

blis_avg_stdev = model_df[model_df['scattering_type'] == 'blis']['stdev'].mean()
print(f"Average standard deviation for 'blis' scattering type with {model} model: {blis_avg_stdev:.4f}")

# Compute the average score for 'modulus' scattering type
modulus_avg_score = model_df[model_df['scattering_type'] == 'modulus']['score'].mean()
print(f"Average score for 'modulus' scattering type with {model} model: {modulus_avg_score:.4f}")

modulus_avg_stdev = model_df[model_df['scattering_type'] == 'modulus']['stdev'].mean()
print(f"Average standard deviation for 'modulus' scattering type with {model} model: {modulus_avg_stdev:.4f}")

print(f'Difference is {blis_avg_score - modulus_avg_score}')
print(f'Fractional improvement is {(blis_avg_score - modulus_avg_score)/modulus_avg_score}')

direct_improvements = []
for sub_dataset in range(155):
    blis_score = model_df[(model_df['scattering_type'] == 'blis') & (model_df['sub_dataset'] == sub_dataset)]['score']
    modulus_score = model_df[(model_df['scattering_type'] == 'modulus') & (model_df['sub_dataset'] == sub_dataset)]['score']
    diff = blis_score.item() - modulus_score.item()
    direct_improvements.append(diff)

direct_improvements = np.array(direct_improvements)

import pdb; pdb.set_trace()

# blis_avg_scores = model_df[model_df['scattering_type'] == 'blis']['score']
# modulus_avg_scores = model_df[model_df['scattering_type'] == 'modulus']['score']
# can't plot histogram. need to align by the sub dataset and then compute the difference later
# import pdb; pdb.set_trace()
# improvement = blis_avg_scores - modulus_avg_scores 

# plt.hist(improvement, bins=20, edgecolor='black', alpha=0.7)
# plt.title('Improvement Histogram (blis - modulus)')
# plt.xlabel('Improvement Value')
# plt.ylabel('Frequency')
# plt.grid(True, axis='y', linestyle='--', linewidth=0.7, alpha=0.7)
# plt.tight_layout()

# # Save the histogram
# plt.savefig('improvement_histogram.png')
# plt.show()