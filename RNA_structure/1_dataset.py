import pandas as pd
import numpy as np

def process_file(filename, start_mol_id):
    df = pd.read_csv(filename)

    df['mol_base'] = df['ID'].apply(lambda x: '_'.join(x.split('_')[:2]))
    df['res_index'] = df['ID'].apply(lambda x: int(x.split('_')[-1]))

    output_rows = []
    global_mol_id = start_mol_id

    for mol_base, group in df.groupby('mol_base'):
        group = group.sort_values('resid').reset_index(drop=True)

        current_segment = []
        prev_resid = None

        for _, row in group.iterrows():
            missing_coords = pd.isna(row['x_1']) or pd.isna(row['y_1']) or pd.isna(row['z_1'])
            resid_break = (prev_resid is not None and row['resid'] != prev_resid + 1)

            if (resid_break or missing_coords) and current_segment:
                for r in current_segment:
                    r['mol_name'] = mol_base
                    r['mol_id'] = global_mol_id
                    output_rows.append(r)
                global_mol_id += 1
                current_segment = []

            if not missing_coords:
                current_segment.append(row.to_dict())

            prev_resid = row['resid']

        if current_segment:
            for r in current_segment:
                r['mol_name'] = mol_base
                r['mol_id'] = global_mol_id
                output_rows.append(r)
            global_mol_id += 1

    final_df = pd.DataFrame(output_rows)
    final_df = final_df[['mol_name', 'mol_id', 'resname', 'resid', 'x_1', 'y_1', 'z_1']]
    return final_df, global_mol_id

# Process both datasets
df1, next_id = process_file('./data/train_labels.csv', start_mol_id=1)
df2, _ = process_file('./data/train_labels.v2.csv', start_mol_id=next_id)

# Merge
merged_df = pd.concat([df1, df2], ignore_index=True)

# Save
merged_df.to_csv('RNA_combined_data.csv', index=False)
