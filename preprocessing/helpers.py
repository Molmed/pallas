# Function to add items to a silo
def add_to_silo(samplesheet_df, items, silo_name):
    # Only set data_silo if it is currently empty
    mask = samplesheet_df.index.isin(items) & (
        samplesheet_df['data_silo'] == '')
    samplesheet_df.loc[mask, 'data_silo'] = silo_name
    return samplesheet_df
