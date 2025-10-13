# Function to add items to a silo
def add_to_silo(samplesheet_df, items, silo_name, force=False):
    if force:
        # Set data_silo to silo_name for all items
        mask = samplesheet_df.index.isin(items)
    else:
        # Only set data_silo if it is currently empty
        mask = samplesheet_df.index.isin(items) & (
            samplesheet_df['data_silo'] == '')
    samplesheet_df.loc[mask, 'data_silo'] = silo_name
    return samplesheet_df


def silo_unlabeled(samplesheet_df):
    unknown_rows = samplesheet_df['subtype'].str.endswith(
        ('other', 'ALLNOS'), na=False)

    if unknown_rows.any():
        samplesheet_df = add_to_silo(samplesheet_df,
                                     samplesheet_df[unknown_rows].index,
                                     'unlabeled')
    return samplesheet_df


def silo_excluded(samplesheet_df):
    excluded_rows = samplesheet_df['subtype'].str.contains(
        'EXCLUDED', na=False)
    if excluded_rows.any():
        samplesheet_df = add_to_silo(
            samplesheet_df, samplesheet_df[excluded_rows].index, 'excluded')
    return samplesheet_df


def silo_unrecognized(samplesheet_df):
    unrecognized_rows = samplesheet_df['subtype'].str.contains(
        'UNRECOGNIZED', na=False)
    if unrecognized_rows.any():
        samplesheet_df = add_to_silo(
            samplesheet_df, samplesheet_df[unrecognized_rows].index,
            'unrecognized')
    return samplesheet_df


def silo_remaining(samplesheet_df, silo_name='labeled_diagnostic'):
    # Set all items whose data_silo is still empty to the given silo_name
    mask = samplesheet_df['data_silo'] == ''
    samplesheet_df.loc[mask, 'data_silo'] = silo_name
    return samplesheet_df
