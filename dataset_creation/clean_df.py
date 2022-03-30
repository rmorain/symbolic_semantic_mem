def clean_df(df):
    # Remove disambiguation pages
    text_filter = df.knowledge.str.contains("disambiguation page")
    text_filter = ~text_filter
    df = df[text_filter]

    # Remove numbers
    text_filter = df.knowledge.str.contains("playing card")
    text_filter = ~text_filter
    df = df[text_filter]

    # Reset index
    df = df.reset_index(drop=True)
    return df
