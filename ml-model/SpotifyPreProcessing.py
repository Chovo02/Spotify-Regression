def DataPreProcessing(df):
    def tempo_classifier(x):
        if x < percentile_33:
            return 1
        elif x > percentile_33 and x < percentile_66:
            return 2
        elif x > percentile_66:
            return 3
    
    df.drop_duplicates(subset=['track_id'], keep='first', inplace=True)
    df.dropna(inplace=True)
    df["duration_ms"] = df["duration_ms"]/60000
    df.rename(columns={"duration_ms": "duration_min"}, inplace=True)
    df = df[df["tempo"]>0]
    df = df[df["time_signature"]>0]
    df = df[df["duration_min"]>0]
    df = df[df["duration_min"]<=10]
    df = df[df["popularity"]>0]
    df = df[df["popularity"]<=100]

    percentile_33 = df["tempo"].quantile(0.33)
    percentile_66 = df["tempo"].quantile(0.66)

    df["tempo"] = df["tempo"].apply(lambda x: tempo_classifier(x))
    df.drop(["mode", "key", "time_signature"], axis=1, inplace=True)
    return df